package dev.mcrobot;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * Client side of the socket to the Python server. All socket I/O runs off the server thread. After start() a daemon
 * thread reconnects whenever the trainer is away (it restarts on its own schedule); chat only says "connected" and
 * "disconnected", once per change, never per retry.
 */
public final class Bridge {
    private static final Gson GSON = new Gson();

    private final Config config;
    private final ExecutorService io = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "mcrobot-io");
        t.setDaemon(true);
        return t;
    });
    private volatile Socket socket;
    private volatile Consumer<String> chat = s -> {};
    private volatile Thread reader;
    private volatile boolean connected; // what chat last said
    private volatile boolean closed;
    private volatile long reconnectMillis = 3000;
    private String lastFailure = ""; // io thread only; keeps the log to one line per distinct failure

    public record Config(String host, int port, String token) {}

    public Bridge(Config config) {
        this.config = config;
    }

    /** Where incoming STATUS/ERROR/ACK text goes (must be thread-safe; McRobotMod hops to the server thread). */
    public void setChat(Consumer<String> chat) {
        this.chat = chat;
    }

    public boolean isConnected() {
        Socket s = socket;
        return s != null && !s.isClosed();
    }

    /** Starts the background reconnect loop. */
    public void start(long reconnectMillis) {
        this.reconnectMillis = reconnectMillis;
        Thread t = new Thread(() -> {
            while (!closed) {
                if (!isConnected()) io.execute(this::tryConnect);
                try {
                    Thread.sleep(this.reconnectMillis);
                } catch (InterruptedException e) {
                    return;
                }
            }
        }, "mcrobot-reconnect");
        t.setDaemon(true);
        t.start();
    }

    private void tryConnect() {
        try {
            ensureConnected();
            lastFailure = "";
        } catch (IOException e) {
            String f = String.valueOf(e.getMessage());
            if (!f.equals(lastFailure)) McRobotMod.LOG.info("trainer not reachable at {}:{}: {} (retrying every {} s)", config.host(), config.port(), f, reconnectMillis / 1000);
            lastFailure = f;
        }
    }

    public void submit(Runnable r) {
        io.execute(() -> {
            try {
                ensureConnected();
                r.run();
            } catch (IOException e) {
                closeQuietly();
                chat.accept("Cannot reach trainer at " + config.host() + ":" + config.port() + ": " + e.getMessage());
            }
        });
    }

    public void send(int type, byte[] payload) {
        try {
            Socket s = socket;
            if (s == null) throw new IOException("not connected");
            OutputStream out = s.getOutputStream();
            synchronized (this) {
                out.write(Frames.encode(type, payload));
                out.flush();
            }
        } catch (IOException e) {
            closeQuietly();
            disconnected("send failed: " + e.getMessage());
        }
    }

    public void sendJson(int type, JsonObject o) {
        send(type, GSON.toJson(o).getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    private void ensureConnected() throws IOException {
        if (socket != null && !socket.isClosed()) return;
        Socket s = new Socket();
        s.connect(new InetSocketAddress(config.host(), config.port()), 2000);
        s.setSoTimeout(5000);
        JsonObject hello = new JsonObject();
        hello.add("proto", GSON.toJsonTree(new int[] {Frames.APP_PROTO_MIN, Frames.APP_PROTO_MAX}));
        hello.addProperty("mod", "mcrobot " + McRobotMod.VERSION);
        hello.addProperty("token", config.token());
        s.getOutputStream().write(Frames.encode(Frames.HELLO, GSON.toJson(hello)));
        DataInputStream in = new DataInputStream(s.getInputStream());
        Frames.Frame ack = Frames.read(in);
        if (ack.type() != Frames.HELLO_ACK) {
            s.close();
            throw new IOException("handshake refused: " + ack.text());
        }
        s.setSoTimeout(0);
        socket = s;
        Thread t = new Thread(() -> readLoop(s, in), "mcrobot-reader");
        t.setDaemon(true);
        reader = t;
        t.start();
        if (!connected) {
            connected = true;
            chat.accept("Connected to trainer: " + ack.text());
        }
    }

    /** Reports the loss once (reader, send and explicit paths can all notice it) and clears the ghosts. */
    private void disconnected(String why) {
        McRobotMod.ghosts().requestClear("trainer disconnected");
        if (connected) {
            connected = false;
            chat.accept("Trainer disconnected");
            McRobotMod.LOG.info("trainer disconnected: {}", why);
        }
    }

    private void readLoop(Socket s, DataInputStream in) {
        String why = "closed";
        try {
            while (!s.isClosed()) {
                Frames.Frame f = Frames.read(in);
                switch (f.type()) {
                    case Frames.PONG, Frames.REGION_ACK, Frames.STATUS, Frames.ERROR -> chat.accept(describe(f));
                    case Frames.GHOST_PARTS -> McRobotMod.ghosts().offer(f.payload());
                    case Frames.GHOSTS -> {} // pose-only ghosts are unused; M1 sends GHOST_PARTS
                    default -> chat.accept("Unexpected frame type " + f.type());
                }
            }
        } catch (IOException e) {
            why = String.valueOf(e.getMessage());
        } finally {
            boolean current = socket == s;
            if (current) socket = null;
            try {
                s.close();
            } catch (IOException ignored) {
            }
            if (current && !closed) disconnected(why);
        }
    }

    private static String describe(Frames.Frame f) {
        String t = f.text();
        try {
            JsonObject o = GSON.fromJson(t, JsonObject.class);
            if (f.type() == Frames.STATUS && o.has("text")) return o.get("text").getAsString();
            if (f.type() == Frames.ERROR) return "Trainer error: " + o.get("message").getAsString();
        } catch (RuntimeException ignored) {
        }
        return (f.type() == Frames.PONG ? "pong " : f.type() == Frames.REGION_ACK ? "export ack " : "") + t;
    }

    private void closeQuietly() {
        Socket s = socket;
        socket = null;
        if (s != null) {
            try {
                s.close();
            } catch (IOException ignored) {
            }
        }
    }

    public void shutdown() {
        closed = true;
        Socket s = socket;
        if (s != null) {
            try {
                s.getOutputStream().write(Frames.encode(Frames.BYE, "{}"));
            } catch (IOException ignored) {
            }
        }
        closeQuietly();
        io.shutdownNow();
    }
}
