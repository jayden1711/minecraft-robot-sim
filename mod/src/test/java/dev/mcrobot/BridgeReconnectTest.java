package dev.mcrobot;

import static org.junit.jupiter.api.Assertions.*;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import org.junit.jupiter.api.Test;

class BridgeReconnectTest {
    private static Socket accept(ServerSocket ss) throws IOException {
        Socket c = ss.accept();
        DataInputStream in = new DataInputStream(c.getInputStream());
        assertEquals(Frames.HELLO, Frames.read(in).type());
        c.getOutputStream().write(Frames.encode(Frames.HELLO_ACK, "{\"proto\":2}"));
        return c;
    }

    private static void waitFor(java.util.function.BooleanSupplier c) throws InterruptedException {
        for (int i = 0; i < 100 && !c.getAsBoolean(); i++) Thread.sleep(50);
        assertTrue(c.getAsBoolean());
    }

    @Test
    void reconnectsAfterTrainerRestartAndChatsOnlyOnChange() throws Exception {
        ServerSocket ss = new ServerSocket(0);
        int port = ss.getLocalPort();
        Bridge b = new Bridge(new Bridge.Config("127.0.0.1", port, ""));
        List<String> chat = new CopyOnWriteArrayList<>();
        b.setChat(chat::add);
        b.start(100);

        Socket first = accept(ss);
        waitFor(b::isConnected);
        first.close(); // trainer goes away
        waitFor(() -> !b.isConnected());
        ss.close();
        Thread.sleep(500); // several failed retries: must stay silent
        ServerSocket ss2 = new ServerSocket(port); // trainer restarted on the same port
        Socket second = accept(ss2);
        waitFor(b::isConnected);

        assertEquals(3, chat.size(), chat.toString());
        assertTrue(chat.get(0).startsWith("Connected to trainer"));
        assertEquals("Trainer disconnected", chat.get(1));
        assertTrue(chat.get(2).startsWith("Connected to trainer"));
        second.close();
        b.shutdown();
        ss2.close();
    }
}
