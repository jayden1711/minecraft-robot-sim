package dev.mcrobot;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.google.gson.JsonObject;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/** Real Bridge against the real Python server. Runs only when MCROBOT_PY (python with mcrobot deps) is set. */
class InteropTest {
    @Test
    void handshakePingAndRegion() throws Exception {
        String py = System.getenv("MCROBOT_PY");
        assumeTrue(py != null, "MCROBOT_PY not set");
        Path out = Files.createTempDirectory("mcrobot-interop");
        ProcessBuilder pb = new ProcessBuilder(py, "-m", "mcrobot.server", "--port", "0", "--out", out.toString());
        pb.environment().put("PYTHONPATH", System.getenv().getOrDefault("MCROBOT_PYPATH", "../server"));
        pb.environment().put("MCROBOT_TOKEN", "t0k");
        pb.redirectErrorStream(true);
        Process proc = pb.start();
        try {
            BufferedReader r = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = r.readLine();
            assertNotNull(line, "server printed nothing");
            int port = Integer.parseInt(line.substring(line.lastIndexOf(':') + 1).trim());

            BlockingQueue<String> chat = new LinkedBlockingQueue<>();
            Bridge b = new Bridge(new Bridge.Config("127.0.0.1", port, "t0k"));
            b.setChat(chat::add);
            JsonObject ping = new JsonObject();
            ping.addProperty("n", 7);
            b.submit(() -> b.sendJson(Frames.PING, ping));
            assertTrue(next(chat, "Connected").contains("proto"));
            assertTrue(next(chat, "pong").contains("\"n\":7"));

            String header = "{\"origin\":[0,64,0],\"size\":[2,1,2],\"palette\":["
                    + "{\"name\":\"minecraft:air\",\"shape\":\"empty\"},{\"name\":\"minecraft:stone\",\"shape\":\"full\"}]}";
            byte[] payload = Frames.encodeRegion(header, new int[] {1, 1, 1, 1});
            b.submit(() -> b.send(Frames.REGION, payload));
            String ack = next(chat, "export ack");
            assertTrue(ack.contains("\"collider_boxes\":1"), ack);
            assertTrue(ack.contains("\"solid_cells\":4"), ack);
            b.shutdown();
        } finally {
            proc.destroy();
        }
    }

    private static String next(BlockingQueue<String> q, String prefix) throws InterruptedException {
        for (int i = 0; i < 10; i++) {
            String s = q.poll(5, TimeUnit.SECONDS);
            assertNotNull(s, "timed out waiting for " + prefix);
            if (s.startsWith(prefix)) return s;
        }
        return fail("no message starting with " + prefix);
    }
}
