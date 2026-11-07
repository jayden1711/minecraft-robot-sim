package dev.mcrobot;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class FramesTest {
    private static Frames.Frame read(byte[] b) throws IOException {
        return Frames.read(new DataInputStream(new ByteArrayInputStream(b)));
    }

    @Test
    void goldenBytesMatchPython() {
        // same bytes asserted in server/tests/test_protocol.py::test_golden_header_bytes
        assertArrayEquals(new byte[] {'M', 'R', 1, 1, 0, 0, 0, 2, 'x', 'y'}, Frames.encode(Frames.HELLO, "xy"));
    }

    @Test
    void roundTrip() throws IOException {
        Frames.Frame f = read(Frames.encode(Frames.PING, "{\"n\":1}"));
        assertEquals(Frames.PING, f.type());
        assertEquals("{\"n\":1}", f.text());
    }

    @Test
    void rejectsBadMagicVersionAndLength() {
        assertThrows(IOException.class, () -> read(new byte[] {'X', 'X', 1, 3, 0, 0, 0, 0}));
        assertThrows(IOException.class, () -> read(new byte[] {'M', 'R', 9, 3, 0, 0, 0, 0}));
        assertThrows(IOException.class, () -> read(new byte[] {'M', 'R', 1, 3, (byte) 0xff, (byte) 0xff, (byte) 0xff, (byte) 0xff}));
    }

    @Test
    void truncatedIsEof() {
        byte[] b = Frames.encode(Frames.PING, "abcdef");
        assertThrows(java.io.EOFException.class, () -> read(Arrays.copyOf(b, b.length - 2)));
    }

    @Test
    void regionLayout() throws IOException {
        byte[] p = Frames.encodeRegion("{}", new int[] {0, 1, 2, 3});
        assertEquals(2, p[3]); // header length, big-endian u32
        assertTrue(p.length > 6);
    }
}
