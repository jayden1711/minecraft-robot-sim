package dev.mcrobot;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

/** Wire framing, mirror of server/mcrobot/protocol.py. No Minecraft dependencies, so it is unit-testable. */
public final class Frames {
    public static final int HELLO = 1, HELLO_ACK = 2, PING = 3, PONG = 4, REGION = 5, REGION_ACK = 6,
            COMMAND = 7, STATUS = 8, GHOSTS = 9, ERROR = 10, BYE = 11;
    public static final int FRAMING_VERSION = 1;
    public static final int APP_PROTO_MIN = 1, APP_PROTO_MAX = 1;
    public static final int MAX_FRAME = 64 * 1024 * 1024;

    public record Frame(int type, byte[] payload) {
        public String text() {
            return new String(payload, StandardCharsets.UTF_8);
        }
    }

    private Frames() {}

    public static byte[] encode(int type, byte[] payload) {
        if (payload.length > MAX_FRAME) throw new IllegalArgumentException("payload too large: " + payload.length);
        ByteBuffer b = ByteBuffer.allocate(8 + payload.length);
        b.put((byte) 'M').put((byte) 'R').put((byte) FRAMING_VERSION).put((byte) type).putInt(payload.length).put(payload);
        return b.array();
    }

    public static byte[] encode(int type, String json) {
        return encode(type, json.getBytes(StandardCharsets.UTF_8));
    }

    public static Frame read(DataInputStream in) throws IOException {
        byte[] h = new byte[8];
        in.readFully(h);
        if (h[0] != 'M' || h[1] != 'R') throw new IOException("bad magic");
        if (h[2] != FRAMING_VERSION) throw new IOException("unsupported framing version " + h[2]);
        int type = h[3] & 0xff;
        int len = ByteBuffer.wrap(h, 4, 4).getInt();
        if (len < 0 || len > MAX_FRAME) throw new IOException("bad frame length " + len);
        byte[] p = new byte[len];
        in.readFully(p);
        return new Frame(type, p);
    }

    /** cells: palette indices, index (y*sz + z)*sx + x. Layout matches protocol.encode_region. */
    public static byte[] encodeRegion(String headerJson, int[] cells) throws IOException {
        byte[] h = headerJson.getBytes(StandardCharsets.UTF_8);
        ByteBuffer raw = ByteBuffer.allocate(cells.length * 2);
        for (int c : cells) {
            if (c < 0 || c > 0xffff) throw new IllegalArgumentException("palette index " + c);
            raw.putShort((short) c);
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(ByteBuffer.allocate(4).putInt(h.length).array());
        out.write(h);
        try (DeflaterOutputStream z = new DeflaterOutputStream(out, new Deflater(6))) {
            z.write(raw.array());
        }
        return out.toByteArray();
    }
}
