package dev.mcrobot;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.brigadier.exceptions.CommandSyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.entity.Entity;

/**
 * Ghost robots as server-side block_display entities (vanilla clients render them, no client mod).
 * The Python side sends, per ghost, the full display transformation of every part; this class only turns that into
 * entity commands, so the geometry lives in Python. Latest-wins per ghost: frames arriving faster than the server tick
 * are dropped (never queued), so a slow game cannot back up the trainer.
 *
 * Ghosts never outlive a streaming session: every session gets its own tag and UUIDs, the first spawn of a session
 * removes every existing ghost entity, a stale ghost that loads later (saved in an unloaded chunk) is discarded, and
 * stop / trainer loss / server shutdown remove them all.
 */
public final class GhostManager {
    public static final String TAG = "mcrobot_ghost";
    private static final Gson GSON = new Gson();
    private static final long UUID_HI = 0x6d63726f_626f7400L;
    private static final long LOG_EVERY_NANOS = 30_000_000_000L;

    private final Map<Integer, JsonObject> latest = new ConcurrentHashMap<>();
    private final Map<Integer, Boolean> spawnPending = new ConcurrentHashMap<>();
    private final Queue<Entity> stale = new ConcurrentLinkedQueue<>();
    private final Queue<String> runCommands = new ConcurrentLinkedQueue<>();
    private final AtomicLong posesReceived = new AtomicLong();
    private volatile long lastPoseNanos;
    private volatile boolean clearPending;
    private volatile String clearReason = "";
    private volatile String sessionTag; // null when no session is streaming; read by the entity-load callback
    private long sessionBits;
    private final Set<Integer> spawned = new HashSet<>(); // server thread only
    private final Map<Integer, Integer> partCount = new HashMap<>(); // server thread only
    private long lastLogNanos;
    private String forceloaded; // "x0 z0 x1 z1" of the area this class forceloaded (server thread only)

    /**
     * Reader thread. payload: JSON {op:"parts", ghost, ticks, spawn, anchor, cull?, parts:[{b|m,t,r,s}]} (b = block id for a block_display, m =
     * item model id for an item_display), {op:"clear", reason?}, or {op:"run", event:"start|progress|finish|fall|cue", pct?, cs?, n?} (arena timer, arena v2 cues).
     */
    public void offer(byte[] payload) {
        JsonObject o = GSON.fromJson(new String(payload, StandardCharsets.UTF_8), JsonObject.class);
        String op = o.has("op") ? o.get("op").getAsString() : "parts";
        if (op.equals("clear")) {
            requestClear(o.has("reason") ? o.get("reason").getAsString() : "stopped by trainer");
            return;
        }
        if (op.equals("run")) {
            String cmd = runCommand(o);
            if (cmd != null) runCommands.add(cmd);
            else McRobotMod.LOG.warn("ignored unknown run event: {}", payload.length < 200 ? new String(payload, StandardCharsets.UTF_8) : "(long)");
            return;
        }
        int id = o.get("ghost").getAsInt();
        if (o.has("spawn") && o.get("spawn").getAsBoolean()) spawnPending.put(id, true);
        posesReceived.incrementAndGet();
        lastPoseNanos = System.nanoTime();
        latest.put(id, o);
    }

    /** Any thread: remove every ghost on the next tick. reason is announced if a session was streaming. */
    public void requestClear(String reason) {
        latest.clear();
        spawnPending.clear();
        clearReason = reason;
        clearPending = true;
    }

    /** Entity-load callback: a ghost that is not part of the current session is a leftover from a saved world. */
    public void onEntityLoad(Entity e) {
        Set<String> tags = tagsOf(e);
        if (!tags.contains(TAG)) return;
        String s = sessionTag;
        if (s == null || !tags.contains(s)) stale.add(e);
    }

    private static Set<String> tagsOf(Entity e) {
        //? if >=26.3 {
        return e.entityTags();
        //?} else {
        /*return e.getTags();
        *///?}
    }

    /** Server thread, once per tick. */
    public void tick(MinecraftServer server) {
        Entity e;
        while ((e = stale.poll()) != null) e.discard();
        String rc;
        while ((rc = runCommands.poll()) != null) run(server, rc);
        if (clearPending) {
            clearPending = false;
            String reason = clearReason;
            boolean was = sessionTag != null;
            clearNow(server);
            if (was) announce(server, "ghost streaming stopped: " + reason + " (" + posesReceived.get() + " poses received)");
        }
        if (!latest.isEmpty()) {
            int newGhosts = 0, newParts = 0;
            String where = null;
            for (Integer id : latest.keySet().toArray(new Integer[0])) {
                JsonObject o = latest.remove(id);
                if (o == null) continue;
                boolean spawn = spawnPending.remove(id) != null;
                if (spawn && !spawned.contains(id)) {
                    if (sessionTag == null) beginSession(server);
                    String w = spawn(server, id, o);
                    newGhosts++;
                    newParts += o.getAsJsonArray("parts").size();
                    if (where == null) where = w;
                }
                if (spawned.contains(id)) update(server, id, o);
            }
            if (newGhosts > 0) {
                verify(server);
                announce(server, (newGhosts == 1 ? "ghost spawned: " : "ghosts spawned: " + newGhosts + " ghosts, ") + newParts + " parts at " + where);
            }
        }
        long now = System.nanoTime();
        if (sessionTag != null && now - lastLogNanos >= LOG_EVERY_NANOS) {
            lastLogNanos = now;
            McRobotMod.LOG.info("ghost stream: {} poses received, last pose {} ms ago, {} ghosts",
                    posesReceived.get(), (now - lastPoseNanos) / 1_000_000, spawned.size());
            int expected = partCount.values().stream().mapToInt(Integer::intValue).sum();
            int have = query(server, "execute if entity @e[tag=" + sessionTag + "]");
            if (have < expected) { // entities vanished: drop the session so the next spawn flag re-summons fresh ghosts
                announce(server, "ERROR: only " + have + " of " + expected + " ghost parts exist; re-summoning");
                kill(server);
                spawned.clear();
                partCount.clear();
                sessionTag = null;
            }
        }
    }

    /** Server thread: remove every ghost entity and the forceload, and end the session. */
    public void clearNow(MinecraftServer server) {
        kill(server);
        if (forceloaded != null) {
            run(server, "forceload remove " + forceloaded);
            forceloaded = null;
        }
        spawned.clear();
        partCount.clear();
        sessionTag = null;
        latest.clear();
        spawnPending.clear();
    }

    private static void kill(MinecraftServer server) {
        run(server, "kill @e[tag=" + TAG + "]");
    }

    private void beginSession(MinecraftServer server) {
        kill(server); // never reuse what an earlier session left in the world
        sessionBits = (System.nanoTime() ^ (System.currentTimeMillis() << 20)) & 0x0000_ffff_ffff_0000L;
        sessionTag = "mcrobot_s" + Long.toHexString(sessionBits >>> 16);
        posesReceived.set(0);
        lastLogNanos = System.nanoTime();
    }

    private String spawn(MinecraftServer server, int id, JsonObject o) {
        JsonArray parts = o.getAsJsonArray("parts");
        int ticks = o.has("ticks") ? o.get("ticks").getAsInt() : 2;
        JsonArray anchor = o.getAsJsonArray("anchor");
        if (forceloaded == null) { // summon needs loaded chunks; keep the area around the first ghost loaded
            int x = (int) Math.floor(anchor.get(0).getAsDouble()), z = (int) Math.floor(anchor.get(2).getAsDouble());
            forceloaded = (x - 8) + " " + (z - 8) + " " + (x + 8) + " " + (z + 8);
            run(server, "forceload add " + forceloaded);
        }
        double cull = o.has("cull") ? o.get("cull").getAsDouble() : 0.0;
        for (int i = 0; i < parts.size(); i++) {
            JsonObject p = parts.get(i).getAsJsonObject();
            run(server, summonCommand(anchor, uuid(id, i), sessionTag, p, ticks, cull));
        }
        spawned.add(id);
        partCount.put(id, parts.size());
        return String.format(Locale.ROOT, "%.1f %.1f %.1f", anchor.get(0).getAsDouble(), anchor.get(1).getAsDouble(), anchor.get(2).getAsDouble());
    }

    private void update(MinecraftServer server, int id, JsonObject o) {
        JsonArray parts = o.getAsJsonArray("parts");
        int ticks = o.has("ticks") ? o.get("ticks").getAsInt() : 2;
        for (int i = 0; i < parts.size(); i++) {
            run(server, String.format(Locale.ROOT, "data merge entity %s {transformation:%s,interpolation_duration:%d,start_interpolation:0}",
                    uuid(id, i), transformation(parts.get(i).getAsJsonObject()), ticks));
        }
    }

    /** Reads the entities back: every part must exist and none may have decoded to air (invisible). */
    private void verify(MinecraftServer server) {
        int expected = partCount.values().stream().mapToInt(Integer::intValue).sum();
        int have = query(server, "execute if entity @e[tag=" + sessionTag + "]");
        int air = 0;
        for (String q : airQueries(sessionTag)) air += query(server, q);
        if (air > 0 || have < expected) {
            announce(server, "ERROR: ghost check failed: " + have + " of " + expected + " parts exist, " + air + " have block_state air (invisible)");
        }
    }

    /** Selector tests for session entities whose block_state reads back as air: plain string (26.3) and {Name:..} (older). */
    static String[] airQueries(String sessionTag) {
        return new String[] {
            "execute if entity @e[tag=" + sessionTag + ",nbt={block_state:\"minecraft:air\"}]",
            "execute if entity @e[tag=" + sessionTag + ",nbt={block_state:{Name:\"minecraft:air\"}}]"};
    }

    private UUID uuid(int id, int part) {
        return partUuid(id, sessionBits, part);
    }

    static UUID partUuid(int id, long sessionBits, int part) {
        return new UUID(UUID_HI | id, sessionBits | part);
    }

    static String summonCommand(JsonArray anchor, UUID u, String sessionTag, JsonObject p, int ticks) {
        return summonCommand(anchor, u, sessionTag, p, ticks, 0.0);
    }

    /**
     * block_display for a part with "b", item_display for a part with "m" (an item model of the server resource pack). cull > 0 sets the
     * culling box (blocks) and a longer view range: the parts are translated away from the entity's own position as the robot walks, and
     * a display is culled by its entity position, so a large course needs a large box.
     */
    static String summonCommand(JsonArray anchor, UUID u, String sessionTag, JsonObject p, int ticks, double cull) {
        String box = cull > 0 ? String.format(Locale.ROOT, ",width:%.1ff,height:%.1ff,view_range:2.0f", cull, cull) : "";
        String kind, content;
        if (p.has("m")) {
            kind = "item_display";
            content = "item:" + itemStack(p.get("m").getAsString());
        } else {
            kind = "block_display";
            content = "block_state:" + blockState(p.get("b").getAsString());
        }
        return String.format(Locale.ROOT,
                "summon minecraft:%s %s %s %s {UUID:%s,Tags:[\"%s\",\"%s\"],%s,transformation:%s,interpolation_duration:%d,start_interpolation:0%s}",
                kind, num(anchor.get(0)), num(anchor.get(1)), num(anchor.get(2)), nbtUuid(u), TAG, sessionTag, content, transformation(p), ticks, box);
    }

    /** SNBT of the item stack shown by an item_display: a stick whose item_model component points at the pack's model. */
    static String itemStack(String model) {
        //? if >=1.21.11 {
        return "{id:\"minecraft:stick\",count:1,components:{\"minecraft:item_model\":\"" + model + "\"}}";
        //?} else {
        /*return "{id:\"minecraft:stick\",Count:1b}"; // 1.20.1 has no item_model component: the model style needs >= 1.21.11
        *///?}
    }

    /** The function command for an arena run event, or null for an unknown event. Numbers only, never free text. */
    static String runCommand(JsonObject o) {
        String ev = o.has("event") ? o.get("event").getAsString() : "";
        switch (ev) {
            case "start": return "function mcrobot_demo:run/start";
            case "fall": return "function mcrobot_demo:run/fall";
            case "progress": return "function mcrobot_demo:run/progress {pct:" + clampInt(o, "pct", 0, 100) + "}";
            case "finish": return "function mcrobot_demo:run/finish {cs:" + clampInt(o, "cs", 0, 3_600_00) + "}";
            case "cue": return "function mcrobot_demo:run/cue {n:" + clampInt(o, "n", 0, 99) + "}"; // arena v2: checkpoint glow (1-8), piston pushes (11-13)
            default: return null;
        }
    }

    private static int clampInt(JsonObject o, String key, int lo, int hi) {
        int v = o.has(key) ? (int) Math.round(o.get(key).getAsDouble()) : 0;
        return Math.max(lo, Math.min(hi, v));
    }

    /** NBT for a block_display's block_state: 26.3 wants a plain string, older versions {Name:..} ([verify] where it changed: 26.1, 26.2). */
    static String blockState(String block) {
        //? if >=26.3 {
        return "\"" + block + "\"";
        //?} else {
        /*return "{Name:\"" + block + "\"}";
        *///?}
    }

    static String transformation(JsonObject p) {
        return String.format(Locale.ROOT, "{translation:%s,left_rotation:%s,scale:%s,right_rotation:[0f,0f,0f,1f]}",
                floats(p.getAsJsonArray("t")), floats(p.getAsJsonArray("r")), floats(p.getAsJsonArray("s")));
    }

    private static String floats(JsonArray a) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < a.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append(String.format(Locale.ROOT, "%.4ff", a.get(i).getAsDouble()));
        }
        return sb.append(']').toString();
    }

    private static String num(JsonElement e) {
        return String.format(Locale.ROOT, "%.4f", e.getAsDouble());
    }

    static String nbtUuid(UUID u) {
        long m = u.getMostSignificantBits(), l = u.getLeastSignificantBits();
        return String.format(Locale.ROOT, "[I;%d,%d,%d,%d]", (int) (m >> 32), (int) m, (int) (l >> 32), (int) l);
    }

    private static void announce(MinecraftServer server, String msg) {
        McRobotMod.LOG.info("{}", msg);
        server.getPlayerList().broadcastSystemMessage(Component.literal("[robot] " + msg), false);
    }

    private static void run(MinecraftServer server, String cmd) {
        CommandSourceStack src = server.createCommandSourceStack().withSuppressedOutput();
        server.getCommands().performPrefixedCommand(src, cmd);
    }

    /** Result of a command such as `execute if entity ...` (the match count); 0 when the test fails. */
    private static int query(MinecraftServer server, String cmd) {
        try {
            return server.getCommands().getDispatcher().execute(cmd, server.createCommandSourceStack().withSuppressedOutput());
        } catch (CommandSyntaxException e) {
            return 0;
        }
    }
}
