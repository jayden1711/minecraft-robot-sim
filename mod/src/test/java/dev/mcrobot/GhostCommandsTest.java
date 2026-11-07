package dev.mcrobot;

import static org.junit.jupiter.api.Assertions.*;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.UUID;
import org.junit.jupiter.api.Test;

class GhostCommandsTest {
    private static JsonObject part() {
        return JsonParser.parseString("{\"b\":\"minecraft:gray_concrete\",\"t\":[0,1,2],\"r\":[0,0,0,1],\"s\":[1,1,1]}").getAsJsonObject();
    }

    private static JsonArray anchor() {
        return JsonParser.parseString("[1.5,64,-3]").getAsJsonArray();
    }

    @Test
    void blockStateIsNeverTheNameCompoundOn263() {
        String s = GhostManager.blockState("minecraft:gray_concrete");
        // The bug: 26.3 decodes {Name:".."} to air. Only the plain string is valid there; older versions want the compound.
        if (s.startsWith("{")) assertEquals("{Name:\"minecraft:gray_concrete\"}", s);
        else assertEquals("\"minecraft:gray_concrete\"", s);
    }

    @Test
    void summonCarriesSessionTagAndBlockState() {
        UUID u = GhostManager.partUuid(3, 0xabcd_0000L, 7);
        String cmd = GhostManager.summonCommand(anchor(), u, "mcrobot_sabcd", part(), 2);
        assertTrue(cmd.startsWith("summon minecraft:block_display 1.5000 64.0000 -3.0000 "));
        assertTrue(cmd.contains("Tags:[\"mcrobot_ghost\",\"mcrobot_sabcd\"]"));
        assertTrue(cmd.contains("block_state:" + GhostManager.blockState("minecraft:gray_concrete")));
    }

    @Test
    void uuidsDifferBetweenSessionsAndParts() {
        assertNotEquals(GhostManager.partUuid(0, 0x1_0000L, 0), GhostManager.partUuid(0, 0x2_0000L, 0));
        assertNotEquals(GhostManager.partUuid(0, 0x1_0000L, 0), GhostManager.partUuid(0, 0x1_0000L, 1));
        assertNotEquals(GhostManager.partUuid(0, 0x1_0000L, 0), GhostManager.partUuid(1, 0x1_0000L, 0));
    }

    @Test
    void modelPartBecomesItemDisplayWithItemModelComponent() {
        JsonObject p = JsonParser.parseString("{\"m\":\"mcrobot:g1/pelvis\",\"t\":[0,1,2],\"r\":[0,0,0,1],\"s\":[1,1,1]}").getAsJsonObject();
        String cmd = GhostManager.summonCommand(anchor(), UUID.randomUUID(), "mcrobot_sabcd", p, 2);
        assertTrue(cmd.startsWith("summon minecraft:item_display "));
        assertTrue(cmd.contains("item:" + GhostManager.itemStack("mcrobot:g1/pelvis")));
        assertFalse(cmd.contains("block_state"));
        assertTrue(cmd.contains("Tags:[\"mcrobot_ghost\",\"mcrobot_sabcd\"]"));
        assertTrue(GhostManager.itemStack("mcrobot:g1/pelvis").contains("minecraft:stick"));
    }

    @Test
    void cullBoxIsOnlyAddedWhenRequested() {
        String none = GhostManager.summonCommand(anchor(), UUID.randomUUID(), "mcrobot_sabcd", part(), 2);
        assertFalse(none.contains("width:"));
        String big = GhostManager.summonCommand(anchor(), UUID.randomUUID(), "mcrobot_sabcd", part(), 2, 96.0);
        assertTrue(big.contains(",width:96.0f,height:96.0f,view_range:2.0f}"));
    }

    @Test
    void runEventsMapToFixedFunctionsWithNumbersOnly() {
        assertEquals("function mcrobot_demo:run/start", GhostManager.runCommand(JsonParser.parseString("{\"event\":\"start\"}").getAsJsonObject()));
        assertEquals("function mcrobot_demo:run/fall", GhostManager.runCommand(JsonParser.parseString("{\"event\":\"fall\"}").getAsJsonObject()));
        assertEquals("function mcrobot_demo:run/progress {pct:42}", GhostManager.runCommand(JsonParser.parseString("{\"event\":\"progress\",\"pct\":42.4}").getAsJsonObject()));
        assertEquals("function mcrobot_demo:run/progress {pct:100}", GhostManager.runCommand(JsonParser.parseString("{\"event\":\"progress\",\"pct\":900}").getAsJsonObject()));
        assertEquals("function mcrobot_demo:run/finish {cs:7431}", GhostManager.runCommand(JsonParser.parseString("{\"event\":\"finish\",\"cs\":7431}").getAsJsonObject()));
        assertEquals("function mcrobot_demo:run/cue {n:12}", GhostManager.runCommand(JsonParser.parseString("{\"event\":\"cue\",\"n\":12}").getAsJsonObject()));
        assertEquals("function mcrobot_demo:run/cue {n:99}", GhostManager.runCommand(JsonParser.parseString("{\"event\":\"cue\",\"n\":4000}").getAsJsonObject()));
        assertEquals("function mcrobot_demo:run/cue {n:0}", GhostManager.runCommand(JsonParser.parseString("{\"event\":\"cue\",\"n\":-5}").getAsJsonObject()));
        assertNull(GhostManager.runCommand(JsonParser.parseString("{\"event\":\"say hi\"}").getAsJsonObject()));
        assertNull(GhostManager.runCommand(JsonParser.parseString("{\"op\":\"run\"}").getAsJsonObject()));
    }

    @Test
    void scaleSettingIsValidated() {
        assertEquals(0.5, McRobotMod.parseScale(" 0.5 "));
        assertEquals(1.0, McRobotMod.parseScale("1"));
        assertThrows(IllegalArgumentException.class, () -> McRobotMod.parseScale("0"));
        assertThrows(IllegalArgumentException.class, () -> McRobotMod.parseScale("-1"));
        assertThrows(IllegalArgumentException.class, () -> McRobotMod.parseScale("NaN"));
        assertThrows(NumberFormatException.class, () -> McRobotMod.parseScale("half"));
        assertEquals(0.5, McRobotMod.DEFAULT_METERS_PER_BLOCK);
    }

    @Test
    void airReadBackCoversBothNbtForms() {
        String[] q = GhostManager.airQueries("mcrobot_sabcd");
        assertEquals(2, q.length);
        assertTrue(q[0].contains("block_state:\"minecraft:air\""));
        assertTrue(q[1].contains("block_state:{Name:\"minecraft:air\"}"));
        for (String s : q) assertTrue(s.contains("tag=mcrobot_sabcd"));
    }
}
