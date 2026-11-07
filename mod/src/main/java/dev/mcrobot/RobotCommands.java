package dev.mcrobot;

import static net.minecraft.server.command.CommandManager.argument;
import static net.minecraft.server.command.CommandManager.literal;

import com.google.gson.JsonObject;
import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import java.io.IOException;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.command.argument.BlockPosArgumentType;
import net.minecraft.server.command.CommandManager;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.text.Text;
import net.minecraft.util.math.BlockPos;

/** /robot ping | export <from> <to> | spawn <g1|go2|rover> | train | play | stop. Operators only. */
final class RobotCommands {
    private RobotCommands() {}

    static void register(CommandDispatcher<ServerCommandSource> d) {
        d.register(literal("robot")
                .requires(CommandManager.requirePermissionLevel(CommandManager.GAMEMASTERS_CHECK))
                .then(literal("ping").executes(c -> ping(c)))
                .then(literal("export").then(argument("from", BlockPosArgumentType.blockPos())
                        .then(argument("to", BlockPosArgumentType.blockPos()).executes(c -> export(c)))))
                .then(literal("spawn").then(argument("type", StringArgumentType.word())
                        .suggests((c, b) -> {
                            for (String t : new String[] {"g1", "go2", "rover"}) b.suggest(t);
                            return b.buildFuture();
                        })
                        .executes(c -> command(c, "spawn", StringArgumentType.getString(c, "type")))))
                .then(literal("train").executes(c -> command(c, "train", null)))
                .then(literal("play").executes(c -> command(c, "play", null)))
                .then(literal("stop").executes(c -> command(c, "stop", null))));
    }

    private static int ping(CommandContext<ServerCommandSource> c) {
        long t0 = System.nanoTime();
        JsonObject o = new JsonObject();
        o.addProperty("t0", t0);
        McRobotMod.bridge().submit(() -> McRobotMod.bridge().sendJson(Frames.PING, o));
        c.getSource().sendFeedback(() -> Text.literal("pinging trainer..."), false);
        return 1;
    }

    private static int export(CommandContext<ServerCommandSource> c) {
        ServerCommandSource src = c.getSource();
        BlockPos a = BlockPosArgumentType.getBlockPos(c, "from");
        BlockPos b = BlockPosArgumentType.getBlockPos(c, "to");
        byte[] payload;
        try {
            String mc = FabricLoader.getInstance().getModContainer("minecraft")
                    .map(m -> m.getMetadata().getVersion().getFriendlyString()).orElse("unknown");
            payload = RegionExporter.export(src.getWorld(), a, b, mc); // reads blocks: server thread only
        } catch (IllegalArgumentException | IOException e) {
            src.sendError(Text.literal("export failed: " + e.getMessage()));
            return 0;
        }
        McRobotMod.bridge().submit(() -> McRobotMod.bridge().send(Frames.REGION, payload));
        src.sendFeedback(() -> Text.literal("exporting " + payload.length + " bytes of region to trainer"), false);
        return 1;
    }

    private static int command(CommandContext<ServerCommandSource> c, String cmd, String arg) {
        JsonObject o = new JsonObject();
        o.addProperty("cmd", cmd);
        if (arg != null) o.addProperty("arg", arg);
        McRobotMod.bridge().submit(() -> McRobotMod.bridge().sendJson(Frames.COMMAND, o));
        c.getSource().sendFeedback(() -> Text.literal("sent /robot " + cmd), false);
        return 1;
    }
}
