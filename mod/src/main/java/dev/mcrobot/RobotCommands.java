package dev.mcrobot;

import static net.minecraft.commands.Commands.argument;
import static net.minecraft.commands.Commands.literal;

import com.google.gson.JsonObject;
import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import java.io.IOException;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.commands.arguments.coordinates.BlockPosArgument;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;

/** /robot ping | export <from> <to> | spawn <g1|go2|rover> | train | play | stop. Operators only. */
final class RobotCommands {
    private RobotCommands() {}

    static void register(CommandDispatcher<CommandSourceStack> d) {
        d.register(literal("robot")
                //? if >=1.21.11 {
                .requires(Commands.hasPermission(Commands.LEVEL_GAMEMASTERS))
                //?} else {
                /*.requires(src -> src.hasPermission(2))
                *///?}
                .then(literal("ping").executes(c -> ping(c)))
                .then(literal("export").then(argument("from", BlockPosArgument.blockPos())
                        .then(argument("to", BlockPosArgument.blockPos()).executes(c -> export(c)))))
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

    private static int ping(CommandContext<CommandSourceStack> c) {
        long t0 = System.nanoTime();
        JsonObject o = new JsonObject();
        o.addProperty("t0", t0);
        McRobotMod.bridge().submit(() -> McRobotMod.bridge().sendJson(Frames.PING, o));
        c.getSource().sendSuccess(() -> Component.literal("pinging trainer..."), false);
        return 1;
    }

    private static int export(CommandContext<CommandSourceStack> c) {
        CommandSourceStack src = c.getSource();
        BlockPos a = BlockPosArgument.getBlockPos(c, "from");
        BlockPos b = BlockPosArgument.getBlockPos(c, "to");
        byte[] payload;
        try {
            String mc = FabricLoader.getInstance().getModContainer("minecraft")
                    .map(m -> m.getMetadata().getVersion().getFriendlyString()).orElse("unknown");
            payload = RegionExporter.export(src.getLevel(), a, b, mc); // reads blocks: server thread only
        } catch (IllegalArgumentException | IOException e) {
            src.sendFailure(Component.literal("export failed: " + e.getMessage()));
            return 0;
        }
        McRobotMod.bridge().submit(() -> McRobotMod.bridge().send(Frames.REGION, payload));
        src.sendSuccess(() -> Component.literal("exporting " + payload.length + " bytes of region to trainer"), false);
        return 1;
    }

    private static int command(CommandContext<CommandSourceStack> c, String cmd, String arg) {
        JsonObject o = new JsonObject();
        o.addProperty("cmd", cmd);
        if (arg != null) o.addProperty("arg", arg);
        McRobotMod.bridge().submit(() -> McRobotMod.bridge().sendJson(Frames.COMMAND, o));
        c.getSource().sendSuccess(() -> Component.literal("sent /robot " + cmd), false);
        return 1;
    }
}
