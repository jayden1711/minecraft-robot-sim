package dev.mcrobot;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerEntityEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class McRobotMod implements ModInitializer {
    public static final String VERSION = "0.0.6";
    public static final Logger LOG = LoggerFactory.getLogger("mcrobot");
    /** World scale: physics metres per block. config/mcrobot.properties `meters_per_block`, env MCROBOT_METERS_PER_BLOCK overrides. */
    public static final double DEFAULT_METERS_PER_BLOCK = 0.5;

    private static Bridge bridge;
    private static double metersPerBlock = DEFAULT_METERS_PER_BLOCK;

    public static double metersPerBlock() {
        return metersPerBlock;
    }

    /** Parses a scale setting; anything that is not a finite number in [0.01, 100] is rejected. */
    static double parseScale(String s) {
        double v = Double.parseDouble(s.trim());
        if (!(v >= 0.01 && v <= 100.0)) throw new IllegalArgumentException("meters_per_block must be in [0.01, 100], got " + s);
        return v;
    }
    private static final GhostManager GHOSTS = new GhostManager();
    private static MinecraftServer server;

    public static Bridge bridge() {
        return bridge;
    }

    public static GhostManager ghosts() {
        return GHOSTS;
    }

    @Override
    public void onInitialize() {
        bridge = new Bridge(loadConfig());
        bridge.setChat(msg -> {
            MinecraftServer s = server;
            if (s != null) s.execute(() -> s.getPlayerList().broadcastSystemMessage(Component.literal("[robot] " + msg), false));
            LOG.info(msg);
        });
        ServerLifecycleEvents.SERVER_STARTED.register(s -> {
            server = s;
            GHOSTS.clearNow(s); // ghosts saved with the world by an earlier run (unloaded ones are culled on load)
            bridge.start(3000);
        });
        ServerTickEvents.END_SERVER_TICK.register(GHOSTS::tick);
        ServerEntityEvents.ENTITY_LOAD.register((entity, level) -> GHOSTS.onEntityLoad(entity));
        ServerLifecycleEvents.SERVER_STOPPING.register(s -> {
            bridge.shutdown();
            GHOSTS.clearNow(s); // none may be written to the world save
            LOG.info("ghosts removed on server shutdown");
            server = null;
        });
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> RobotCommands.register(dispatcher));
        LOG.info("mcrobot {} loaded", VERSION);
    }

    /** config/mcrobot.properties: host, port, token. The MCROBOT_TOKEN environment variable overrides token. */
    private static Bridge.Config loadConfig() {
        Properties pr = new Properties();
        Path f = FabricLoader.getInstance().getConfigDir().resolve("mcrobot.properties");
        if (Files.exists(f)) {
            try (InputStream in = Files.newInputStream(f)) {
                pr.load(in);
            } catch (IOException e) {
                LOG.warn("cannot read {}: {}", f, e.toString());
            }
        }
        String scale = System.getenv().getOrDefault("MCROBOT_METERS_PER_BLOCK", pr.getProperty("meters_per_block", ""));
        if (!scale.isBlank()) {
            try {
                metersPerBlock = parseScale(scale);
            } catch (RuntimeException e) {
                LOG.warn("bad meters_per_block '{}': {}; using {}", scale, e.getMessage(), DEFAULT_METERS_PER_BLOCK);
            }
        }
        LOG.info("world scale: {} m of physics per block", metersPerBlock);
        String token = System.getenv().getOrDefault("MCROBOT_TOKEN", pr.getProperty("token", ""));
        return new Bridge.Config(pr.getProperty("host", "127.0.0.1"), Integer.parseInt(pr.getProperty("port", "25599")), token);
    }
}
