package dev.mcrobot;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class McRobotMod implements ModInitializer {
    public static final String VERSION = "0.0.2";
    public static final Logger LOG = LoggerFactory.getLogger("mcrobot");

    private static Bridge bridge;
    private static MinecraftServer server;

    public static Bridge bridge() {
        return bridge;
    }

    @Override
    public void onInitialize() {
        bridge = new Bridge(loadConfig());
        bridge.setChat(msg -> {
            MinecraftServer s = server;
            if (s != null) s.execute(() -> s.getPlayerList().broadcastSystemMessage(Component.literal("[robot] " + msg), false));
            LOG.info(msg);
        });
        ServerLifecycleEvents.SERVER_STARTED.register(s -> server = s);
        ServerLifecycleEvents.SERVER_STOPPING.register(s -> {
            bridge.shutdown();
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
        String token = System.getenv().getOrDefault("MCROBOT_TOKEN", pr.getProperty("token", ""));
        return new Bridge.Config(pr.getProperty("host", "127.0.0.1"), Integer.parseInt(pr.getProperty("port", "25599")), token);
    }
}
