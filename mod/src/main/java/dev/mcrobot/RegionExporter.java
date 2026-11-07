package dev.mcrobot;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.shapes.VoxelShape;

/** Reads a block region on the server thread and serializes it for the Python exporter. */
public final class RegionExporter {
    public static final int MAX_CELLS = 4_000_000;
    private static final Gson GSON = new Gson();

    private RegionExporter() {}

    /** Must run on the server thread. Returns the REGION payload. */
    public static byte[] export(ServerLevel world, BlockPos a, BlockPos b, String mcVersion) throws IOException {
        int x0 = Math.min(a.getX(), b.getX()), y0 = Math.min(a.getY(), b.getY()), z0 = Math.min(a.getZ(), b.getZ());
        int x1 = Math.max(a.getX(), b.getX()), y1 = Math.max(a.getY(), b.getY()), z1 = Math.max(a.getZ(), b.getZ());
        int sx = x1 - x0 + 1, sy = y1 - y0 + 1, sz = z1 - z0 + 1;
        long total = (long) sx * sy * sz;
        if (total > MAX_CELLS) throw new IllegalArgumentException("region has " + total + " blocks, max " + MAX_CELLS);

        Map<BlockState, Integer> index = new HashMap<>();
        JsonArray palette = new JsonArray();
        int[] cells = new int[(int) total];
        BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
        for (int y = 0; y < sy; y++) {
            for (int z = 0; z < sz; z++) {
                for (int x = 0; x < sx; x++) {
                    p.set(x0 + x, y0 + y, z0 + z);
                    BlockState st = world.getBlockState(p);
                    Integer i = index.get(st);
                    if (i == null) {
                        i = palette.size();
                        index.put(st, i);
                        palette.add(entry(world, p, st));
                    }
                    cells[(y * sz + z) * sx + x] = i;
                }
            }
        }
        JsonObject header = new JsonObject();
        header.add("origin", GSON.toJsonTree(new int[] {x0, y0, z0}));
        header.add("size", GSON.toJsonTree(new int[] {sx, sy, sz}));
        header.add("palette", palette);
        //? if >=1.21.11 {
        header.addProperty("dimension", world.dimension().identifier().toString());
        //?} else {
        /*header.addProperty("dimension", world.dimension().location().toString());
        *///?}
        header.addProperty("mc_version", mcVersion);
        header.addProperty("meters_per_block", McRobotMod.metersPerBlock());
        return Frames.encodeRegion(GSON.toJson(header), cells);
    }

    private static JsonObject entry(ServerLevel world, BlockPos pos, BlockState st) {
        JsonObject o = new JsonObject();
        o.addProperty("name", BuiltInRegistries.BLOCK.getKey(st.getBlock()).toString());
        VoxelShape shape = st.getCollisionShape(world, pos);
        JsonElement s;
        if (shape.isEmpty()) {
            s = new JsonPrimitive("empty");
        } else if (Block.isShapeFullBlock(shape)) {
            s = new JsonPrimitive("full");
        } else {
            JsonArray boxes = new JsonArray();
            for (AABB bx : shape.toAabbs()) {
                boxes.add(GSON.toJsonTree(new double[] {bx.minX, bx.minY, bx.minZ, bx.maxX, bx.maxY, bx.maxZ}));
            }
            s = boxes;
        }
        o.add("shape", s);
        return o;
    }
}
