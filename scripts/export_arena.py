"""Write the sky arena as an export the way the mod's /robot export + the Python server would (MJCF + header JSON), without Minecraft,
plus the route sidecar the scripted demo follows. Also the file the navigation training can point at.

usage: PYTHONPATH=server python scripts/export_arena.py [--out results/m1/arena] [--mpb 0.5] [--beam 1]
Writes <out>/arena_region.xml, <out>/arena_region.json (header: origin, size, palette, meters_per_block), <out>/arena_route.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcrobot import arena
from mcrobot.mjcf import build_scene, mjcf_sha256, to_mjcf


def header_of(region) -> dict:
    return {
        "origin": list(region.origin),
        "size": list(region.size),
        "palette": [{"name": p.name, "shape": p.shape} for p in region.palette],
        "dimension": region.dimension,
        "mc_version": "synthetic (mcrobot.arena)",
        "meters_per_block": region.meters_per_block,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m1/arena")
    ap.add_argument("--mpb", type=float, default=arena.DEFAULT_METERS_PER_BLOCK)
    ap.add_argument("--beam", type=int, default=arena.BEAM_W)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = arena.build_arena(beam_w=a.beam)
    region = arena.to_region(plan, meters_per_block=a.mpb)
    scene = build_scene(region)
    xml = to_mjcf(scene)
    (out / "arena_region.xml").write_text(xml)
    (out / "arena_region.json").write_text(json.dumps(header_of(region), indent=1))
    route = {
        "meters_per_block": a.mpb,
        "origin": list(arena.ORIGIN),
        "floor_top_m": arena.floor_top_m(a.mpb),
        "route_m": [[x, y, lab] for x, y, lab in arena.route_mj(plan, a.mpb)],
        "sections": [{"name": s.name, "x0": s.x0, "x1": s.x1, "note": s.note} for s in plan.sections],
        "plates": [[x, z] for x, z in plan.plates],
        "cams": {str(k): list(v) for k, v in arena.cam_presets().items()},
    }
    (out / "arena_route.json").write_text(json.dumps(route, indent=1))
    print(f"wrote {out}/arena_region.xml  sha256={mjcf_sha256(xml)}  colliders={len(scene.colliders)}  markers={len(scene.markers)}  solid_cells={scene.n_solid_cells}")
    ox, oy, oz = region.origin
    sx, sy, sz = region.size
    print(f"export_args {ox} {oy} {oz} {ox + sx - 1} {oy + sy - 1} {oz + sz - 1}")  # for `/robot export`, used by smoke_server.sh


if __name__ == "__main__":
    main()
