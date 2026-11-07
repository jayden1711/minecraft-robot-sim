"""Write sky arena v2 as an export the way the mod's /robot export + the Python server would (MJCF + header JSON), without Minecraft,
plus the route sidecar (centre line, checkpoint triggers, push schedule, camera spots) the scripted demo and the navigation training use.

usage: PYTHONPATH=server python scripts/export_arena_v2.py [--out results/m1/arena_v2] [--mpb 0.5]
Writes <out>/arena_region.xml, <out>/arena_region.json (header: origin, size, palette, meters_per_block), <out>/arena_route.json.
The ice blocks carry friction (and priority) in the MJCF; see mcrobot.region.BLOCK_FRICTION.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcrobot import arena_v2 as A
from mcrobot.mjcf import build_scene, mjcf_sha256, to_mjcf


def header_of(region) -> dict:
    return {
        "origin": list(region.origin),
        "size": list(region.size),
        "palette": [{"name": p.name, "shape": p.shape} for p in region.palette],
        "dimension": region.dimension,
        "mc_version": "synthetic (mcrobot.arena_v2)",
        "meters_per_block": region.meters_per_block,
    }


def route_doc(plan, mpb: float) -> dict:
    return {
        "arena": "v2",
        "meters_per_block": mpb,
        "origin": list(A.ORIGIN),
        "floor_top_m": A.floor_top_m(mpb),
        "route_m": [[x, y, lab] for x, y, lab in A.route_mj(plan, mpb)],
        "blend": True,  # the route has a densely sampled curve (slalom): walk through waypoints without the one-step stop
        "sections": [{"key": s.key, "label": s.label, "sub": s.sub, "x0_m": s.x0 * mpb, "x1_m": s.x1 * mpb, "x0": s.x0, "x1": s.x1, "note": s.note} for s in plan.sections],
        "checkpoints": A.checkpoints_m(plan, mpb),
        "pushes": A.pushes_m(plan, mpb),
        "cams": {str(k): list(v) for k, v in A.cam_presets().items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m1/arena_v2")
    ap.add_argument("--mpb", type=float, default=A.DEFAULT_METERS_PER_BLOCK)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = A.build()
    region = A.to_region(plan, meters_per_block=a.mpb)
    scene = build_scene(region)
    xml = to_mjcf(scene)
    (out / "arena_region.xml").write_text(xml)
    (out / "arena_region.json").write_text(json.dumps(header_of(region), indent=1))
    (out / "arena_route.json").write_text(json.dumps(route_doc(plan, a.mpb), indent=1))
    print(f"wrote {out}/arena_region.xml  sha256={mjcf_sha256(xml)}  colliders={len(scene.colliders)}  markers={len(scene.markers)}  solid_cells={scene.n_solid_cells}")
    ox, oy, oz = region.origin
    sx, sy, sz = region.size
    print(f"export_args {ox} {oy} {oz} {ox + sx - 1} {oy + sy - 1} {oz + sz - 1}")  # for `/robot export`, used by smoke_server.sh
    # probes for smoke_server.sh: checkpoint 3's floor strip and the lamp of the gate it lights (world coordinates)
    st = plan.strips[2]
    gate = next(g for g in plan.gates if g["key"] == "door")
    lx, ly, lz = gate["lamps"][0]
    print(f"probe_strip3 {ox + st['x']} {oy + A.FL} {oz + A.ZC}")
    print(f"probe_lamp3 {ox + lx} {oy + ly} {oz + lz}")
    print(f"probe_labels {len(plan.labels) + sum(1 for lb in plan.labels if lb['sub'])} {len(plan.rams)}")


if __name__ == "__main__":
    main()
