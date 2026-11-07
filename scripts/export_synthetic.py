"""Export the synthetic demo maze to MJCF without Minecraft (also the build the owner recreates in-game for the M0 comparison).

usage: export_synthetic.py out_dir
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from mcrobot.mjcf import build_scene, mjcf_sha256, to_mjcf
from mcrobot.synthetic import demo_maze


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene(demo_maze())
    xml = to_mjcf(scene)
    (out / "demo_maze.xml").write_text(xml)
    print(f"cells={scene.n_solid_cells} boxes={len(scene.colliders)} markers={len(scene.markers)} sha256={mjcf_sha256(xml)[:12]}")


if __name__ == "__main__":
    main()
