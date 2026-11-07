"""Render an exported MJCF to a PNG on the CPU (Mesa llvmpipe via EGL; never touches the GPU).

usage: render_mjcf.py model.xml out.png [camera_name]
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")

import mujoco
from PIL import Image


def main() -> None:
    xml, out = sys.argv[1], sys.argv[2]
    cam = sys.argv[3] if len(sys.argv) > 3 else "overview"
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 720, 1280)
    opt = mujoco.MjvOption()
    opt.sitegroup[4] = 1  # task-marker sites
    r.update_scene(d, camera=cam, scene_option=opt)
    Image.fromarray(r.render()).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
