# THIRD_PARTY

Every dependency, asset and tool, with upstream, pinned revision and license. Entries marked **[verify]** could not be confirmed offline in this session
(web prompts are disabled); confirm against the upstream LICENSE file before relying on them or before M1 ships.
This project is Apache-2.0 (LICENSE). Nothing below is vendored into git except where noted.

## Minecraft (proprietary)

| Item | Version | License / terms | In git? |
|------|---------|-----------------|---------|
| Minecraft: Java Edition client and server (incl. server jar fetched by the Fabric launcher) | 26.3 (primary), 1.21.11, 1.20.1 | Proprietary: Mojang/Microsoft EULA (https://aka.ms/MinecraftEULA) and Minecraft Usage Guidelines. The owner accepts the EULA; this repo never sets `eula=true` | **No.** Never committed or redistributed. `server-mc/` is git-ignored. Downloaded on the NUC only |
| Minecraft assets (textures, sounds, models) | | same | No. Nothing from Minecraft is used for training or exported to MuJoCo; block names are the only data read |

"Minecraft" is a Mojang trademark; this project is not affiliated with or endorsed by Mojang or Microsoft.
Mojang's official mappings are used for development on every supported version (26.x needs them: it ships unobfuscated, with no Yarn). Their header: "You may copy and use the mappings for development purposes, but you may not redistribute the mappings complete and unmodified" and use of Minecraft is governed by the Minecraft EULA. Loom downloads them at build time into the Gradle cache; they are never in git or shipped (the built mod jars contain remapped class names only).

## Java / Fabric toolchain (build and runtime of the mod)

| Item | Version | License | Where |
|------|---------|---------|-------|
| Eclipse Temurin JDK (Adoptium) | 25.0.5+8 (Gradle, 26.3), 21.0.13+9 (1.21.11), 17.0.21+9 (1.20.1) | GPL-2.0 with Classpath Exception | `~/opt/jdk-*`; tarball sha256 25: 68982529869257d0ee6b824ec5aa270ff8b118bb438670da0b97ebdab178d7fb, 21: d7f5d5cf865a2902b70ff549b97586a7d3f234e016f041ab447ea740c7d66d1c, 17: 8b9b611d917e3dfd8419d3566b5949226f69b052fed7e109c1368add1dfe8711 (25 and 17 match the Adoptium API values); not in git |
| Gradle | 9.8.2 (wrapper) | Apache-2.0 | `mod/gradlew`, `mod/gradle/wrapper/gradle-wrapper.jar` (Gradle's wrapper jar is checked in, standard practice) |
| Fabric Loader | 0.20.1 | Apache-2.0 | Maven, runtime in `server-mc/` |
| Fabric API | 0.165.2+26.3, 0.141.6+1.21.11, 0.92.12+1.20.1 | Apache-2.0 | `server-mc/mods/` (downloaded and SHA-256-checked by `scripts/setup_server.sh`, from `scripts/pins.tsv`; not in git) |
| Fabric Loom (Gradle plugin) | 1.19.4 | MIT [verify: the jar and POM carry no licence text] | Fabric Maven |
| Stonecutter (Gradle plugin, multi-version preprocessor) | 0.9.8 | MPL-2.0 [verify: the 0.9.8 jar and POM carry no licence text; the bundled LICENSE file is the Unicode licence of a shaded dependency; MPL-2.0 is from memory of the upstream repository] | maven.kikugie.dev; build-time only, not shipped |
| Fabric server launcher / installer | installer 1.1.2 | Apache-2.0 | `server-mc/` not in git |
| Mojang official mappings | per version, see above | Mojang terms above | via Loom; not in git |
| JUnit Jupiter | 5.11.4 | EPL-2.0 | test-only, Maven |
| Gson, SLF4J, Brigadier | as bundled by Minecraft | Apache-2.0 / MIT / MIT | provided by the game at runtime, not redistributed |

(Licenses of Fabric components: Loader, API Apache-2.0; Loom MIT per upstream READMEs [verify].)

## Server-side Via (NUC Minecraft server only; we run these, we do not ship them)

Downloaded by `scripts/setup_server.sh --via` from the Modrinth CDN into git-ignored server directories (`server-mc/mods/`, `server-mc-test/*/mods/`),
checked against the SHA-256 in `scripts/pins.tsv` (the SHA-512 values Modrinth publishes matched the downloads on 2026-11-06). They are never committed, bundled into any
artifact, or redistributed. They are GPL-3.0 programs that run next to the mod in the same JVM as separate mods; the Apache-2.0 mod does not link against them and does not use their API.
(Whether loading them in the same JVM raises any combined-work question is the owner's call [verify]; nothing here distributes the combination.)

| Item | Version | Role | License (Modrinth project metadata; confirm against the upstream LICENSE) | sha256 |
|------|---------|------|---------|--------|
| ViaFabric | 0.4.22+188-26.x (alpha channel) | Fabric platform for ViaVersion; works on a dedicated server | GPL-3.0-only | 8d3d691199988b931d725b792aa6e0d4e24efe1fb15869e6b52f24b91a0b88e5 |
| ViaVersion | 5.12.2 | lets newer clients join older servers (here: also the engine ViaBackwards builds on) | GPL-3.0-or-later | c0b63badcb0f7d38f4443e06820f2b8142dd4f0d3b0511f9330ff1050430e6b2 |
| ViaBackwards | 5.12.2 | lets older clients (1.9 and up) join the newer server | GPL-3.0-only | e989440e0aad7e12d4685d7a57df854691b67660966a95ec0450e20ce98e87ab |

ViaFabric 0.4.22 embeds ViaVersion 5.12.1-SNAPSHOT as a nested jar; the top-level ViaVersion 5.12.2 above overrides it (the documented way). Not used: ViaFabricPlus (client-side), ViaRewind, ViaProxy.
Mojang's server jar is fetched by the Fabric launcher at first start (not hash-pinned: the launcher is generated per request by meta.fabricmc.net [verify]).

## Python (server, exporter, tests)

| Item | Version | License |
|------|---------|---------|
| Python | 3.12 | PSF-2.0 |
| NumPy | 2.5.4 | BSD-3-Clause |
| MuJoCo (pip: mujoco, mujoco-core, mujoco-studio) | 4.0.0 | Apache-2.0 [verify mujoco-studio] |
| pytest | 9.2.0 | MIT |
| Pillow | 13.0.0 | MIT-CMU (HPND) |
| Mesa (llvmpipe/EGL, system library used for CPU rendering) | system | MIT |

## Planned (M1-M3), not yet used

| Item | Role | License | Status |
|------|------|---------|--------|
| MuJoCo Warp (`mujoco_warp`) | GPU-parallel MuJoCo | Apache-2.0 | [verify] version/licence at pin time |
| mjlab | RL envs/training on MuJoCo Warp | Apache-2.0 | [verify] |
| MuJoCo Menagerie: Unitree Go2, Unitree G1 | robot models | each model has its own LICENSE; Unitree Go2 and G1 are BSD-3-Clause | [verify] per-model LICENSE file at pin time; assets copied only if the licence allows redistribution, otherwise referenced and fetched |
| PyTorch (+ CUDA wheels) | training | BSD-3-Clause (NVIDIA CUDA libs under NVIDIA licence) | [verify] |
| Locomotion base policy for Go2 / G1 | M2 starting point | **unknown yet** | Candidates to check, in order: policies/configs shipped with mjlab's velocity-tracking tasks; Menagerie-compatible open policies; Unitree's open RL code. Licence and weight redistribution terms must be recorded here before use. Weights never go in git |
| ONNX / onnxruntime | policy export | MIT | [verify] |
| Diff-drive rover model | M1 | own MJCF, Apache-2.0 | written in this repo |

## Training venv (`.venv-train`, git-ignored; GPU navigation training, M1)

Versions and licences read from the installed packages' metadata on 2026-11-06 (offline); the upstream LICENSE texts were not re-read online [verify].

| Item | Version | License (package metadata) | Role |
|------|---------|---------|------|
| mjlab | 1.6.0 | Apache-2.0 | pins the MuJoCo / MuJoCo Warp versions below; **not used as a framework**: the navigation env (`server/mcrobot/navenv.py`) drives MuJoCo Warp directly, because the task is a custom exported arena with a frozen policy underneath. mjlab's own G1 model and policy come from the `unitree_rl_mjlab` checkout below |
| mujoco_warp | 3.11.0 | Apache-2.0 | GPU-parallel MuJoCo physics (the batched env) |
| mujoco (training venv) | 3.11.0 | Apache-2.0 | model building (`MjSpec`), CPU reference |
| warp-lang | 1.19.0 | Apache-2.0 (its wheel also bundles third-party licences under `licenses/`) | kernels for MuJoCo Warp |
| torch | 2.15.0 (CUDA 13) | Apache-2.0 AND BSD-2/3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib (as listed in the wheel) | policy, PPO |
| onnx | 1.23.2 | Apache-2.0 | reads the frozen walking policy's weights |
| onnxruntime | 1.31.1 | MIT | CPU reference walker (`g1.G1Walker`) |

Frozen low-level policy and G1 model (unchanged from the demo): `unitree_rl_mjlab`, policy v0 (`deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx`) and `g1.xml`, Apache-2.0 per the checkout's LICENSE, read from `MCROBOT_G1_ROOT`; its weights are loaded at run time, ported to a torch MLP in memory, and never copied into this repo. The trained navigation policies are our own results (git-ignored under `results/m1/nav/`).

## Resource pack (generated, not committed)

| Item | Role | License | Status |
|------|------|---------|--------|
| MuJoCo Menagerie `unitree_g1` visual meshes, rev `98a70c569c4c1b6da467a20012f766ee1f8dc521` | source of the G1 item models in the server resource pack (`scripts/build_pack.py`) | BSD-3-Clause (the upstream LICENSE text is copied into the pack as `LICENSE-menagerie-g1.txt`) | read from `~/projects/robot-arena-sim/third_party/mujoco_menagerie`; only simplified voxel approximations (cuboid elements) are derived; the logo mesh and all original textures are left out; the zip and its JSON are generated into git-ignored `results/m1/pack/`, never committed |

## Assets

None committed. Colors used in renders are arbitrary constants chosen in `server/mcrobot/mjcf.py`, not Minecraft textures.
