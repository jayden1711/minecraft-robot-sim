# THIRD_PARTY

Every dependency, asset and tool, with upstream, pinned revision and license. Entries marked **[verify]** could not be confirmed offline in this session
(web prompts are disabled); confirm against the upstream LICENSE file before relying on them or before M1 ships.
This project is Apache-2.0 (LICENSE). Nothing below is vendored into git except where noted.

## Minecraft (proprietary)

| Item | Version | License / terms | In git? |
|------|---------|-----------------|---------|
| Minecraft: Java Edition client and server (incl. server jar fetched by the Fabric launcher) | 1.21.11 | Proprietary: Mojang/Microsoft EULA (https://aka.ms/MinecraftEULA) and Minecraft Usage Guidelines. The owner accepts the EULA; this repo never sets `eula=true` | **No.** Never committed or redistributed. `server-mc/` is git-ignored. Downloaded on the NUC only |
| Minecraft assets (textures, sounds, models) | | same | No. Nothing from Minecraft is used for training or exported to MuJoCo; block names are the only data read |

"Minecraft" is a Mojang trademark; this project is not affiliated with or endorsed by Mojang or Microsoft.
Mojang's obfuscation mappings are not used; Yarn mappings are.

## Java / Fabric toolchain (build and runtime of the mod)

| Item | Version | License | Where |
|------|---------|---------|-------|
| Eclipse Temurin JDK (Adoptium) | 21.0.13+9 | GPL-2.0 with Classpath Exception | `~/opt/jdk-21.0.13+9`; tarball sha256 d7f5d5cf865a2902b70ff549b97586a7d3f234e016f041ab447ea740c7d66d1c; not in git |
| Gradle | 9.8.2 (wrapper) | Apache-2.0 | `mod/gradlew`, `mod/gradle/wrapper/gradle-wrapper.jar` (Gradle's wrapper jar is checked in, standard practice) |
| Fabric Loader | 0.20.1 | Apache-2.0 | Maven, runtime in `server-mc/` |
| Fabric API | 0.141.6+1.21.11 | Apache-2.0 | `server-mc/mods/` (downloaded, not in git) |
| Fabric Loom (Gradle plugin) | 1.14.10 | MIT | Gradle plugin portal / Fabric Maven |
| Fabric server launcher / installer | installer 1.1.2 | Apache-2.0 | `server-mc/` not in git |
| Yarn mappings | 1.21.11+build.6 | CC0-1.0 | via Loom; not in git |
| JUnit Jupiter | 5.11.4 | EPL-2.0 | test-only, Maven |
| Gson, SLF4J, Brigadier | as bundled by Minecraft 1.21.11 | Apache-2.0 / MIT / MIT | provided by the game at runtime, not redistributed |

(Licenses of Fabric components: Loader, API Apache-2.0; Loom MIT; Yarn CC0 per upstream READMEs [verify].)

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

## Assets

None committed. Colors used in renders are arbitrary constants chosen in `server/mcrobot/mjcf.py`, not Minecraft textures.
