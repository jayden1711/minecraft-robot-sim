# minecraft-robot-sim

Train robots in Minecraft: build a world, mark the task with blocks, type /robot train, watch robots learn live in-game, export a
policy that runs in plain MuJoCo (or later on a real robot). Plan: PLAN.md. State: HANDOFF.md (3-line header). Critiques: CRITIQUES.md.

Rules: read ~/projects/NUC_RULES.md first (shared NUC; Quatern has priority). Nothing from Quatern, ever.
- GPU only through `~/projects/bin/gpu-run --preemptible` (training yields to other jobs and resumes from checkpoints).
- Sims through `~/projects/bin/sim-run`. **Keep total NUC load under 20** (`cut -d' ' -f1 /proc/loadavg`): if it's above, wait.
- results/ (and logs, runs, clips) are git-ignored; record clips and screenshots there only. No result numbers in tracked docs.
- Apache-2.0. Every dependency and asset with its license in THIRD_PARTY.md. Minecraft itself is proprietary (Mojang EULA): never
  commit or redistribute Minecraft jars or assets; the owner accepts the EULA himself.
- Commit locally; never push (the owner decides; there is no remote yet). Worktrees only under ~/projects/wt/ (never under .claude/).
- No rm (move to results/_trash/), no inline python heredocs, no `cd X && ...` chains, no web prompts (mark [verify]).
- Simulation is stated plainly everywhere. Physics is MuJoCo, never Minecraft's.
