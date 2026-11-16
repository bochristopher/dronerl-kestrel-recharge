# DroneRL 2026: battery-aware DQN (Kestrel recharge study)

This is Bo Redfearn's entry for the [AIcrowd DroneRL 2026 round](https://www.aicrowd.com/challenges/dronerl). The round closes **Nov 24, 2026 00:00 UTC (Nov 23, 4:00 PM PT)**.

A drone on an 11×11 grid shares the airspace with five fixed baseline drones. It has to pick up packages, deliver them, avoid crashes, and recharge before its battery runs out. A full battery lasts 9 moves (−10 per step off a station, +20 per step on a station). Scoring is the mean total reward over 10 seeded 1,000-step episodes: +1 per delivery, −1 per crash or dead battery, and −0.1 per charging step.

The submission is a single PyTorch `.safetensors` checkpoint in the starter kit's **convolutional DQN** format (`network_type: conv`). It loads with the starter's own `BaseDQNFactory.from_checkpoint` and needs no custom code:

`checkpoints/kestrel_dqn_conv.safetensors` (851 KB; conv 6→32→32 with 3×3 kernels, then dense 128 → 5 Q-values)

## Results (local, real numbers)

All numbers come from the starter kit's evaluator logic: same env params, the same 5 baseline opponents, our drone in slot 0, and greedy argmax. `fast_eval.py` is a copy of that loop with video rendering removed. On the official seeds it reproduces `evaluate_agent.py` exactly. For example, it gives `sample_models/dqn-agent-1` −64.98, which is the score shown for that model on the public 2026 leaderboard.

| Agent | Official seeds (10 ep), mean ± std | Held-out seeds 20000–20099 (100 ep), mean ± std |
|---|---|---|
| Starter `train_torch.py` default (1k steps, dense 16-16) | −91.69 ± 7.05 | −86.15 ± 8.99 |
| Starter sample model `dqn-agent-1` | −64.98 ± 6.11 | −60.92 ± 9.53 |
| Stage 1: conv DQN, 2M steps (`runs/A_conv_g97`) | 56.82 ± 38.01 | 71.85 ± 15.14 |
| **Stage 2: greedy fine-tune, +1M steps (submitted checkpoint)** | **78.36 ± 9.64** | **84.34 ± 14.30** |

The official-seed score for the final checkpoint was checked with the starter's own `evaluate_agent.py` via `./verify_official.sh`, which reported `score 78.36, score_secondary 9.64`. On the official seeds the final agent averages about 147 deliveries, 48 crashes, and 207 charging steps per episode. Some of those crashes are collisions started by the baseline drones.

For context, the top public 2026 leaderboard entry was 70.12 when we checked on Nov 15, 2026. That is someone else's result, and a local score is not a leaderboard placement.

**How the checkpoint was chosen.** Checkpoints were picked during training on validation seeds 10000–10049. The final pick between runs used held-out seeds 20000–20099. The official seeds were never used for training or for choosing between runs, only for reporting.

### What was tried (held-out 100-seed means unless noted)
| Run | Change | Result |
|---|---|---|
| A | conv 32-32 + dense 128, γ=0.97, 3-step Double DQN | 71.85 |
| B | treat crashes as terminal | worse (≈36 on 20 val seeds) |
| C | +0.2 pickup bonus (training-only shaping) | 69.53 |
| D | γ=0.99 | worse early, stopped |
| E | dense 256-256 | 56.33 |
| F | also learn from the 5 opponents' transitions | 66.49 |
| H | dense 256-256 + opponent transitions, 6M steps | 69.89 |
| I | conv + pickup bonus + opponent transitions | 72.98 (best checkpoint, run still in progress when finalized) |
| **K** | **A, then fine-tuned 1M steps with ε=0** | **84.34** |
| L | A, then fine-tuned with ε=0.01 | 80.98 |
| M | C, then fine-tuned with ε=0 | 80.05 |

The biggest gain came from the **greedy (ε=0) fine-tune**. The stage-1 agent sometimes fell into a deterministic loop when no package or dropzone was in view, bouncing between a charging station and a nearby cell. Two of the 10 official episodes scored −12.4 and −22.1 this way. During ε-greedy training, random actions broke these loops, so the network never paid their full cost. Fine-tuning on purely greedy experience made the loops show up in the replay buffer, and the Q-values learned to avoid them.

## Approach
- **The same setting as the evaluator:** 11×11 grid, 6 drones, the 5 official baseline checkpoints as opponents, 1,000-step episodes, 16 environments stepped together with batched inference (`train_dqn.py`).
- **DQN improvements, all in training only, with the network classes unchanged:** Double DQN targets, 3-step returns, Huber loss, batch 256, Adam 5e-4, soft target updates (τ=0.005), gradient-norm clipping at 10, and a linear ε schedule over the first 30% of steps. Crashes are not terminal, because the drone respawns and the episode continues.
- **Dueling heads are not used.** The evaluator only rebuilds the starter's plain `Sequential` dense or conv nets, so a dueling checkpoint would not load.
- **Battery awareness comes from the observation the starter already provides.** Channel 4 holds the drone's own charge/100, and the 7×7 window shows stations, obstacles, and other drones. No extra input features were added, since the evaluator only feeds the raw 7×7×6 window. The final agent learned when to detour to a station from the delivery, crash, and charging rewards. No reward shaping is used in the submitted model's lineage (runs A and K).
- `heuristic.py` is a hand-written planner that uses only the window. It was a diagnostic and scored 20.42 on 20 validation seeds. It was **not** used to train the submitted model.

## Reproduce
```bash
git clone https://github.com/nyx-ai/droneRL.git            # official starter kit (sibling directory)
git clone https://github.com/bochristopher/dronerl-kestrel-recharge.git
cd droneRL && uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install --extra-index-url https://download.pytorch.org/whl/cpu -r torch_impl/requirements.txt
cd ../dronerl-kestrel-recharge
# stage 1 (about 1 h on one CPU core)
python train_dqn.py --name A_conv_g97
# stage 2: greedy fine-tune (about 25 min)
python train_dqn.py --name K_A_ft_greedy --init runs/A_conv_g97/best.safetensors \
  --eps_start 0 --eps_end 0 --lr 1e-4 --steps 1000000 --eval_every 100000 --eval_seeds 50 --seed 7
# evaluate
python fast_eval.py checkpoints/kestrel_dqn_conv.safetensors            # official seeds, no video
python robust_eval.py checkpoints/kestrel_dqn_conv.safetensors          # 100 held-out seeds
./verify_official.sh checkpoints/kestrel_dqn_conv.safetensors           # starter's evaluate_agent.py
python diagnose.py checkpoints/kestrel_dqn_conv.safetensors             # deliveries / crashes / charging
```
Set `DRONERL_DIR` if the starter kit is somewhere other than `../droneRL`. Training is seeded, but PyTorch CPU kernels and the timing of replay sampling mean a rerun gives a similar agent, not a bit-identical checkpoint. The submitted checkpoint's SHA-256 is `828132fe878b0b29820769d48e5ff8b5ed9bd9c3d164cf6ded36eec19cc9c526`.

## How this relates to Kestrel
Chargebotic's **Kestrel** is an early-stage (TRL 4) drone concept that would perch on power lines and recharge by induction instead of returning to base. This project is a toy problem in the same spirit: a drone with a small battery has to decide when an energy stop is worth its cost against the mission, while avoiding other aircraft. The learned trade-off is the kind of decision layer an autonomous Kestrel would eventually need.

To be clear about the gap:
- The grid world is discrete and has no physics. Its "stations" are just cells.
- Nothing here models line approach, perching, induction coupling, wind, or real battery chemistry.
- Kestrel flights today are **piloted, not autonomous**. A landing attempt on Nov 5, 2026 **struck the line**.
- This agent has not been flown, simulated in a physics engine, or validated on hardware. It is a learning exercise and a competition entry, not evidence of Kestrel's performance.

## AI-use disclosure
An AI assistant (Grok Bot) helped write the training, evaluation, and diagnostic code and this README, and it ran the experiments. Bo Redfearn decides whether and what to submit. The environment, the baseline opponents, the network classes, and the evaluator come from the official starter kit [nyx-ai/droneRL](https://github.com/nyx-ai/droneRL) (credits: @MasterScrat, @mar-muel, @spMohanty, @pacm, @metataro; EPFL Extension School and AIcrowd). They are used unmodified and are not vendored here. No external datasets were used.
