# Rules and submission checklist (checked Nov 15, 2026, PT)

Sources:
- https://www.aicrowd.com/challenges/dronerl
- https://www.aicrowd.com/challenges/dronerl/challenge_rules
- https://github.com/nyx-ai/droneRL/blob/master/torch_impl/agents/dqn.py
- https://github.com/nyx-ai/droneRL/blob/master/drone_evaluator.py

Round deadline: Nov 23, 2026, 4:00 PM PT (Nov 24 00:00 UTC). Limit: 1,000 submissions/day.

Upload one `.safetensors` PyTorch checkpoint via Create Submission after joining the challenge and accepting participation rules. JAX checkpoints are rejected. The loader accepts `network_type` equal to `dense` or `conv`. All metadata values are strings. Required keys are `network_type`, `obs_shape`, `action_shape`, `dense_layers`; conv additionally needs `conv_layers` (tuple/list of dictionaries with out_channels, kernel_size, stride and padding). Shapes for this evaluator are (7, 7, 6) and (5,). The tensors must exactly match the corresponding starter network state_dict. Conv input is permuted HWC to CHW internally. Dense layers use the starter naming and activation placement. No custom inference code or additional features are uploaded. The provided checkpoint meets these requirements and has been loaded and scored with the official local evaluator.

Evaluation: five fixed baseline opponents, 10 seeded 1,000-step episodes. Rank by mean total episode reward; std is secondary. Training reward per step is not the same scale.

Eligibility, external data/code, shaping, and AI disclosure: the public challenge overview does not state special eligibility restrictions or an explicit AI-disclosure requirement. The linked rules page still contains an unrelated 2024 workshop conduct/beverage clause, not substantive 2026 competition rules. It does not establish permission for arbitrary external data, external code, or reward shaping. Therefore those questions remain UNCONFIRMED pending review of the actual terms presented when joining. Do not describe them as allowed merely because not prohibited on that page. The starter itself exposes reward settings, but that is not a competition permission. The final checkpoint uses only official simulated experience, official network classes, our training code, and unchanged evaluation rewards; no shaping or external dataset. Other experimental runs with pickup bonuses are documented but are not in its lineage. We voluntarily disclose AI assistance in README.md.

Before submission, Bo must:
1. Review this entry, results and limitations and explicitly approve proceeding.
2. Check eligibility and actual 2026 participation/platform terms; decide whether to accept them (the public rules link is stale).
3. Sign into or create an AIcrowd account, join DroneRL, and accept the rules only with authorization.
4. Review the final upload and give the second separate approval. Observe any required waiting period before submission.
5. Use Create Submission to upload checkpoints/kestrel_dqn_conv.safetensors. Review grading/video; do not promise local performance equals the hosted score.

No account was created, rules accepted, registration made, or checkpoint uploaded to AIcrowd in preparing this entry.
