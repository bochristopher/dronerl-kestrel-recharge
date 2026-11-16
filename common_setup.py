"""Shared helpers: locate the official starter kit and build the evaluation env."""
import os
import sys

STARTER_DIR = os.path.abspath(os.environ.get(
    "DRONERL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "droneRL")))
if STARTER_DIR not in sys.path:
    sys.path.insert(0, STARTER_DIR)

from torch_impl.env.env import DeliveryDrones  # noqa: E402
from torch_impl.env.wrappers import WindowedGridView  # noqa: E402

# Exactly the env params used by the official drone_evaluator.py (6 drones -> 11x11 grid)
EVAL_ENV_PARAMS = {
    'charge_reward': -0.1, 'crash_reward': -1, 'delivery_reward': 1,
    'charge': 20, 'discharge': 10, 'drone_density': 0.05, 'dropzones_factor': 2,
    'n_drones': 6, 'packets_factor': 3, 'pickup_reward': 0, 'rgb_render_rescale': 1.0,
    'skyscrapers_factor': 3, 'stations_factor': 2,
}
OFFICIAL_SEEDS = [845, 99, 65, 96, 85, 39, 51, 17, 52, 35]
BASELINES = [os.path.join(STARTER_DIR, "sample_models", f"dqn-agent-{i}.safetensors") for i in range(1, 6)]


def make_env(**overrides):
    params = dict(EVAL_ENV_PARAMS)
    params.update(overrides)
    return WindowedGridView(DeliveryDrones(params), radius=3)
