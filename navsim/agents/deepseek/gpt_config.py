from dataclasses import dataclass
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

@dataclass
class GPTConfig:
    """Global GPT config."""
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    