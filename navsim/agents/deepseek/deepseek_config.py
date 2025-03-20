from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

@dataclass
class DeepSeekConfig:
    """Global DeepSeek config."""

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    
    vlm_model: str = "deepseek-ai/Janus-Pro-7B"
    llm_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    