import json
import random
import os
import tqdm
import json
import hydra 

from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, AgentInput, Trajectory, SceneFilter
from pathlib import Path
from navsim.common.dataclasses import SensorConfig
from hydra._internal.instantiate._instantiate2 import instantiate
from gpt_test import *
from navsim.agents.deepseek.gpt_utils import *



navsim_log_path = "/home/ubuntu/project_ws/navsim/dataset/navsim_logs/test"
metric_cache_path = "/home/ubuntu/project_ws/navsim/exp/metric_cache"
cfg_path = "/home/ubuntu/project_ws/navsim/navsim/planning/script/config/pdm_scoring"
cfg_name ="default_run_pdm_score"


SPLIT = "test"
FILTER = "all_scenes"
FILTER = "navtest"


def main():
    hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split/scene_filter", version_base=None)
    cfg = hydra.compose(config_name=FILTER)
    scene_filter: SceneFilter = instantiate(cfg)
    openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

    scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{SPLIT}",
        openscene_data_root / f"sensor_blobs/{SPLIT}",
        scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )
    tokens = np.random.choice(scene_loader.tokens, 50)
    print(len(tokens))


if __name__ == "__main__":
    main()