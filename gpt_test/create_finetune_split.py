import json
import random
from pathlib import Path

import hydra 
import json

from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from pathlib import Path
from navsim.common.dataclasses import SensorConfig
from hydra._internal.instantiate._instantiate2 import instantiate
from gpt_test import *

# ==== CONFIG ====
navsim_log_path = "/home/ubuntu/project_ws/navsim/dataset/navsim_logs/test"
metric_cache_path = "/home/ubuntu/project_ws/navsim/exp/metric_cache"
cfg_path = "/home/ubuntu/project_ws/navsim/navsim/planning/script/config/pdm_scoring"
cfg_name ="default_run_pdm_score"
CHALLENGING_TOKENS_PATH = Path("/home/ubuntu/project_ws/navsim/gpt_test/jsons/difficult_tokens.jsonl")
OUTPUT_DIR = Path("/home/ubuntu/project_ws/navsim/gpt_test/splits")
OUTPUT_DIR.mkdir(exist_ok=True)
EVAL_SET_SIZE = 50
FINETUNE_SET_SIZE = 500
VAL_SPLIT_RATIO = 0.3  # 30% of finetune set goes to validation
# ================

def write_jsonl_token_file(path, token_list):
    with path.open("w") as f:
        for token in token_list:
            json.dump({"token": token}, f)
            f.write("\n")

@hydra.main(config_path=cfg_path, config_name=cfg_name, version_base=None)
def main(cfg):
    with CHALLENGING_TOKENS_PATH.open("r") as f:
        challenging_tokens = [json.loads(line)["token"] for line in f]

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_all_sensors(),
    )
    all_tokens = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))

    random.shuffle(challenging_tokens)
    eval_tokens = challenging_tokens[:EVAL_SET_SIZE]
    remaining_challenging_tokens = challenging_tokens[EVAL_SET_SIZE:]

    num_challenging_for_finetune = len(remaining_challenging_tokens)
    num_generic_needed = FINETUNE_SET_SIZE - num_challenging_for_finetune
        
    generic_tokens = [t for t in all_tokens if t not in challenging_tokens]

    if num_generic_needed > len(generic_tokens):
        raise ValueError("Not enough generic tokens to reach 500 examples.")

    random.shuffle(generic_tokens)
    finetune_generic_tokens = generic_tokens[:num_generic_needed]

    # Combine challenging + generic for finetune set
    finetune_tokens = remaining_challenging_tokens + finetune_generic_tokens
    random.shuffle(finetune_tokens)

    # Step 3: Split into train and validation
    val_size = int(len(finetune_tokens) * VAL_SPLIT_RATIO)
    finetune_val_tokens = finetune_tokens[:val_size]
    finetune_train_tokens = finetune_tokens[val_size:]

    write_jsonl_token_file(OUTPUT_DIR / "eval_challenging_tokens.jsonl", eval_tokens)
    write_jsonl_token_file(OUTPUT_DIR / "finetune_train_tokens.jsonl", finetune_train_tokens)
    write_jsonl_token_file(OUTPUT_DIR / "finetune_val_tokens.jsonl", finetune_val_tokens)

if __name__ =="__main__":
    main()