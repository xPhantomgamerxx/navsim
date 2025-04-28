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
from openai import OpenAI

myapi_key = os.environ.get("OPENAI_API_KEY")
if myapi_key is None:
    print(f"Please set OPENAI_API_KEY in your environment variables.")
    exit()
client = OpenAI(api_key=myapi_key)

navsim_log_path = "/home/ubuntu/project_ws/navsim/dataset/navsim_logs/test"
metric_cache_path = "/home/ubuntu/project_ws/navsim/exp/metric_cache"
cfg_path = "/home/ubuntu/project_ws/navsim/navsim/planning/script/config/pdm_scoring"
cfg_name ="default_run_pdm_score"
FULL_DATA_PATH = Path("full_dataset.jsonl")  # your entire dataset
CHALLENGING_TOKENS_PATH = Path("/home/ubuntu/project_ws/navsim/gpt_test/jsons/difficult_tokens.jsonl")
OUTPUT_DIR = Path("/home/ubuntu/project_ws/navsim/gpt_test/splits")
OUTPUT_DIR.mkdir(exist_ok=True)
finetune_train_tokens_path = "/home/ubuntu/project_ws/navsim/gpt_test/splits/finetune_train_tokens.jsonl"
finetune_train_data_path = "/home/ubuntu/project_ws/navsim/gpt_test/splits/finetune_train_data.jsonl"
finetune_val_tokens_path = "/home/ubuntu/project_ws/navsim/gpt_test/splits/finetune_val_tokens.jsonl"
finetune_val_data_path = "/home/ubuntu/project_ws/navsim/gpt_test/splits/finetune_val_data.jsonl"
command_map = {
            (1, 0, 0, 0): "left",
            (0, 1, 0, 0): "straight",
            (0, 0, 1, 0): "right"
        }

SPLIT = "test"  # ["mini", "test", "trainval"]
FILTER = "all_scenes"
FILTER = "navtest"

def write_jsonl(record: dict, fh):
    json.dump(record, fh, ensure_ascii=False)
    fh.write("\n")     # newline = record separator
    fh.flush()         # push Python buffer to OS
    os.fsync(fh.fileno())   # push OS buffer to disk

def do_thing(
        scene: Scene,
        agent_input: AgentInput,
        token,
        file,
):  
    imgs = [agent_input.cameras[-1].cam_l0.image,
            agent_input.cameras[-1].cam_f0.image,
            agent_input.cameras[-1].cam_r0.image,]
    curr_frame = scene.scene_metadata.num_history_frames-1
    # Get the GT trajectory and convert to string
    waypoints = scene.get_future_trajectory(8).poses
    gt_speed_curvatures = pose_to_vel_cur(waypoints)
    gt_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in gt_speed_curvatures]
    gt_speed_curvature_str = ", ".join(gt_speed_curvature_str)
    # Get the History trajectory and convert to string
    ego_history = scene.get_history_trajectory()
    history_trajectory = pose_to_vel_cur(ego_history.poses)
    past_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in history_trajectory]
    past_vel_cur = ", ".join(past_speed_curvature_str)
    # Parse command from agent input
    command = command_map.get(tuple(agent_input.ego_statuses[curr_frame].driving_command))
    message = []
    # Load and encode images for API format
    encoded_images = read_images(imgs)
    image_content = [{
        "type": "input_image",
        "image_url": f"data:image/jpeg;base64,{enc}",} for enc in encoded_images
    ]
    image_content_json = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{enc}"}} for enc in encoded_images
    ]

    # This is to get the scene, object, intent descriptions, but not predicting the actual trajectory here
    message.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text", 
                    "text": f"""Using the provided images and trajectory, you need to complete these  following instructions and questions.
---
1. {scene_description_prompt}

---
2. {object_description_prompt}

---
3. {command}{intent_description_prompt}

---
4. The historical velocities and curvatures of the ego car over the last 2 seconds (at 0.5-second intervals) are:{past_vel_cur}. 
The future velocities and curvatures over the next 4 seconds (at 0.5-second intervals) are: {gt_speed_curvature_str}.
{finetuning_prediction_prompt}"""},*image_content,],
        },
    )
    
    response = client.responses.create(
        model="gpt-4o",
        instructions = system_message_v3,
        input=message,
        max_output_tokens=2048,
        metadata={"token": token},
    )
    # output contains just the descriptions
    output = response.output[0].content[0].text
    formatted_output = f"{output}\nValues:{gt_speed_curvature_str}"

    # put together finetuning data, with proper system message, prompt, and then the gpt descriptions + GT data instead of speed_cur predictions
    json_data = {
        "messages":[
            {
                "role": "system",
                "content": system_message_v2
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"""Using the provided images, you need to complete these  following instructions and questions.
---
1. {scene_description_prompt}

---
2. {object_description_prompt}

---
3. {command}{intent_description_prompt}

---
4. The historical velocities and curvatures of the ego car of the last 2 seconds at an interval of 0.5s up until the present are: {past_vel_cur}. {prediction_prompt}"""},*image_content_json,],
            },
            {
                "role": "assistant",
                "content": f"{formatted_output}"
            }
        ]
    }

    write_jsonl(json_data, file)


# @hydra.main(config_path=cfg_path, config_name=cfg_name, version_base=None)
def main():
    hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split/scene_filter")
    cfg = hydra.compose(config_name=FILTER)
    scene_filter: SceneFilter = instantiate(cfg)
    openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

    scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{SPLIT}",
        openscene_data_root / f"sensor_blobs/{SPLIT}",
        scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )
    with open(finetune_train_tokens_path, "r") as f:
        finetune_train_tokens = [json.loads(line)["token"] for line in f]
    with open(finetune_val_tokens_path, "r") as f:
        finetune_val_tokens = [json.loads(line)["token"] for line in f]
    
    with open(finetune_train_data_path, "w") as f:
        for i, token in enumerate(finetune_train_tokens):
            print(f"Processing scenario {i + 1} / {len(finetune_train_tokens)}")
            scene = scene_loader.get_scene_from_token(token)
            agent_input = scene_loader.get_agent_input_from_token(token)
            do_thing(scene, agent_input,token, f)

    with open(finetune_val_data_path, "w") as f:
        for i, token in enumerate(finetune_val_tokens):
            print(f"Processing scenario {i + 1} / {len(finetune_val_tokens)}")
            scene = scene_loader.get_scene_from_token(token)
            agent_input = scene_loader.get_agent_input_from_token(token)
            do_thing(scene, agent_input,token, f)
        


if __name__ == "__main__":
    main()