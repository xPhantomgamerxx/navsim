import hydra 
import os
import json
import time
import re

from tqdm import tqdm
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from pathlib import Path
from navsim.common.dataclasses import SensorConfig
from hydra._internal.instantiate._instantiate2 import instantiate
from gpt_test import *
from openai import OpenAI


navsim_log_path = "/home/ubuntu/project_ws/navsim/dataset/navsim_logs/test"
metric_cache_path = "/home/ubuntu/project_ws/navsim/exp/metric_cache"
cfg_path = "/home/ubuntu/project_ws/navsim/navsim/planning/script/config/pdm_scoring"
cfg_name ="default_run_pdm_score"

myapi_key = os.environ.get("OPENAI_API_KEY")
if myapi_key is None:
    print(f"Please set OPENAI_API_KEY in your environment variables.")
    exit()
client = OpenAI(api_key=myapi_key)

def _get_image_arrays(agent_input, timestep=-1):
    images = [
        agent_input.cameras[timestep].cam_f0.image,
        agent_input.cameras[timestep].cam_l0.image,
        agent_input.cameras[timestep].cam_l1.image,
        agent_input.cameras[timestep].cam_l2.image,
        agent_input.cameras[timestep].cam_r0.image,
        agent_input.cameras[timestep].cam_r1.image,
        agent_input.cameras[timestep].cam_r2.image,
        agent_input.cameras[timestep].cam_b0.image
    ]
    return images

def _read_images(imgs_list:list):
    base64_images = []
    i = 0
    for img in imgs_list:
        # Convert to PIL image
        pil_img = Image.fromarray(img, mode='RGB')

        # Encode to JPEG in memory
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        buffer.seek(0)

        # Convert to base64
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        base64_images.append(img_base64)
        # if I want to use later
        # img_tag = f"data:image/jpeg;base64,{img_base64}" # URL base64 encoding for finetuning dataset JSON
    return base64_images

def _get_token_imgs(scene_loader, token):
    images = _get_image_arrays(scene_loader.get_agent_input_from_token(token))
    return _read_images(images)

def _get_client_response(token = None, temperature: int = 0, scene_loader = None, provide_few_shot: bool = False):
    agent_input = scene_loader.get_agent_input_from_token(token)
    # scene = scene_loader.get_scene_from_token(token)
    images = _get_image_arrays(agent_input)
    encoded_images = _read_images(images)
    message =[]
    message.append({"role": "system", "content": fix_system_message_v5})

    if provide_few_shot:
        token_list = ["df11795878cb5419","df240e44ad0d5c3c","9de91fbb8b275885"]
        scenario_list = [scenario_output_1, scenario_output_2, scenario_output_3]
        for idx, scene in enumerate(token_list):
            imgs = _get_token_imgs(scene_loader, scene)
            imgs_content = [{"type": "input_image",
                       "image_url": f"data:image/jpeg;base64,{enc}", "detail": "low"} for enc in imgs ]
            message.append({"role": "user",
                    "content": [{"type": "input_text", "text":"Evaluate how challenging this driving scenario is"},*imgs_content,],},)
            message.append({"role":"assistant", "content": scenario_list[idx]})

    images_content = [{
        "type": "input_image",
        "image_url": f"data:image/jpeg;base64,{enc}",} for enc in encoded_images
    ]           
    message.append({
        "role": "user",
        "content": [{"type": "input_text", "text":"Evaluate how challenging this driving scenario is"},*images_content,],},
    )
  
    response = client.responses.create(
        model="gpt-4o",
        input=message,
        max_output_tokens=1024,
        # temperature=0, # can either use temperature[0,2] or top_p [0,1]
    )
    
    return response

@hydra.main(config_path=cfg_path, config_name=cfg_name, version_base=None)
def main(cfg):
    ### what tokens to do here and what do we need and how will 
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_all_sensors(),
    )

    tokens = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    eval_tokens = []
    with open("/home/ubuntu/project_ws/navsim/gpt_test/jsons/tokens.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            eval_tokens.append(data["token"])
    # loop through number of randomly sampled tokens and get difficulties of scenarios
    # EACH QUERY COSTS 0.03 USD (32 IMAGES + TEXT QUERY TOKENS AND SO)
    for i in tqdm(range(201)):  
        token = np.random.choice(tokens)
        if token in eval_tokens:
            continue 
        eval_tokens.append(token)
        full_response = _get_client_response(token=token, scene_loader=scene_loader, provide_few_shot=True)
        scenario_description = full_response.output[0].content[0].text
        match = re.search(r"2\. Overall, the prediction difficulty of this scene is\s+(\d+)", scenario_description)
        if match:
            difficulty = int(match.group(1))
        else:
            difficulty = None
        data = {
            "difficulty": difficulty,
            "token": token,
            "explanation": scenario_description,
        }
        print(f"Iteration: {i}, Difficulty: {difficulty}")
        with open("/home/ubuntu/project_ws/navsim/gpt_test/jsons/tokens.jsonl", "a") as e:
            e.write(json.dumps({"token":token})+ "\n")
        with open("/home/ubuntu/project_ws/navsim/gpt_test/jsons/difficulties.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")
       

if __name__ == "__main__":
    main()