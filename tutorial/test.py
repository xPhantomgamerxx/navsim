import os
import cv2
import re
import argparse
import torch
import logging
import json
import pytz
import numpy as np
import matplotlib.pyplot as plt

from math import atan2
from datetime import datetime
# from truckscenes import TruckScenes
from transformers import AutoModelForCausalLM, pipeline
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig


SPLIT = "test"  # ["mini", "test", "trainval"]
FILTER = "all_scenes"
FILTER = "navtest"

def vlm_inference(
    message:list[dict] = None, 
    chat_processor: VLChatProcessor = None, 
    model: MultiModalityCausalLM = None,
    verbose: bool = False
) -> str:
    """ Runs inference on the provided model and returns the response from the VLM

    Args:
        message (list[dict]): The message that should be passed to the MLLM, in form of a dictionary with roles, content and images
        chat_processor (VLChatProcessor): The VLM chat processor to tokenize the input for the VLM
        model (MultiModalityCausalLM): VLM model to process the query and generate the response
        verbose (bool): Enables print statements

    Returns:
        answer (str): The answer of the VLM 
    """

    pil_images = load_pil_images(message)
    prepare_inputs = chat_processor(conversations=message, images=pil_images, force_batchify=True).to(model.device)
    
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=4096,
        do_sample=False,
        use_cache=True)

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).replace("\n\n", " ")
    if verbose:
        full_answer = (f"{prepare_inputs['sft_format'][0]}", answer)
        print("answer: \n", answer)
        print("full_answer \n", full_answer)
    return answer

def process_answer(answer: str):
    answer_dict = {}

if __name__ == "__main__":
    vlm_version: str = "deepseek-ai/Janus-Pro-7B"
    vlm_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(vlm_version)
    tokenizer = vlm_chat_processor.tokenizer
    vlm: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(vlm_version, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

    hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split/scene_filter")
    cfg = hydra.compose(config_name=FILTER)
    scene_filter: SceneFilter = instantiate(cfg)
    openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

    scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{SPLIT}",
        openscene_data_root / f"sensor_blobs/{SPLIT}",
        scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),)
    
    for i in range(1):
        token = np.random.choice(scene_loader.tokens)
        token = "19d3dfdf0d2d5b6f"
        scene = scene_loader.get_scene_from_token(token)
        frame_idx = scene.scene_metadata.num_history_frames - 1

        img = scene.frames[frame_idx-1].cameras.cam_f0.image
        cv2.imwrite(f"{token}{0}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))    
        img = scene.frames[frame_idx].cameras.cam_f0.image
        cv2.imwrite(f"{token}{1}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  
        img = scene.frames[frame_idx+1].cameras.cam_f0.image
        cv2.imwrite(f"{token}{2}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  
        prompt = [{
            "role": "User",
            "content": """<image_placeholder>, <image_placeholder>, <image_placeholder>\n You are a mature driver behind the wheel. You see these 3 consecutive images [image1, image2, image3] from the front of a vehicle. 
            Analyze and reason about these images fully. Among all road users you see, describe which objects might have abnormal moving behaviour that you might need to pay more attention to? Then score the overall difficulty of the driving scene by giving it a numerical score from 1 to 10, where 1 is the easiest and 10 is the hardest. If the traffic is very dense and it is bad weather you should give it a high score, if it is a straight road with few vehicles, it is easy to predict, you should give it a low score. If there is any anomalous behaviour or situation, you should give it a high score. Finally give a few keywords that describe the overall scene, e.g. 'construction zone', 'irregular vehicles', 'pedestrians'.""",
            "images": [scene.frames[frame_idx-1].cameras.cam_f0.image, scene.frames[frame_idx].cameras.cam_f0.image, scene.frames[frame_idx+1].cameras.cam_f0.image],
            },
            {"role": "Assistant", "content": ""},
        ]
        # prompt = [{
        #     "role": "User",
        #     "content": "<image_placeholder>\n You are a mature driver behind the wheel. You see this image captured from the front of a vehicle. Score the difficulty of the driving scene by giving it a numerical score from 1 to 10, where 1 is the easiest and 10 is the hardest. Then give a few keywords that describe the scene, e.g. 'construction zone', 'irregular vehicles', 'pedestrians'.",
        #     "images": [img],
        #     },
        #     {"role": "Assistant", "content": ""},
        # ]
        answer = vlm_inference(prompt, vlm_chat_processor, vlm)
        print(f"{token}: {answer}")
