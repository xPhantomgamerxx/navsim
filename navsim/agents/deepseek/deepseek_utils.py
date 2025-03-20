from __future__ import annotations

import os
import cv2
import re
import argparse
import torch
import logging
import json
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import atan2
from datetime import datetime
from transformers import AutoModelForCausalLM, pipeline
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from navsim.agents.utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo
from navsim.agents.deepseek.deepseek_config import DeepSeekConfig


def vlm_inference(
    message:list[dict] = None, 
    chat_processor: VLChatProcessor = None, 
    model: MultiModalityCausalLM = None,
    tokenizer: AutoModelForCausalLM = None,
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
        max_new_tokens=1024,
        do_sample=False,
        use_cache=True)

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).replace("\n\n", " ")
    if verbose:
        full_answer = (f"{prepare_inputs['sft_format'][0]}", answer)
        print("answer: \n", answer)
        print("full_answer \n", full_answer)
    return answer

def call_vlm(
    message: list[dict] = None,
    img: str = None,
    chat_processor: VLChatProcessor = None,
    vlm: MultiModalityCausalLM = None,
    tokenizer = None, 
    task: str = None,
    nav_goal: str = None,
    verbose: bool = False
) -> str:
    """ Calls the VLM with the task specific prompt
    
    Args:
        message (list[dict]): prompt to describe the scene 
        img_path (str): path to the img file that should be described
        chat_processor (VLChatProcessor): Texxt tokenizer
        model (MultiMidalityCausalLM): VLM model that will be prompted
        task (str): What task is being addressed
        verbose (bool): Enables printing
        
    Returns:
        answer (str): answer of the model
    """
    if task == None:
        prompt = [{
            "role": "User",
            "content": "<image_placeholder>\nYou are an advanced autonomous driving labeller, with access to a front-view camera image of a vehicle. Carefully analyze the input image and describe every detail relevant to driving safely. If available, include information about the road layout, lane markings, traffic signs, traffic signals, nearby vehicles, pedestrians, cyclists, environmental conditions (lighting, weather, road surface), potential obstacles, and any other noteworthy elements that could impact driving decisions. Your description should be comprehensive and precise, focusing on the aspects necessary for an autonomous vehicle to understand and navigate the environment reliably. Present your observations in a way that reflects how a self-driving car would perceive and label each element in the scene.",
            "images": [img],
            },
            {"role": "Assistant", "content": ""},
        ]
    elif task =="scene":
        prompt = [
            {"role": "User",
            "content": f"<image_placeholder>\n You are an autonomous driving labeller. You have access to this front-view camera image of a car. Imagine you are driving the car and describe the driving scene according to all aspects you think are important for driving safety. This could include traffic lights, movement of other cars or pedestrians, and lane markings. Give your reason as to why all of these objects are important to driving safely, but do not try to describe the movement of the ego vehicle.",
            "images": [img]},
            {"role": "Assistant", "content": ""}]
    elif task == "object":
        prompt = [
            {"role": "User",
            "content": f"<image_placeholder>\n You are an autonomous driving labeller. You have access to this front-view camera image taken from a driving car. Imagine you are the driver of the car. What other road users are you paying attention to in the driving scene? List as many as you think are important, specifying the location within the image of the driving scene and provide a short description of what that road user is currently doing, what they might do in the future, and why it is important to you. Dont try to describe the movement of the ego vehicle",
            "images": [img]},
            {"role": "Assistant", "content": ""}] 
    elif task == "intent":
        if message == None:
            prompt = [{
                "role": "User",
                "content": f"<image_placeholder>\n You are an autonomous driving labeller. You have access to this front-view camera image taken from a driving vehicle. Imagine you are driving the car. A high level navigation goal has been given as {nav_goal}. Based on the lane markings and the movement of other cars and pedestrians, describe the best course of action for the current car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?",
                "images": [img],
                },
                {"role": "Assistant", "content": ""},
            ]
        else:
            prompt = [{
                "role": "User",
                "content": f"<image_placeholder>\n You are an autonomous driving labeller. You have access to this front-view camera image taken from a driving vehicle. Imagine you are driving the car. The critical objects in the image have been described as: {message} A high level navigation goal has been given as {nav_goal}. Based on the image you see and the description of the critical objects, give a high level course of action that the ego vehicle should follow. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?",
                "images": [img],
                },
                {"role": "Assistant", "content": ""},
            ]
    elif task == "final":
        prompt = [
            {"role": "User", 
            "content": f"<image_placeholder>\n {message}",
            "images": [img]},
            {"role": "Assistant", "content": ""}]

    answer = vlm_inference(prompt, chat_processor, vlm, tokenizer)
    if verbose: 
        print("answer: \n", answer)
    return answer

def call_llm(
    message: (str) = None,
    llm_pipe: (pipeline) = None
) -> str:
    """Calls the LLM with the given prompt and returns the answer
    Args:
        message (str): The prompt to pass to the LLM (DeepSeek-R1-Distill-Qwen)
        llm_pipe (pipeline): The pipeline object that contains the LLM
    Returns:
        answer (str): The LLM's response to the prompt
    """
    prompt = [{"role": "user", 
               "content": f"{message}"}]
    answer = llm_pipe(prompt)
    return answer


def GenerateMotion(
    current_image: str = None, 
    past_waypoints = None, 
    past_velocities = None, 
    past_curvatures = None, 
    past_intent = None, 
    chat_processor: VLChatProcessor = None,
    vlm: MultiModalityCausalLM = None,
    llm: pipeline = None,
    tokenizer = None,
    command: str = None,
    verbose: bool = None,
    method: str = "vlm"
) -> str:
    """Applies the OpenEMMA method of generating the reasoning process behind the prediction.
    
    Args:
        current_image (str): current image
        
    Returns:
        str
    """
    scene_description = call_vlm(message=None, img=current_image, chat_processor=chat_processor, vlm=vlm, tokenizer = tokenizer, task="scene")
    if verbose: print(f"Scene Description: \n{scene_description}")
    object_description = call_vlm(message=None, img=current_image, chat_processor=chat_processor, vlm=vlm, tokenizer = tokenizer, task="object")
    if verbose: print(f"Object Description:\n{object_description}")
    intent_description = call_vlm(message=object_description, img=current_image, chat_processor=chat_processor, vlm=vlm,tokenizer = tokenizer, task="intent")
    if verbose: print(f"Intent Description: \n{intent_description}")
    
    past_curvatures = past_curvatures * 100
    past_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(past_velocities, past_curvatures)]
    past_speed_curvature_str = ", ".join(past_speed_curvature_str)

    message = f"You are a driving expert in this scenario. The scene you must analyze is described by: {scene_description} The most important objects have been described as: {object_description} The current intent of the vehicle is described as: {intent_description} The historical velocities and curvatures of the ego car of the last 5 seconds up until the present are: {past_speed_curvature_str} {f'For the previous frame, this prediction was given for the best motion: {past_intent} ' if past_intent else ''}. You must reason about the scene fully, then make a prediction about the next 10 velocities and curvatures the vehicle shall take. Provide these in the format of [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10] in the style of a python tuple. f there is ambiguity, assume the 5 seconds of historical velocities are correct. The predicted speed and curvature should continue from where the past values left off. "

    while True:
        speed_curvature_pred = []
        if method == "llm":
            if verbose: print(f"Message that will be passed to LLM: \n{message}")
            ticc = datetime.now()
            prediction = call_llm(message=message, llm_pipe=llm)
            tocc = datetime.now()
            print(f"Final call done in {tocc-ticc}")
            output = prediction[-1]['generated_text'][-1]['content']
            print(output)
            keyword = '</think>'
            pre, sep, post =  output.partition(keyword)
            if sep: 
                coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", post)
                if len(coordinates) == 0:
                    coordinates = re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", post)
                speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
        elif method == "vlm":
            msg = f"{message} Base your prediction off the information as well as what you observe in the image"
            # print(msg)
            if verbose: print(f"Message that will be passed to VLM: \n{msg}")
            prediction = call_vlm(message=msg, img=current_image, chat_processor=chat_processor, vlm=vlm, tokenizer= tokenizer, task="final")
            print(f'{prediction}')
            coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", prediction)
            if len(coordinates) == 0:
                coordinates = re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", prediction)
            speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
        if not speed_curvature_pred == []:
            break
    
    return prediction, speed_curvature_pred, scene_description, object_description, intent_description



def pose_to_vel_cur(poses, dt=0.5):
    """
    Compute velocity and curvature from a sequence of historic poses using the 3-point circle method.

    Parameters:
        poses (numpy array): An array of shape (N, 3) where each row is [x, y, heading].
        dt (float): The time step between each pose (default is 0.5s).

    Returns:
        list of lists: [[velocity_1, curvature_1], [velocity_2, curvature_2], ...]
    """
    velocities = []
    curvatures = []

    for i in range(1, len(poses)):
        # Compute velocity (Euclidean distance / dt)
        dx = poses[i, 0] - poses[i - 1, 0]
        dy = poses[i, 1] - poses[i - 1, 1]
        ds = np.sqrt(dx**2 + dy**2)  # Arc length
        velocity = ds / dt  # Velocity = distance / time
        velocities.append(velocity)

        # Compute curvature using the 3-point circle method (if enough points exist)
        if i > 1:
            x1, y1 = poses[i - 2][:2]
            x2, y2 = poses[i - 1][:2]
            x3, y3 = poses[i][:2]

            # Compute determinant (twice the signed area of the triangle)
            det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            norm = np.linalg.norm

            # Compute edge lengths
            a = norm([x2 - x1, y2 - y1])
            b = norm([x3 - x2, y3 - y2])
            c = norm([x3 - x1, y3 - y1])

            # Semi-perimeter
            s = (a + b + c) / 2

            # Area of the triangle using Heron's formula
            area = max(s * (s - a) * (s - b) * (s - c), 0)  # Avoid negative sqrt
            area = np.sqrt(area) if area > 0 else 0

            # Compute the circumradius
            if area > 0:
                R = (a * b * c) / (4 * area)
                curvature = 1 / R  # Curvature = 1 / Radius
            else:
                curvature = 0  # If collinear or degenerate triangle

            curvatures.append(curvature)
        else:
            curvatures.append(0)  # First point has no curvature estimate

    # Return as a 2D list
    return np.array([[v, k] for v, k in zip(velocities, curvatures)])

