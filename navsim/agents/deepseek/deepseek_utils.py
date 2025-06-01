from __future__ import annotations

import os
import cv2
import re
import argparse
import torch
import logging
import json
import pytz
import math
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64


from math import atan2
from datetime import datetime
from transformers import AutoModelForCausalLM, pipeline
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from navsim.agents.utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo
from navsim.agents.deepseek.deepseek_config import DeepSeekConfig
from scipy.integrate import cumulative_trapezoid
from PIL import Image


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
        max_new_tokens=2048,
        do_sample=False,
        use_cache=False)

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
    verbose: bool = True
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
            "content": "<image_placeholder> \n \
            You are an advanced autonomous driving labeller, with access to a front-view camera image of a vehicle. \
            Carefully analyze the input image and describe every detail relevant to driving safely. \
            If available, include information about the road layout, lane markings, traffic signs, traffic signals, nearby vehicles, pedestrians, cyclists, environmental conditions (lighting, weather, road surface), potential obstacles, and any other noteworthy elements that could impact driving decisions. \
            Your description should be comprehensive and precise, focusing on the aspects necessary for an autonomous vehicle to understand and navigate the environment reliably. \
            Present your observations in a way that reflects how a self-driving car would perceive and label each element in the scene.",
            "images": [img],
            },
            {"role": "Assistant", "content": ""},
        ]
    elif task =="scene":
        prompt = [
            {"role": "User",
            "content": f"<image_placeholder>\n \
            You are an autonomous driving labeler with access to this front-view image from a car.\
            Imagine you are driving the car.\
            Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings.",
            "images": [img]},
            {"role": "Assistant", "content": ""}]
    elif task == "object":
        prompt = [
            {"role": "User",
            "content": f"<image_placeholder>\n \
            You are an autonomous driving labeler with access to this front-view image from a car.\
            Imagine you are the driver of the car. \
            What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you.",
            "images": [img]},
            {"role": "Assistant", "content": ""}] 
    elif task == "intent":
        if message == None:
            prompt = [{
                "role": "User",
                "content": f"<image_placeholder>\n \
                You are an autonomous driving labeler with access to this front-view image from a car.\
                Imagine you are driving the car. \
                A high level navigation goal has been given as {nav_goal}. \
                Based on the lane markings and the movement of other cars and pedestrians, describe the best course of action for the current car. \
                Is it going to follow the lane to turn left, turn right, or go straight? \
                Should it maintain the current speed or slow down or speed up?",
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
            "content": f"<image_placeholder>\n {message}. MAKE SURE TO FOLLOW THE SPECIFIED OUTPUT FORMAT.",
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
    verbose: bool = True,
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
    intent_description = call_vlm(message=None, img=current_image, chat_processor=chat_processor, vlm=vlm,tokenizer = tokenizer, nav_goal=command, task="intent")
    if verbose: print(f"Intent Description: \n{intent_description}")
    
    past_curvatures = past_curvatures * 100
    past_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(past_velocities, past_curvatures)]
    past_speed_curvature_str = ", ".join(past_speed_curvature_str)

    message = f"You are an expert driver, who is driving the ego vehicle.\
        The scene is described by:{scene_description}\
        The most important objects to pay attention to have been described as:{object_description}\
        The current intent of the vehicle is described as:{intent_description}\
        The historical velocities and curvatures of the ego car of the last 1.5 seconds at an interval of 0.5s up until the present are: {past_speed_curvature_str}\
        They are given in the format of [[speed_1, curvature_1],[speed_2, curvature_2],[speed_3,curvature_3]] with a positive curvature for left turn, negative curvature for right turn, where the last entry is the last known speed and curvature.\
        You must reason about the scene fully, then make a prediction about the next 8 velocities and curvatures the vehicle shall take. Provide these in the format of [speed_1, curvature_1], ..., [speed_8, curvature_8]. If there is ambiguity, assume the 2 seconds of historical velocities are correct. The predicted speed and curvature should continue from where the past values left off. "

    while True:
        speed_curvature_pred = []
        if method == "llm":
            if verbose: print(f"Message that will be passed to LLM: \n{message}")
            print("Calling LLM...")
            ticc = datetime.now()
            prediction = call_llm(message=message, llm_pipe=llm)
            tocc = datetime.now()
            print(f"Final call done in {tocc-ticc}")
            output = prediction[-1]['generated_text'][-1]['content']
            # print(output)
            keyword = '</think>'
            pre, sep, post =  output.partition(keyword)
            if sep: 
                coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", post)
                if len(coordinates) == 0:
                    coordinates = re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", post)
                speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
        elif method == "vlm":
            msg = f"{message} Base your prediction off the information as well as what you observe in the image"
            if verbose: print(f"Message that will be passed to VLM: \n{msg}")
            prediction = call_vlm(message=msg, img=current_image, chat_processor=chat_processor, vlm=vlm, tokenizer= tokenizer, task="final")

            pattern = r'[\[\(]\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*[\]\)]'
            matches = re.findall(pattern, prediction)

            # Convert matched strings to Python tuples safely
            parsed = [tuple(ast.literal_eval(match)) for match in matches]
            coordinates = parsed[-8:]
            # # Get the final 8 elements (assumed to be the predicted values)
            # coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", prediction)
            # if len(coordinates) == 0:
            #     coordinates = re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", prediction)
            speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
        if not speed_curvature_pred == []:
            break
    
    return prediction, speed_curvature_pred, scene_description, object_description, intent_description

def pose_to_vel_cur(poses, dt=0.5):
    velocities = []
    curvatures = []

    for i in range(1, len(poses)):
        # Compute velocity (Euclidean distance / dt)
        velocity_threshold = 0.2
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
                sign = np.sign(det)  # +1 for left turn, -1 for right turn
                curvature = sign * (1 / R)  # Curvature = 1 / Radius
            else:
                curvature = 0  # If collinear or degenerate triangle

            if velocity < velocity_threshold:
                curvature = 0.0
            curvatures.append(curvature)
        else:
            curvatures.append(0)  # First point has no curvature estimate
    curvatures[0] = curvatures[1]  # Set first curvature to second curvature
    # Return as a 2D list
    return np.array([[v, k] for v, k in zip(velocities, curvatures)])


def integrate_curvature_velocity_to_waypoints(curvatures,velocities,dt=0.5,initial_position=(0.0, 0.0),initial_heading=0.0):

    curvatures = np.array(curvatures).flatten()
    velocities = np.array(velocities).flatten()
    
    t = np.arange(len(curvatures)) * dt

    # Integrate heading using trapezoidal rule
    theta = cumulative_trapezoid(curvatures * velocities, t, initial=initial_heading)

    # Compute velocity components
    v_x = velocities * np.cos(theta)
    v_y = velocities * np.sin(theta)

    # Integrate position using trapezoid for first N-1 steps
    x = cumulative_trapezoid(v_x, t, initial=initial_position[0])
    y = cumulative_trapezoid(v_y, t, initial=initial_position[1])

    # Do an explicit Euler step to get the final waypoint
    x_final = x[-1] + v_x[-1] * dt
    y_final = y[-1] + v_y[-1] * dt

    # Stack waypoints: 7 from trapezoid + 1 from final step
    waypoints = np.vstack([
        np.stack((x[1:], y[1:]), axis=1),  # first 7 waypoints
        np.array([x_final, y_final])       # 8th waypoint
    ])
    waypoints = np.concatenate((waypoints, theta.reshape(8, 1)), axis=1)  # Add heading
    return waypoints


def predict_future_waypoints(pred_speeds, pred_curvatures, initial_pose = [0,0,0], dt=0.5):
    speeds = np.array(pred_speeds)
    curvatures = np.array(pred_curvatures)
    
    x, y, theta = initial_pose
    waypoints = [[x, y, theta]]

    for i in range(len(speeds) - 1):
        v1, v2 = speeds[i], speeds[i + 1]
        k1, k2 = curvatures[i], curvatures[i + 1]
        
        v_avg = 0.5 * (v1 + v2)
        k_avg = 0.5 * (k1 + k2)
        dtheta = v_avg * k_avg * dt

        if abs(k_avg) > 1e-6:
            R = 1.0 / k_avg
            dx = R * (np.sin(theta + dtheta) - np.sin(theta))
            dy = -R * (np.cos(theta + dtheta) - np.cos(theta))
        else:
            dx = v_avg * dt * np.cos(theta)
            dy = v_avg * dt * np.sin(theta)

        x += dx
        y += dy
        theta += dtheta
        waypoints.append([x, y, theta])

    # Final step using classical integration (Euler)
    v = speeds[-1]
    k = curvatures[-1]
    dtheta = v * k * dt

    if abs(k) > 1e-6:
        R = 1.0 / k
        dx = R * (np.sin(theta + dtheta) - np.sin(theta))
        dy = -R * (np.cos(theta + dtheta) - np.cos(theta))
    else:
        dx = v * dt * np.cos(theta)
        dy = v * dt * np.sin(theta)

    x += dx
    y += dy
    theta += dtheta
    waypoints.append([x, y, theta])
    final = np.array(waypoints)
    return final[-8:,:]

def predict_future_waypoints_rk4(speeds, curvatures, initial_pose=[0,0,0], dt=0.5):
    """
    4th-order Runge-Kutta integrator

    Args
    ----------
    speeds : (N,) array_like
        speeds in m/s
    curvatures : (N,) array_like
        curvatures in 1/m
    initial_pose : (3,) array_like
        initial pose [x,y,theta] is basically always [0,0,0]
    dt : float
        timestep between frames, is always 0.5s

    Returns
    ----------
    waypoints : (N,3) np.ndarray
        waypoints in format of [[x,y,heading], ...]
    """
    speeds = np.asarray(speeds, dtype=float)
    curvatures = np.asarray(curvatures, dtype=float)
    x, y, theta = initial_pose
    waypoints = [[x, y, theta]]

    def deriv(v_local, k_local, theta_local):
        return (v_local * math.cos(theta_local), v_local * math.sin(theta_local), v_local * k_local)
    
    def vk(alpha):
        return v1 + dv * alpha, k1 + dk * alpha
    
    n = len(speeds)
    for i in range(n):
        v1 = speeds[i]
        k1 = curvatures[i]
        if i < n - 1:
            v2 = speeds[i + 1]
            k2 = curvatures[i + 1]
        else:  
            v2, k2 = v1, k1

        dv = v2 - v1
        dk = k2 - k1

        vA, kA = vk(0.0)
        kx1, ky1, kth1 = deriv(vA, kA, theta)

        vB, kB = vk(0.5)
        kx2, ky2, kth2 = deriv(vB, kB, theta + 0.5 * dt * kth1)

        kx3, ky3, kth3 = deriv(vB, kB, theta + 0.5 * dt * kth2)

        vC, kC = vk(1.0)
        kx4, ky4, kth4 = deriv(vC, kC, theta + dt * kth3)

        dx = dt / 6.0 * (kx1 + 2 * kx2 + 2 * kx3 + kx4)
        dy = dt / 6.0 * (ky1 + 2 * ky2 + 2 * ky3 + ky4)
        dtheta = dt / 6.0 * (kth1 + 2 * kth2 + 2 * kth3 + kth4)

        x += dx
        y += dy
        theta += dtheta
        waypoints.append([x, y, theta])

    return np.array(waypoints)

def read_images(imgs_list:list):
    base64_images = []
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

system_message = f"""
You are an advanced autonomous driving labeller, with access to these three front view images.
They are presented in this order: front-left, front, front-right.
For all following prompts, you need to imagine you are driving the ego vehicle, then reason about the images like a human driver would, and respond with the level of detail needed for a self-driving car to understand the scene.

Respond in the following format:
Scene description: <description>
Object description: <description>
Intent description: <description>
Prediction: <explanation>
Values: [[speed_1, curvature_1], [speed_2, curvature_2], ..., [speed_8, curvature_8]]
"""

scene_description_prompt = f"""
Describe the provided scene according the most noteworthy elements in the scene that would influence the behaviour of a self-driving car. 
This could include the road layout, road markings, traffic signs, traffic signals, nearby vehicles and pedestrians, environmental conditions and anything else noteworthy."""

object_description_prompt = f"""
Describe the most important agents in the scene that you should be paying attention to as a self driving car.
List the most important ones, specifying their location within the driving scene and provide a short description of what that road user is doing, and why it is important to pay attention to.
"""

intent_description_prompt = f"""
 is the high-level navigation goal that has been given.
Based on the lane markings, the movement of the other agents in the scene and the high-level navigation goal, describe the current best low-level course of action for you to take as a driver.
Is it going to follow the lane to turn left, turn right, or go straight? 
Should it maintain the current speed or slow down or speed up?
"""

prediction_prompt = f"""
They are given in the format of [[speed_1, curvature_1],...,[speed_9,curvature_9]] with a positive curvature for left turn, negative curvature for right turn, where the last entry is the last known speed and curvature.
Taking the given images and all of the descriptions into account, predict the next 8 curvature and velocity pairs that describe the optimal driving path. 
YOU MUST PROVIDE these in the format of [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_8, curvature_8]. 
The predicted speed and curvature should continue from where the past values left off. 
"""