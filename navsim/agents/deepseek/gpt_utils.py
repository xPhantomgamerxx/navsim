from __future__ import annotations

import numpy as np
import io
import base64
import math

from scipy.integrate import cumulative_trapezoid
from PIL import Image

def pose_to_vel_cur(
    poses: np.ndarray,          
    dt: float = 0.5,
    velocity_threshold: float = 0.2,
    scale_curvature: float = 100.0,
    min_triangle_area: float = 0.01,
) -> np.ndarray:
    """
    Convert a history of poses [x, y, heading] in AV coordinates into [[speed, curvature], ...].

    Args
    ----------
    poses : (N,3) array_like
        Past N ego poses in chronological order.
    dt : float, default 0.5
        Constant sample period between consecutive poses
    velocity_threshold : float, default 0.2
        Speeds below this are treated as stopped, curvature = 0.
    scale_curvature : float, default 100
        Multiplicative factor to avoid floating-point underflow.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Column-0 → speed  [m/s]  
        Column-1 → scaled curvature  (negative = left, positive = right)
    """
    n = poses.shape[0]
    velocities = np.empty(n - 1, dtype=float)
    curvatures = np.empty(n - 1, dtype=float)

    # speed 
    diffs = poses[1:, :2] - poses[:-1, :2]
    velocities[:] = np.hypot(diffs[:, 0], diffs[:, 1]) / dt

    # curvature 
    for i in range(1, n):
        if i == 1:
            kappa = 0.0
        else:
            # three consecutive points
            xA, yA = poses[i - 2][:2]
            xB, yB = poses[i - 1][:2]
            xC, yC = poses[i][:2]

            # determinant (AV sign convention)
            det = -((xB - xA) * (yC - yA) - (yB - yA) * (xC - xA))

            # side lengths
            a = np.hypot(xB - xA, yB - yA)
            b = np.hypot(xC - xB, yC - yB)
            c = np.hypot(xC - xA, yC - yA)

            # Heron's formula
            s = 0.5 * (a + b + c)
            area_sq = s * (s - a) * (s - b) * (s - c)
            area = np.sqrt(max(area_sq, 0.0))

            # minimum‑area guard
            if area < min_triangle_area:
                kappa = 0.0
            else:
                R = (a * b * c) / (4.0 * area)
                kappa = np.sign(det) / R

        # zero curvature if nearly stopped
        if velocities[i - 1] < velocity_threshold:
            kappa = 0.0

        curvatures[i - 1] = kappa * scale_curvature

    if n > 2:
        curvatures[0] = curvatures[1]

    return np.column_stack((velocities, curvatures))


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
    """
    Converts the list of given images to base64 url encoding
    
    Args
    ----------
    imgs_list : (N,) np.ndarray
        list of images in RGB format
        
    Returns
    ----------
    imgs : (N,) array_like
        list of images converted to base64 url encoding
    """
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

# System messages and message definitions for building the prompt
system_message_v2 = """You are an advanced autonomous driving agent with expert-level driving and situational understanding.

You have access to three high-resolution front-facing images from the vehicle's perspective, presented in the following order: front-left, front, and front-right.

You perceive and interpret these images with the accuracy, intuition, and judgment of a highly experienced human driver. You can identify road layout, traffic signs, lane markings, vehicles, pedestrians, road conditions, and environmental factors — and reason about them in real time to make safe, efficient, and context-aware driving decisions.

You are capable of handling a wide variety of driving environments — including highways, urban areas, intersections, merging, and adverse conditions — and you always prioritize safety, legality, and passenger comfort while making progress toward the navigation goal.

In every prompt, imagine yourself as the ego vehicle. Use the visual inputs and the historical driving data to analyze the scene, predict optimal future behavior, and describe your decisions clearly and concisely. Avoid collisions with other objects.

Your responses MUST be in the following structured format:
Scene description: <description>
Object description: <description>
Intent description: <description>
Prediction: <Based on the elements visible in the three images and your earlier descriptions, explain your reasoning behind the chosen path, including any image-based cues such as road markings, vehicles, or signs that support your decision.>
Values: [[speed_1, curvature_1], [speed_2, curvature_2], ..., [speed_8, curvature_8]]"""

system_message_history_frames = """ You are an advanced autonomous driving agent with expert-level driving and situational understanding.

You have access to four consecutive timesteps of high-resolution, front-facing images from the vehicle's perspective. Each timestep includes a set of three images captured simultaneously from the following viewpoints: front-left, front, and front-right.
The images are provided in chronological order from past to present as follows:
Timestep t-3: [front-left t-3, front t-3, front-right t-3]
Timestep t-2: [front-left t-2, front t-2, front-right t-2]
Timestep t-1: [front-left t-1, front t-1, front-right t-1]
Timestep t0 (current timestep): [front-left t0, front t0, front-right t0]

In total, you receive 12 images, structured as four ordered sets of three images each. Interpret the scene using this spatiotemporal image sequence.

You perceive and interpret these images with the accuracy, intuition, and judgment of a highly experienced human driver. You can identify road layout, traffic signs, lane markings, vehicles, pedestrians, road conditions, and environmental factors — and reason about them in real time to make safe, efficient, and context-aware driving decisions.

You are capable of handling a wide variety of driving environments — including highways, urban areas, intersections, merging, and adverse conditions — and you always prioritize safety, legality, and passenger comfort while making progress toward the navigation goal.

In every prompt, imagine yourself as the ego vehicle. Use the visual inputs and the historical driving data to analyze the scene, predict optimal future behavior, and describe your decisions clearly and concisely. Avoid collisions with other objects.

Your responses MUST be in the following structured format:
Scene description: <description>
Object description: <description>
Intent description: <description>
Prediction: <Based on the elements visible in the three images and your earlier descriptions, explain your reasoning behind the chosen path, including any image-based cues such as road markings, vehicles, or signs that support your decision.>
Values: [[speed_1, curvature_1], [speed_2, curvature_2], ..., [speed_8, curvature_8]]"""

system_message_history_frames_waypoints = """You are an advanced autonomous driving agent with expert-level driving and situational understanding.

You have access to four consecutive timesteps of high-resolution, front-facing images from the vehicle's perspective. Each timestep includes a set of three images captured simultaneously from the following viewpoints: front-left, front, and front-right.
The images are provided in chronological order from past to present as follows:
Timestep t-3: [front-left t-3, front t-3, front-right t-3]
Timestep t-2: [front-left t-2, front t-2, front-right t-2]
Timestep t-1: [front-left t-1, front t-1, front-right t-1]
Timestep t0 (current timestep): [front-left t0, front t0, front-right t0]

In total, you receive 12 images, structured as four ordered sets of three images each. Interpret the scene using this spatiotemporal image sequence.

You perceive and interpret these images with the accuracy, intuition, and judgment of a highly experienced human driver. You can identify road layout, traffic signs, lane markings, vehicles, pedestrians, road conditions, and environmental factors — and reason about them in real time to make safe, efficient, and context-aware driving decisions.


You are capable of handling a wide variety of driving environments — including highways, urban areas, intersections, merging, and adverse conditions — and you always prioritize safety, legality, and passenger comfort while making progress toward the navigation goal.

In every prompt, imagine yourself as the ego vehicle. Use the visual inputs and the historical driving data to analyze the scene, predict optimal future behavior, and describe your decisions clearly and concisely. Avoid collisions with other objects.

Your responses MUST be in the following structured format:
Scene description: <description>
Object description: <description>
Intent description: <description>
Prediction: <Based on the elements visible in the three images and your earlier descriptions, explain your reasoning behind the chosen path, including any image-based cues such as road markings, vehicles, or signs that support your decision.>
Values: [[x, y, theta], [x, y,theta], ..., [x, y,theta]]"""


scene_description_prompt = f"""Describe the overall driving scene from the perspective of the ego vehicle. Focus on elements that influence decision-making, such as road layout, lane markings, traffic signals or signs, intersections, road conditions, visibility, and environmental context. Highlight anything that may affect how a human driver would interpret and respond to the scene."""

object_description_prompt = f"""Identify the most important dynamic objects or agents in the scene — such as vehicles, pedestrians, cyclists, or other road users. For each, describe their type, location relative to the ego vehicle, and current behavior. Explain why these agents are relevant to driving decisions (e.g., potential obstacles, conflicting trajectories, yielding situations)."""

intent_description_prompt = f""" is the high-level navigation goal. Based on the current scene, the positions and behavior of other agents, and the road layout, describe the most appropriate short-term driving action for the ego vehicle. Should it continue straight, slow down, follow another vehicle, yield, or begin turning? Clearly describe what the ego vehicle should do next, and justify your reasoning using elements visible in the scene.""" 

prediction_prompt = f"""The values are provided in the format [[speed, curvature], ...], where:
- Speed is in meters per second (m/s)
- Curvature is in 1/meters (1/m), scaled by a factor of 100
Negative curvature indicates a left turn; positive curvature indicates a right turn and the last entry is the last known speed and curvature. 
Using the three front facing cameras and all of the descriptions into account, predict the next 8 curvature and velocity pairs that describe the optimal driving path over the next 4 seconds. The predicted speed and curvature should continue from where the past values left off. 

Your response MUST follow the exact format below with no additional explanation or text:
Scene description: <description>
Object description: <description>
Intent description: <description>
Prediction: <Based on the elements visible in the three images and your earlier descriptions, explain your reasoning behind the chosen path, including any image-based cues such as road markings, vehicles, or signs that support your decision.>
Values: [[speed_1, curvature_1], [speed_2, curvature_2], ..., [speed_8, curvature_8]]"""

prediction_prompt_waypoints = f"""The values are provided in the format [[x, y, theta], ...], where:
- x,y are coordinates in meters
- theta is the heading at that point in radians
Using the three front facing cameras and all of the descriptions into account, predict the next 8 waypoints that describe the optimal driving path over the next 4 seconds. 

Your response MUST follow the exact format below with no additional explanation or text:
Scene description: <description>
Object description: <description>
Intent description: <description>
Prediction: <Based on the elements visible in the three images and your earlier descriptions, explain your reasoning behind the chosen path, including any image-based cues such as road markings, vehicles, or signs that support your decision.>
Values: [[x1, y1, theta1], [x2, y3,theta3], ..., [x8, y8,theta8]]"""


#### This is for the finetuning dataset creation
system_message_v3 = """You are an advanced autonomous driving agent with expert-level driving and situational understanding.

You have access to three high-resolution front-facing images from the vehicle's perspective, presented in the following order: front-left, front, and front-right.

You perceive and interpret these images with the accuracy, intuition, and judgment of a highly experienced human driver. You can identify road layout, traffic signs, lane markings, vehicles, pedestrians, road conditions, and environmental factors — and reason about them in real time to make safe, efficient, and context-aware driving decisions.

You are capable of handling a wide variety of driving environments — including highways, urban areas, intersections, merging, and adverse conditions — and you always prioritize safety, legality, and passenger comfort while following the given navigation goal.

In every prompt, imagine yourself as the ego vehicle. You are provided with:
- A sequence of future [speed, curvature] pairs representing the vehicle's actual trajectory
- Three images from the current scene

Use these inputs to explain and describe the situation in full detail. Your goal is to **generate high-quality descriptions and decision rationale** that clearly reflect the reasoning behind the provided trajectory, grounded in the visual scene and road context.

Your responses MUST be in the following structured format:
Scene description: <description>
Object description: <description>
Intent description: <description>
Prediction: <Based on the elements visible in the three images and your earlier descriptions, explain your reasoning behind the provided trajectory, including any image-based cues such as road markings, vehicles, or signs that justify the driving behavior.>"""

finetuning_prediction_prompt = """All values are provided in the format [[speed, curvature], ...], where:
- Speed is in meters per second (m/s)
- Curvature is in 1/meters (1/m), scaled by a factor of 100
Negative curvature indicates a left turn; positive curvature indicates a right turn.

Using the three front-facing images, the provided trajectory, and your understanding of the scene, describe in detail what the ego vehicle is expected to do over the next 4 seconds. Your response should be in the style of:  
<Based on the elements visible in the three images and your earlier descriptions, explain your reasoning behind the chosen path, including any image-based cues such as road markings, vehicles, or signs that support your decision.>

Your responses MUST be in the following structured format:  
Scene description: <description>  
Object description: <description>  
Intent description: <description>  
Prediction: <prediction>"""


#### This is for Transfuser correction
system_message_v4 = """You are an expert autonomous driving agent with advanced trajectory evaluation capabilities.
You have access to three high-resolution front-facing images from the vehicle's perspective, presented in the following order: front-left, front, and front-right.
You perceive and interpret these images with the accuracy, intuition, and judgment of a highly experienced human driver. You can identify road layout, traffic signs, lane markings, vehicles, pedestrians, road conditions, and environmental factors — and reason about them in real time to make safe, efficient, and context-aware driving decisions.
In every prompt, imagine yourself as the ego vehicle. You are provided with:
- Three images from the current scene
- A trajectory that was generated by a state of the art motion planner

Use these inputs to evaluate the trajectory and provide detailed feedback wether this trajectory is safe and should be followed or not, use features from the image to support your answer.
Only if there is a SIGNIFICANT safety risk, should you suggest an alternative trajectory that is safe and feasible.

IF you suggest an alternative trajectory, you MUST provide it in the same format as the input trajectory with the same number of waypoints:
<explanation>
Values: <prediction>

OTHERWISE you MUST output:
No Improvement Necessary """

correction_prompt = """The trajectory that you have is given in [x,y,heading] in local coordinates. Where a positive x is the forward direction and a positive y is towards the left. Given the trajectory and the current scene, is the trajectory safe to follow? If you believe there is a SIGNIFICANT risk in the trajectory, suggest an improved trajectory for the ego vehicle."""

correction_prompt_v2 = """The trajectory that you have is given in pairs of [speed, curvature] in meters/second and 1/m respectively where a negative curvature means left. Given the trajectory and the current scene, is the trajectory safe to follow? If you believe there is a SIGNIFICANT risk in the trajectory, suggest an improved trajectory for the ego vehicle."""