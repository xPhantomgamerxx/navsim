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
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig, Scene
from transformers import AutoModelForCausalLM, pipeline
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from navsim.agents.utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo
from navsim.agents.deepseek.deepseek_utils import *
from navsim.agents.deepseek.deepseek_config import DeepSeekConfig



class DeepSeekAgent(AbstractAgent):
    """
    Agent built on DeepSeek Pipeline
    """
    requires_scene = True

    def __init__(
            self,
            config: DeepSeekConfig,
            requires_scene: bool = True,
    ):  
        """
        Initializes the DeepSeek agent object.
        Args:
            trajectory_sampling (TrajectorySampling): trajectory sampling specification
            vlm_version (str): Version of the Vision-Language Model
            llm_version (str): Version of the Language Model
        """
        super().__init__()
        self._config = config
        self._trajectory_sampling = config.trajectory_sampling,
        self.vlm = None,
        self.llm = None,
        self.requires_scene = requires_scene
        # self.vlm_version = config.vlm_version
        # self.llm_version = config.llm_version

    def name(self) -> str:
        """
        Returns the Name of the Agent
        """

        return "DeepSeekAgent"
    
    def initialize(self) -> None:
        """
        Initializes the DeepSeek Agent, loads the models and tokenizer
        """
        self.vlm_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self._config.vlm_model)
        self.tokenizer = self.vlm_chat_processor.tokenizer
        self.vlm: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(self._config.vlm_model, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
        # self.llm = pipeline("text-generation", model=self._config.llm_model, device_map="auto", max_new_tokens=4096)

    def get_sensor_config(self) -> SensorConfig:
        """
        Returns the Sensor Configuration for the Agent
        """
        return SensorConfig(cam_f0 = True, cam_l0=False, cam_l2=False, cam_r0=True, cam_r2=False, cam_b0=False, cam_l1= False, cam_r1=False, lidar_pc=False)
    
    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        Args: 
            agent_input (AgentInput): Dataclass with agent inputs.
            scene (Scene): Scene object
        Returns:
            Trajectory: Trajectory representing the predicted ego's position in future
        """
        img = agent_input.cameras[-1].cam_f0.image
        ego_history = scene.get_history_trajectory(num_trajectory_frames=10)
        trajectory = pose_to_vel_cur(ego_history.poses)
        command = agent_input.ego_statuses[scene.scene_metadata.num_history_frames-1].driving_command
        initial_pose = agent_input.ego_statuses[scene.scene_metadata.num_history_frames-1].ego_pose
        if np.array_equal(command, np.array([1,0,0,0])):
            curr_command = "left"
        elif np.array_equal(command, np.array([0,1,0,0])):
            curr_command = "straight"
        elif np.array_equal(command, np.array([0,0,1,0])):
            curr_command = "right"
        prediction, speed_curvature_pred, scene_description, object_description, intent_description = GenerateMotion(
            current_image =     img, 
            past_waypoints =    ego_history.poses, 
            past_velocities =   trajectory[:,0], 
            past_curvatures =   trajectory[:,1], 
            past_intent =       None, 
            chat_processor =    self.vlm_chat_processor, 
            vlm =             self.vlm, 
            llm =          self.llm, 
            tokenizer =         self.tokenizer, 
            command =           curr_command
            )
        pred_len = len(speed_curvature_pred)
        pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
        pred_speeds = np.array(speed_curvature_pred)[:, 0]
        pred_traj = np.zeros((pred_len, 3))
        
        pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(pred_curvatures,pred_speeds,initial_pose,0, pred_len)
        traj = Trajectory(poses=pred_traj, trajectory_sampling=self._trajectory_sampling)

        return traj
        
        