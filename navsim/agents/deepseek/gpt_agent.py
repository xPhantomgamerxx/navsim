import os
import re
import ast
import numpy as np

from datetime import datetime
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig, Scene
from navsim.agents.deepseek.gpt_utils import *
from navsim.agents.deepseek.gpt_config import GPTConfig
from openai import OpenAI


class GPTAgent(AbstractAgent):
    requires_scene = True

    def __init__(
            self,
            config: GPTConfig,
            requires_scene: bool = True,
    ):
        super().__init__()
        self._trajectory_sampling =  TrajectorySampling(time_horizon=4, interval_length=0.5)
        self.client: OpenAI = None,
        self.requires_scene = requires_scene
        self.command_map = {
            (1, 0, 0, 0): "left",
            (0, 1, 0, 0): "straight",
            (0, 0, 1, 0): "right"
        }

    def name(self) -> str:
        return("GPTAgent")
    
    def initialize(self):
        myapi_key = os.environ.get("OPENAI_API_KEY")
        if myapi_key is None:
            print(f"Please set OPENAI_API_KEY in your environment variables.")
            exit()
        self.client = OpenAI(api_key=myapi_key)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(cam_f0 = True, cam_l0=True, cam_l1= False, cam_l2=False, cam_r0=True, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)
    
    def call_gpt(
            self, 
            imgs, 
            past_vel_cur,
            command,
            token,
            gpt_model = "ft:gpt-4.1-2025-04-14:scania-eearp:av-finetune-7:BNKqGQNC", # "gpt-4.1"
            ):
        message = []
        encoded_images = read_images(imgs)
        image_content = [{
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{enc}"}} for enc in encoded_images
        ]
        message.append({
            "role": "developer",
            "content": f"{system_message_v2}"}
        )
        
        message.append({
            "role": "user",
            "content": [{
                "type": "text", "text": f"""Using the provided images, you need to complete these  following instructions and questions.
---
1.{scene_description_prompt}

---
2.{object_description_prompt}

---
3.{command}{intent_description_prompt}

---
4. The historical velocities and curvatures of the ego car of the last 2 seconds at an interval of 0.5s up until the present are: {past_vel_cur}. {prediction_prompt}"""},
                *image_content,],
            },
        )
        response = self.client.chat.completions.create(
            model = gpt_model,
            messages = message,
            max_completion_tokens = 2048,
            metadata={
                "token": token,
            },
            store = True,
        )
        
        # print(message[0]["content"][0]["text"])
        # response = self.client.responses.create(
        #     model="ft:gpt-4.1-2025-04-14:scania-eearp:av-finetune-7:BNKqGQNC",
        #     instructions=system_message_v2,
        #     input=message,
        #     max_output_tokens=2048,
        #     metadata={
        #         "token": token,
        #     }
        # )
        return response
    
    def generate_motion(
            self, 
            curr_imgs, 
            past_velocities, 
            past_curvatures, 
            command,
            token,
            ):
        # past_curvatures = past_curvatures * 100
        past_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(past_velocities, past_curvatures)]
        past_speed_curvature_str = ", ".join(past_speed_curvature_str)
        full_response = self.call_gpt(imgs = curr_imgs, past_vel_cur=past_speed_curvature_str, command = command, token=token)
        # output = full_response.output[0].content[0].text
        output = full_response.choices[0].message.content
        pattern = r"Values:\s*(\[\[.*?\]\]|\[\(.*?\)\])"
        match = re.search(pattern, output, re.DOTALL)
        if match: vel_cur_preds = ast.literal_eval(match.group(1))
        else:
            raise ValueError("No match found in the output string.")
        return vel_cur_preds
    
    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory: 
        imgs = [
            agent_input.cameras[-1].cam_l0.image,
            agent_input.cameras[-1].cam_f0.image,
            agent_input.cameras[-1].cam_r0.image,
            ]
        curr_frame = scene.scene_metadata.num_history_frames-1
        ego_history = scene.get_history_trajectory()
        trajectory = pose_to_vel_cur(ego_history.poses)
        command = agent_input.ego_statuses[curr_frame].driving_command
        command = self.command_map.get(tuple(command)) # possibly introduces problem
        # initial_pose = agent_input.ego_statuses[curr_frame].ego_pose

        vel_cur_pred = self.generate_motion(
            curr_imgs = imgs, 
            past_velocities=trajectory[:,0],
            past_curvatures= trajectory[:,1],
            command = command,
            token= scene.scene_metadata.initial_token,
            )
        
        pred_curvatures = np.array(vel_cur_pred)[:, 1] / 100
        pred_speeds = np.array(vel_cur_pred)[:, 0]
        pred = predict_future_waypoints(pred_speeds, pred_curvatures)
        traj = Trajectory(pred)
        return traj
