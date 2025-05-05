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
        """
        Function that builds the prompt from the given information and then calls the OpenAI API to get the trajectory prediction.
        
        Args
        -
        imgs: list[Image]
            list of images from frame t0 [0], t-1 [1], t-2 [2], t-3 [3]
        past_vel_cur: str
            string of the past velocities and curvatures
        command: str
            high level driving goal/command
        token: str
            token for current scene
        gpt_model: str
            gpt model to use
        """
        message = []
        message.append({
            "role": "developer",
            "content": f"{system_message_history_frames}"}
        )
        image_content = []
        encoded_img_timeframes = []
        for timeframe in imgs:
            encoded_images = read_images(timeframe)
            encoded_img_timeframes.append(encoded_images)
        
        image_content.extend([{
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{enc}"}} for encoded_images in encoded_img_timeframes for enc in encoded_images
            ])
        
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
    
    def call_gpt_waypoints(
            self, 
            imgs, 
            past_waypoints,
            command,
            token,
            gpt_model = "ft:gpt-4.1-2025-04-14:scania-eearp:av-finetune-7:BNKqGQNC", # "gpt-4.1"
        ):
        """
        Function that builds the prompt from the given information and then calls the OpenAI API to get the trajectory prediction.
        
        Args
        -
        imgs: list[Image]
            list of images from frame t0 [0], t-1 [1], t-2 [2], t-3 [3]
        past_waypoints: str
            string of the past waypoints
        command: str
            high level driving goal/command
        token: str
            token for current scene
        gpt_model: str
            gpt model to use
        """
        message = []
        message.append({
            "role": "developer",
            "content": f"{system_message_history_frames_waypoints}"}
        )

        image_content = []
        encoded_img_timeframes = []
        for timeframe in imgs:
            encoded_images = read_images(timeframe)
            encoded_img_timeframes.append(encoded_images)
        content = []
        timesteps = ["t-3", "t-2", "t-1", "t-0"]

        for i, timestep in enumerate(timesteps):
            content.append({
                "type": "text",
                "text": f"These are the images at timestep {timestep} in order front-left, front, front-right."
            })
            for img in encoded_img_timeframes[i]:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })

        content.append({
            "type": "text",
            "text": f"""Using the provided images, you need to complete these  following instructions and questions.
        ---
        1.{scene_description_prompt}

        ---
        2.{object_description_prompt}

        ---
        3.{command}{intent_description_prompt}

        ---
        4. The historical waypoints of the ego car of the last 2 seconds at an interval of 0.5s up until the present are: {past_waypoints}. {prediction_prompt_waypoints}"""
        })

        # Add user message
        message.append({
            "role": "user",
            "content": content
        })


#         image_content.append([{
#                 "type": "image_url",
#                 "image_url": {"url": f"data:image/jpeg;base64,{enc}"}} for encoded_images in encoded_img_timeframes for enc in encoded_images
#             ])
        
#         message.append({
#             "role": "user",
#             "content": [{
#                 "type": "text", "text": f"""Using the provided images, you need to complete these  following instructions and questions.
# ---
# 1.{scene_description_prompt}

# ---
# 2.{object_description_prompt}

# ---
# 3.{command}{intent_description_prompt}

# ---
# 4. The historical waypoints of the ego car of the last 2 seconds at an interval of 0.5s up until the present are: {past_waypoints}. {prediction_prompt_waypoints}"""},
#                 *image_content,],
#             },
#         )
        response = self.client.chat.completions.create(
            model = gpt_model,
            messages = message,
            max_completion_tokens = 2048,
            metadata={
                "token": token,
            },
            store = True,
        )
        return response
    
    def generate_motion(
            self, 
            curr_imgs, 
            past_velocities, 
            past_curvatures, 
            command,
            token,
            ):
        """
        Function to generate the motion prediction of the vehicle using the GPT API
        
        Args
        -
        curr_imgs: list[Image]
            list of images from current frame [0], t-1 [1], t-2 [2], t-3 [3]
        past_velocities: list[float]
            list of the past velocities
        past_curvatures: list[float]
            list of the past curvatures
        command: str
            high level driving goal/command
        token: str
            token for current scene
        """
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
        return vel_cur_preds, output
    
    def generate_motion_waypoints(
            self, 
            curr_imgs, 
            past_waypoints ,
            command,
            token,
            ):
        """
        Function to generate the motion prediction of the vehicle using the GPT API
        
        Args
        -
        curr_imgs: list[Image]
            list of images from current frame [0], t-1 [1], t-2 [2], t-3 [3]
        past_velocities: list[float]
            list of the past velocities
        past_curvatures: list[float]
            list of the past curvatures
        command: str
            high level driving goal/command
        token: str
            token for current scene
        """
        # past_curvatures = past_curvatures * 100
        past_waypoints_str = [f"[{x[0]:.1f},{x[1]:.1f},{x[2]:.1f}]" for x in past_waypoints]
        past_waypoints_str = ", ".join(past_waypoints_str)
        full_response = self.call_gpt_waypoints(imgs = curr_imgs, past_waypoints=past_waypoints_str, command = command, token=token)
        # output = full_response.output[0].content[0].text
        output = full_response.choices[0].message.content
        pattern = r"Values:\s*(\[\[.*?\]\]|\[\(.*?\)\])"
        match = re.search(pattern, output, re.DOTALL)
        if match: vel_cur_preds = ast.literal_eval(match.group(1))
        else:
            raise ValueError("No match found in the output string.")
        return vel_cur_preds, output
    
    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory: 
        """
        Agent function to compute trajectory of ego vehicle in given scene.
        
        Args
        -
        self: class
            agent object
        agent_input: AgentInput
            scene data that is available for the agent to use for the trajectory prediction
        scene: Scene
            scene object that contains the scene history metadata
            
        Returns
        -
        trajectory: Trajectory
            the predicted trajectory in the Trajectory object
        response: str
            The response of the Model
            """
        imgs_t0 = [
            agent_input.cameras[-1].cam_l0.image,
            agent_input.cameras[-1].cam_f0.image,
            agent_input.cameras[-1].cam_r0.image,
            ]
        imgs_t1 = [
            agent_input.cameras[-2].cam_l0.image,
            agent_input.cameras[-2].cam_f0.image,
            agent_input.cameras[-2].cam_r0.image,
            ]
        imgs_t2 = [
            agent_input.cameras[-3].cam_l0.image,
            agent_input.cameras[-3].cam_f0.image,
            agent_input.cameras[-3].cam_r0.image,
            ]
        imgs_t3 = [
            agent_input.cameras[-4].cam_l0.image,
            agent_input.cameras[-4].cam_f0.image,
            agent_input.cameras[-4].cam_r0.image,
            ]
        imgs = [imgs_t3, imgs_t2, imgs_t1, imgs_t0]
        curr_frame = scene.scene_metadata.num_history_frames-1
        ego_history = scene.get_history_trajectory()
        ego_poses = ego_history.poses
        trajectory = pose_to_vel_cur(ego_history.poses)
        command = agent_input.ego_statuses[curr_frame].driving_command
        command = self.command_map.get(tuple(command)) # possibly introduces problem

        vel_cur_pred, response = self.generate_motion(
            curr_imgs = imgs, 
            past_velocities=trajectory[:,0],
            past_curvatures= trajectory[:,1],
            command = command,
            token= scene.scene_metadata.initial_token,
            )
        
        pred_speeds = np.array(vel_cur_pred)[:, 0]
        pred_curvatures = np.array(vel_cur_pred)[:, 1] / 100
        # pred = predict_future_waypoints(pred_speeds, pred_curvatures)
        pred = predict_future_waypoints_rk4(pred_speeds, pred_curvatures)
        traj = Trajectory(pred[1:])
        return traj, response
    
    def compute_trajectory_waypoints(self, agent_input: AgentInput, scene: Scene) -> Trajectory: 
        """
        Agent function to compute trajectory of ego vehicle in given scene by using the given waypoints rather than the speed curvature pairs.
        
        Args
        -
        self: class
            agent object
        agent_input: AgentInput
            scene data that is available for the agent to use for the trajectory prediction
        scene: Scene
            scene object that contains the scene history metadata
            
        Returns
        -
        trajectory: Trajectory
            the predicted trajectory in the Trajectory object
        response: str
            The response of the Model
            """
        imgs_t0 = [
            agent_input.cameras[-1].cam_l0.image,
            agent_input.cameras[-1].cam_f0.image,
            agent_input.cameras[-1].cam_r0.image,
            ]
        imgs_t1 = [
            agent_input.cameras[-2].cam_l0.image,
            agent_input.cameras[-2].cam_f0.image,
            agent_input.cameras[-2].cam_r0.image,
            ]
        imgs_t2 = [
            agent_input.cameras[-3].cam_l0.image,
            agent_input.cameras[-3].cam_f0.image,
            agent_input.cameras[-3].cam_r0.image,
            ]
        imgs_t3 = [
            agent_input.cameras[-4].cam_l0.image,
            agent_input.cameras[-4].cam_f0.image,
            agent_input.cameras[-4].cam_r0.image,
            ]
        imgs = [imgs_t0, imgs_t1, imgs_t2, imgs_t3]
        curr_frame = scene.scene_metadata.num_history_frames-1
        ego_history = scene.get_history_trajectory()
        ego_poses = ego_history.poses
        command = agent_input.ego_statuses[curr_frame].driving_command
        command = self.command_map.get(tuple(command)) # possibly introduces problem

        waypoints_pred, response = self.generate_motion_waypoints(
            curr_imgs = imgs, 
            past_waypoints=ego_poses,
            command = command,
            token= scene.scene_metadata.initial_token,
            )
        
        
        traj = Trajectory(np.array(waypoints_pred))
        return traj, response