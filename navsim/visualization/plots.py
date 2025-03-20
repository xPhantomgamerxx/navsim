from typing import Any, Callable, List, Tuple
import io

from math import atan2, cos, sin, sqrt, radians, pi

from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from navsim.agents.abstract_agent import AbstractAgent
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from navsim.common.dataclasses import Scene
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, CAMERAS_PLOT_CONFIG
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax, add_map_to_bev_ax, add_annotations_to_bev_ax_with_offset, add_trajectory_to_bev_ax_with_offset
from navsim.visualization.camera import add_annotations_to_camera_ax, add_lidar_to_camera_ax, add_camera_ax

def transform_pose(pose_ego,transform_global):
    """
    Transforms a 2D pose from ego coordinates to global coordinates.

    :param pose_ego: List or tuple [x, y, theta] representing the pose in ego coordinates.
    :param transform_global: List or tuple [X_g, Y_g, Theta_g] representing the transformation in global coordinates.
    :return: Transformed pose [x', y', theta'] in global coordinates.
    """
    x_e, y_e, theta_e = pose_ego
    X_g, Y_g, Theta_g = transform_global

    # Compute the transformed coordinates
    x_g = X_g + x_e * cos(Theta_g) - y_e * sin(Theta_g)
    y_g = Y_g + x_e * sin(Theta_g) + y_e * cos(Theta_g)
    theta_g = Theta_g + theta_e  # Add heading angles
    theta_g = (theta_g + pi) % (2 * pi) - pi  # Normalize theta to be within [-π, π]

    return [x_g, y_g, theta_g]


def relative_transform(transform1, transform2):
    """
    Computes the relative transform T_rel such that transform1 ∘ T_rel = transform2.

    :param transform1: (x1, y1, theta1) - The base transformation
    :param transform2: (x2, y2, theta2) - The target transformation
    :return: (x_rel, y_rel, theta_rel) - The relative transformation
    """
    x1, y1, theta1 = transform1
    x2, y2, theta2 = transform2

    # Compute inverse of T1
    cos_theta1 = cos(-theta1)
    sin_theta1 = sin(-theta1)

    # Apply inverse transform of T1 to T2
    x_rel = (x2 - x1) * cos_theta1 + (y2 - y1) * sin_theta1
    y_rel = -(x2 - x1) * sin_theta1 + (y2 - y1) * cos_theta1
    theta_rel = theta2 - theta1

    # Normalize theta to be within [-π, π]
    theta_rel = (theta_rel + pi) % (2 * pi) - pi

    return x_rel, y_rel, theta_rel


def transform_difference(transform1, transform2):
    """
    Computes the difference between two 2D transforms (x, y, heading).
    
    :param transform1: List or tuple [x1, y1, theta1] (reference frame).
    :param transform2: List or tuple [x2, y2, theta2] (pose to compare).
    :return: Relative transform [dx, dy, dtheta] representing transform2 relative to transform1.
    """
    x1, y1, theta1 = transform1
    x2, y2, theta2 = transform2

    # Compute the difference in position
    dx = x2 - x1
    dy = y2 - y1

    # Rotate into the reference frame of transform1
    dx_rel = dx * cos(-theta1) - dy * sin(-theta1)
    dy_rel = dx * sin(-theta1) + dy * cos(-theta1)

    # Compute the difference in heading
    dtheta = (theta2-theta1 + pi) % (2 * pi) - pi

    return [dx_rel, dy_rel, dtheta]

def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the plt ax object for birds-eye-view plots
    :param ax: matplotlib ax object
    :return: configured ax object
    """

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")

    # NOTE: x forward, y sideways
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)

    # NOTE: left is y positive, right is y negative
    ax.invert_xaxis()

    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def configure_all_ax(ax: List[List[plt.Axes]]) -> List[List[plt.Axes]]:
    """
    Iterates through 2D ax list/array to apply configurations
    :param ax: 2D list/array of matplotlib ax object
    :return: configure axes
    """
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            configure_ax(ax[i][j])

    return ax


def plot_bev_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, plt.Axes]:
    """
    General plot for birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_bev_with_agent(scene: Scene, agent: AbstractAgent) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory()
    if agent.requires_scene:
        agent_trajectory = agent.compute_trajectory(scene.get_agent_input(), scene)
    else:
        agent_trajectory = agent.compute_trajectory(scene.get_agent_input())
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax

def plot_bev_with_agent_with_traj(scene: Scene, agent: AbstractAgent, frame_idx, human_trajectory, agent_trajectory) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])

    init_pose = scene.frames[0].ego_status.ego_pose
    ego_pose = scene.frames[frame_idx].ego_status.ego_pose
    predicted_pose = [*agent_trajectory.poses[frame_idx]]
    transfuser_global_pose = transform_pose(predicted_pose, init_pose)
    # print(f"Ego pose: {ego_pose}, Predicted pose: {predicted_pose}, Transfuser global pose: {transfuser_global_pose}")
    # print(f"Diff: {transform_difference(ego_pose, transfuser_global_pose)}")
    rel_pose = relative_transform(ego_pose, transfuser_global_pose) #### relative pose calculated wrong here
    temp = [0,0,0]
    add_map_to_bev_ax(ax, scene.map_api, StateSE2(*transfuser_global_pose)) # move plot to where ego vehicle is according to prediction
    add_annotations_to_bev_ax_with_offset(ax, scene.frames[frame_idx].annotations, predicted_pose, temp)
    add_trajectory_to_bev_ax_with_offset(ax, human_trajectory, temp, TRAJECTORY_CONFIG["human"] )
    add_trajectory_to_bev_ax_with_offset(ax, agent_trajectory, temp, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)
    return fig, ax


def plot_cameras_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_camera_ax(ax[0, 0], frame.cameras.cam_l0)
    add_camera_ax(ax[0, 1], frame.cameras.cam_f0)
    add_camera_ax(ax[0, 2], frame.cameras.cam_r0)

    add_camera_ax(ax[1, 0], frame.cameras.cam_l1)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_camera_ax(ax[1, 2], frame.cameras.cam_r1)

    add_camera_ax(ax[2, 0], frame.cameras.cam_l2)
    add_camera_ax(ax[2, 1], frame.cameras.cam_b0)
    add_camera_ax(ax[2, 2], frame.cameras.cam_r2)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_lidar(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the lidar pc) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_lidar_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.lidar)

    add_lidar_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.lidar)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_lidar_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.lidar)

    add_lidar_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.lidar)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_annotations(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the bounding boxes) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_annotations_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.annotations)

    add_annotations_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.annotations)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_annotations_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.annotations)

    add_annotations_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.annotations)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def frame_plot_to_pil(callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],scene: Scene,frame_indices: List[int],) -> List[Image.Image]:
    """
    Plots a frame according to plotting function and return a list of PIL images
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices to save
    :return: list of PIL images
    """

    images: List[Image.Image] = []

    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        fig, ax = callable_frame_plot(scene, frame_idx)

        # Creating PIL image from fig
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

    return images

def frame_plot_to_pil_with_traj(callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],agent,scene: Scene,frame_indices: List[int],) -> List[Image.Image]:
    """
    Plots a frame according to plotting function and return a list of PIL images
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices to save
    :return: list of PIL images
    """

    images: List[Image.Image] = []
    agent_trajectory = agent.compute_trajectory(scene.get_agent_input())
    ground_truth_trajectory = scene.get_future_trajectory(len(agent_trajectory.poses))

    for  frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        frame_idx = 1
        fig, ax = callable_frame_plot(scene, agent, frame_idx, ground_truth_trajectory, agent_trajectory)

        # Creating PIL image from fig
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

    return images

def frame_plot_to_gif(
    file_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
    duration: float = 500,
) -> None:
    """
    Saves a frame-wise plotting function as GIF (hard G)
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices
    :param file_name: file path for saving to save
    :param duration: frame interval in ms, defaults to 500
    """
    images = frame_plot_to_pil(callable_frame_plot, scene, frame_indices)
    images[0].save(file_name, save_all=True, append_images=images[1:], duration=duration, loop=0)

def frame_plot_to_gif_with_traj(
        filename: str,
        callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
        scene: Scene,
        agent,
        frame_indices: List[int],
        duration: float = 500,
) -> None:
    """
    Saves a frame-wise plotting function as GIF (hard G)"""

    images = frame_plot_to_pil_with_traj(callable_frame_plot, agent, scene, frame_indices)
    images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=0)