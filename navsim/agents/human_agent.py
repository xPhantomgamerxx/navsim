from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, Scene, SensorConfig

# Notably, Ego Progress (EP) cannot be solved purely by human imitation, given that the maximum progress estimate used for normalization is based on a privileged rule-based motion planner.

class HumanAgent(AbstractAgent):
    """Privileged agent interface of human operator."""

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes the human agent object.
        :param trajectory_sampling: trajectory sampling specification
        """
        self._trajectory_sampling = trajectory_sampling

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # return SensorConfig.build_no_sensors()
        return SensorConfig(cam_f0 = [3], cam_l0=True, cam_l2=False, cam_r0=True, cam_r2=False, cam_b0=False, cam_l1= False, cam_r1=False, lidar_pc=False)



    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """

        return scene.get_future_trajectory(self._trajectory_sampling.num_poses)
