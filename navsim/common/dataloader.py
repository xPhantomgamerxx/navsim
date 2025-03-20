from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path

from tqdm import tqdm
import pickle
import lzma

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
from navsim.planning.metric_caching.metric_cache import MetricCache


def filter_scenes(data_path: Path, scene_filter: SceneFilter) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load a set of scenes from dataset, while applying scene filter configuration.
    :param data_path: root directory of log folder
    :param scene_filter: scene filtering configuration class
    :return: dictionary of raw logs format
    """

    def split_list(input_list: List[Any], num_frames: int, frame_interval: int) -> List[List[Any]]:
        """Helper function to split frame list according to sampling specification."""
        return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]

    filtered_scenes: Dict[str, Scene] = {}
    stop_loading: bool = False

    # filter logs
    log_files = list(data_path.iterdir())
    if scene_filter.log_names is not None:
        log_files = [log_file for log_file in log_files if log_file.name.replace(".pkl", "") in scene_filter.log_names]

    if scene_filter.tokens is not None:
        filter_tokens = True
        tokens = set(scene_filter.tokens)
    else:
        filter_tokens = False

    for log_pickle_path in tqdm(log_files, desc="Loading logs"):

        scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
        for frame_list in split_list(scene_dict_list, scene_filter.num_frames, scene_filter.frame_interval):
            # Filter scenes which are too short
            if len(frame_list) < scene_filter.num_frames:
                continue

            # Filter scenes with no route
            if scene_filter.has_route and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0:
                continue

            # Filter by token
            token = frame_list[scene_filter.num_history_frames - 1]["token"]
            if filter_tokens and token not in tokens:
                continue

            filtered_scenes[token] = frame_list

            if (scene_filter.max_scenes is not None) and (len(filtered_scenes) >= scene_filter.max_scenes):
                stop_loading = True
                break

        if stop_loading:
            break

    return filtered_scenes


class SceneLoader:
    """Simple data loader of scenes from logs."""

    def __init__(
        self,
        data_path: Path,
        sensor_blobs_path: Path,
        scene_filter: SceneFilter,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
    ):
        """
        Initializes the scene data loader.
        :param data_path: root directory of log folder
        :param sensor_blobs_path: root directory of sensor data
        :param scene_filter: dataclass for scene filtering specification
        :param sensor_config: dataclass for sensor loading specification, defaults to no sensors
        """

        self.scene_frames_dicts = filter_scenes(data_path, scene_filter)
        self._sensor_blobs_path = sensor_blobs_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.scene_frames_dicts.keys())

    def __len__(self) -> int:
        """
        :return: number for scenes possible to load.
        """
        return len(self.tokens)

    def __getitem__(self, idx) -> str:
        """
        :param idx: index of scene
        :return: unique scene identifier
        """
        return self.tokens[idx]

    def get_scene_from_token(self, token: str) -> Scene:
        """
        Loads scene given a scene identifier string (token).
        :param token: scene identifier string.
        :return: scene dataclass
        """
        assert token in self.tokens
        return Scene.from_scene_dict_list(
            self.scene_frames_dicts[token],
            self._sensor_blobs_path,
            num_history_frames=self._scene_filter.num_history_frames,
            num_future_frames=self._scene_filter.num_future_frames,
            sensor_config=self._sensor_config,
        )

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        """
        Loads agent input given a scene identifier string (token).
        :param token: scene identifier string.
        :return: agent input dataclass
        """
        assert token in self.tokens
        return AgentInput.from_scene_dict_list(
            self.scene_frames_dicts[token],
            self._sensor_blobs_path,
            num_history_frames=self._scene_filter.num_history_frames,
            sensor_config=self._sensor_config,
        )

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        """
        Collect tokens for each logs file given filtering.
        :return: dictionary of logs names and tokens
        """
        # generate a dict that contains a list of tokens for each log-name
        tokens_per_logs: Dict[str, List[str]] = {}
        for token, scene_dict_list in self.scene_frames_dicts.items():
            log_name = scene_dict_list[0]["log_name"]
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(token)
            else:
                tokens_per_logs.update({log_name: [token]})
        return tokens_per_logs


class MetricCacheLoader:
    """Simple dataloader for metric cache."""

    def __init__(self, cache_path: Path, file_name: str = "metric_cache.pkl"):
        """
        Initializes the metric cache loader.
        :param cache_path: directory of cache folder
        :param file_name: file name of cached files, defaults to "metric_cache.pkl"
        """

        self._file_name = file_name
        self.metric_cache_paths = self._load_metric_cache_paths(cache_path)

    def _load_metric_cache_paths(self, cache_path: Path) -> Dict[str, Path]:
        """
        Helper function to load all cache file paths from folder.
        :param cache_path: directory of cache folder
        :return: dictionary of token and file path
        """
        metadata_dir = cache_path / "metadata"
        metadata_file = [file for file in metadata_dir.iterdir() if ".csv" in str(file)][0]
        with open(str(metadata_file), "r") as f:
            cache_paths = f.read().splitlines()[1:]
        metric_cache_dict = {cache_path.split("/")[-2]: cache_path for cache_path in cache_paths}
        return metric_cache_dict

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.metric_cache_paths.keys())

    def __len__(self):
        """
        :return: number for scenes possible to load.
        """
        return len(self.metric_cache_paths)

    def __getitem__(self, idx: int) -> MetricCache:
        """
        :param idx: index of cache to cache to load
        :return: metric cache dataclass
        """
        return self.get_from_token(self.tokens[idx])

    def get_from_token(self, token: str) -> MetricCache:
        """
        Load metric cache from scene identifier
        :param token: unique identifier of scene
        :return: metric cache dataclass
        """
        with lzma.open(self.metric_cache_paths[token], "rb") as f:
            metric_cache: MetricCache = pickle.load(f)
        return metric_cache

    def to_pickle(self, path: Path) -> None:
        """
        Dumps complete metric cache into pickle.
        :param path: directory of cache folder
        """
        full_metric_cache = {}
        for token in tqdm(self.tokens):
            full_metric_cache[token] = self.get_from_token(token)
        with open(path, "wb") as f:
            pickle.dump(full_metric_cache, f)
