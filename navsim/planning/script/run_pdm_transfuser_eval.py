import traceback
import logging
import lzma
import pickle
import hydra
import os
import re
import ast
import copy
import json
import pandas as pd


from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from hydra.utils import instantiate
from omegaconf import DictConfig
from nuplan.planning.script.builders.logging_builder import build_logger
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig, Trajectory, AgentInput
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.agents.deepseek.gpt_utils import *
from openai import OpenAI
from zoneinfo import ZoneInfo


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

myapi_key = os.environ.get("OPENAI_API_KEY")
if myapi_key is None:
    print(f"Please set OPENAI_API_KEY in your environment variables.")
    exit()
client = OpenAI(api_key=myapi_key)

def eval_trajectory(
        trajectory: Trajectory,
        input: AgentInput,
        ego_history: List[Trajectory],
        token: str,
        convert: bool = True,
    ) -> List[Union[Trajectory, bool, str]]:
    """
    Evaluates the given trajectory using GPT to see if we need to improve it
    
    Args
    -
    trajectory : Trajectory
        Trajectory to be evaluated
    input : AgentInput
        AgentInput object 
    token : str
        Scenario token for metadata
        
    Returns
    -
    trajectory : List[Trajectory, bool]
        Trajectory object with possibly improved trajectory, bool to indicate if trajectory improvement was needed
    """
    message = []
    message.append({
            "role": "developer",
            "content": f"{system_message_v5}"}
        )
    imgs = [input.cameras[-1].cam_l0.image, 
            input.cameras[-1].cam_f0.image, 
            input.cameras[-1].cam_r0.image]
    encoded_imgs = read_images(imgs)
    image_content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}} for enc in encoded_imgs] 
    if convert: 
        input_poses, prompt, history_pose = pose_to_vel_cur(trajectory.poses), correction_prompt_v2, pose_to_vel_cur(ego_history)
        past_waypoints_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in history_pose]
    else: 
        input_poses, prompt, history_pose = trajectory.poses, correction_prompt, ego_history
        past_waypoints_str = [f"[{x[0]:.1f},{x[1]:.1f},{x[2]:.1f}]" for x in history_pose]
    past_waypoints_str = ", ".join(past_waypoints_str)
    message.append({
                "role": "user",
                "content": [
                    *image_content,
                    {"type": "text", "text": f"""The ground truth history data of the ego vehicle, sampled at 0.5-second intervals, is: {past_waypoints_str}. The current trajectory provided by the expert motion planner, also spaced at 0.5-second intervals, is: {input_poses}. {prompt}"""},
                    ],
                },
            )
    
    response = client.chat.completions.create(
            model = "gpt-4.1", #"ft:gpt-4.1-2025-04-14:scania-eearp:av-finetune-7:BNKqGQNC", #
            messages = message,
            store = True,
            metadata={"token": token},
        )
    output = response.choices[0].message.content
    pattern = r"No Improvement Necessary"
    match = re.search(pattern, output, re.DOTALL)
    if match: 
        print("NO IMPROVEMENT NEEDED")
        needed = False
        traj = trajectory
    else: 
        print("IMPROVEMENT NEEDED")
        print("Original Trajectory: \n", trajectory.poses)
        print(output)
        start_index = output.find("Values")
        if start_index == -1:
            raise ValueError("No 'Values:' section found in the text.")
        values_block = output[start_index:]
        array_lines = re.findall(r"\[\s*[-+eE0-9.,\s]+\]", values_block)
        cleaned_array_text = "[" + ",".join(array_lines) + "]"
        cleaned_array_text = re.sub(r'(?<=\d)\s+(?=[\d.-])', ', ', cleaned_array_text)
        values_array = np.array(ast.literal_eval(cleaned_array_text))
        if convert:
            prediction = predict_future_waypoints_rk4(values_array[:,0], values_array[:,1])
            traj = Trajectory(prediction)
        else:
            traj = Trajectory(values_array)
        needed = True
    return [traj, needed, output]


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.

    Args
    ----------
    args: List[Dict[str, Union[List[str], DictConfig]]]
        input arguments for function
    
    Returns
    ----------
    pdm_results: List[Dict[str,Any]]
        List with dict of pdm results for all evaluation scenarios
    """

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (simulator.proposal_sampling == scorer.proposal_sampling), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    improved_pdm_results: List[Dict[str, Any]] = []
    count = 0
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)}, token={token}"
        )
        orig_score_row: Dict[str, Any] = {"token": token, "valid": True}
        improved_score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            agent_input = scene_loader.get_agent_input_from_token(token)
            scene = scene_loader.get_scene_from_token(token)
            trajectory = agent.compute_trajectory(agent_input)
            ego_history = scene.get_history_trajectory()
            improved_trajectory, needed, output = eval_trajectory(trajectory, agent_input, ego_history.poses, token)

            original_pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            orig_score_row.update(asdict(original_pdm_result))
            print("Original Score",orig_score_row)
            if needed: 
                count+=1
                improved_pdm_result = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=improved_trajectory,
                    future_sampling=simulator.proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                )
                improved_score_row.update(asdict(improved_pdm_result))
                print("Improved Score ", improved_score_row)
                improved_score_row.update({"output": output})
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            orig_score_row["valid"] = False

        pdm_results.append(orig_score_row)
        if needed: 
            improved_pdm_results.append(improved_score_row)
        else: 
            smth = copy.deepcopy(orig_score_row)
            smth.update({"output": "No Improvement Necessary"})
            improved_pdm_results.append(smth) #### FIX PROBLEM HERE
        print("")
    return pdm_results, improved_pdm_results, count


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function for running PDMS evaluation
    
    Args
    ----------
    cfg : omegaConf dictionary

    Returns
    ----------
    None
    """

    build_logger(cfg)

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    # if num_missing_metric_cache_tokens > 0:
    #     logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    # if num_unused_metric_cache_tokens > 0:
    #     logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    score_rows, improved_scores, count = run_pdm_score(data_points)

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )

    pdm_score_df = pd.DataFrame(improved_scores)
    reasoning = pdm_score_df["output"]
    tokens = pdm_score_df["token"]
    pdm_score_df.drop(columns=["output"], inplace=True)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    pdm_score_df.to_csv(save_path / f"improved_results.csv")
    data = [{"token": token, "reasoning": reason} for token, reason in zip(tokens, reasoning)]

    with open(save_path / "improved_results.jsonl", "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

    logger.info(
        f"""
        For Improved Results: 
        Number of successful scenarios: {num_sucessful_scenarios}.
        Number of failed scenarios: {num_failed_scenarios}.
        Number of scenarios that needed improvement: {count}.
        Final average score of valid results: {pdm_score_df['score'].mean()}.
        """
    )

if __name__ == "__main__":
    main()