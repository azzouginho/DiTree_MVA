import csv
import os
import time
import random
import sys
import yaml

import numpy as np
import torch

import gymnasium as gym
import gymnasium_robotics

from car_env import CarEnv
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from train_diffusion_policy import init_noise_pred_net

from policies.fm_policy import DiffusionSampler

from planners.RRT import RRT_Planner
from planners.MPC import MPC_Planner

# Loguru import and configuration
from loguru import logger

gym.register_envs(gymnasium_robotics)

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def run_experiment(planner, num_runs=100):
    success_count = 0
    runtimes = []
    iterations = []
    path_lengths = []
    path_avg_speeds = []
    number_of_nodes = []

    for _ in range(num_runs):
        planner.reset()
        path, actions = planner.plan()
        curr_iterations = planner.results["iterations"]
        runtime = planner.results["time"]
        num_nodes = planner.results["number_of_nodes"]

        path_length = 0
        avg_speed = 0

        if path is not None:
            for i in range(len(path) - 1):
                path_length += np.linalg.norm(path[i + 1, :2] - path[i, :2])
            avg_speed = np.mean(np.linalg.norm(path[:, 3:5], axis=1))
            success_count += 1

        runtimes.append(runtime)
        iterations.append(curr_iterations)
        path_lengths.append(path_length)
        path_avg_speeds.append(avg_speed)
        number_of_nodes.append(num_nodes)

    success_rate = success_count / num_runs
    return (
        success_rate,
        runtimes,
        iterations,
        path_lengths,
        path_avg_speeds,
        number_of_nodes,
    )


def evaluate_all_scenarios(
    mazes_dir,
    scenarios_file,
    cfg_file,
    total_runs=100,
    time_budget=60,
    diffusion_sampler_checkpoints=None,
):
    # Settings
    logger.info(f"Starting evaluation for config: {cfg_file}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    unet_dims = {
        "small": [64, 128, 256],
        "medium": [256, 512, 1024],
        "large": [512, 1024, 2048],
        "xlarge": [1024, 2048, 4096],
    }

    cfg_path = f"cfgs/{cfg_file}.yaml"
    if not os.path.exists(cfg_path):
        logger.error(f"Config file not found: {cfg_path}")
        return

    with open(cfg_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    debug = loaded_config.get("debug", False)
    prediction_type = loaded_config.get("prediction_type", "actions")
    obs_history = loaded_config.get("obs_history", 1)
    action_history = loaded_config.get("action_history", 1)
    position_conditioned = False
    goal_conditioned = loaded_config.get("goal_conditioned", True)
    local_map_conditioned = loaded_config.get("local_map_conditioned", True)
    local_map_size = loaded_config.get("local_map_size", 20)
    local_map_scale = loaded_config.get("local_map_scale", 0.2)
    local_map_embedding_dim = loaded_config.get("local_map_embedding_dim", 400)
    env_id = loaded_config.get("env_id", "carmaze")
    # policy = loaded_config.get("policy", "flow_matching")
    num_diffusion_iters = loaded_config.get("planning_diffusion_iters", 5)
    unet_down_dims = unet_dims[loaded_config.get("denoiser_size", "large")]
    pred_horizon = loaded_config.get("pred_horizon", 64)
    action_horizon = loaded_config.get("action_horizon", 8)
    goal_conditioning_bias = loaded_config.get("goal_conditioning_bias", 0.85)
    prop_duration = loaded_config.get("prop_duration", [64])

    logger.info(f"Environment ID: {env_id}")

    goal_dim = 2
    s_global = 1.0

    if "antmaze" in env_id.lower():
        obs_dim = 31  # 6D rotation representation
        if not position_conditioned:
            obs_dim -= 2  # remove (x,y)
        action_dim = 8
        s_global = 4.0
        action_horizon = 2
        local_map_scale = 0.8
    elif "pointmaze" in env_id.lower():
        obs_dim = 4
        if not position_conditioned:
            obs_dim -= 2  # remove (x,y)
        action_dim = 2
    elif "dronemaze" in env_id.lower():
        obs_dim = 10
        if not position_conditioned:
            obs_dim -= 2  # remove (x,y)
        action_dim = 4
    elif "car" in env_id.lower():
        full_obs_dim = 6
        obs_dim = full_obs_dim
        if not position_conditioned:
            obs_dim -= 3  # remove (x,y,theta)
        action_dim = 2
    else:
        raise ValueError(f"Invalid env_id: {env_id}")

    render_mode = "human" if debug else "rgb_array"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    output_dir = "checkpoints/"

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    noise_pred_net = init_noise_pred_net(
        input_dim=action_dim if prediction_type == "actions" else full_obs_dim,
        action_dim=action_dim,
        obs_dim=obs_dim,
        obs_history=obs_history,
        action_history=action_history,
        goal_conditioned=goal_conditioned,
        goal_dim=goal_dim,
        local_map_conditioned=local_map_conditioned,
        local_map_encoder="resnet",
        local_map_embedding_dim=local_map_embedding_dim,
        local_map_size=local_map_size,
        down_dims=unet_down_dims,
    )

    # --------------------------------------------------------------------------
    # CHECKPOINT LOADING FIX
    # --------------------------------------------------------------------------
    checkpoint_path = output_dir + diffusion_sampler_checkpoints["resnet"]
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    try:
        # Load checkpoint (mapped to correct device)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        target_key = "noise_pred_net_state_dict"

        if isinstance(checkpoint, dict):
            # Check if our expected key exists
            if target_key in checkpoint:
                noise_pred_net.load_state_dict(checkpoint[target_key])
                logger.success(f"Loaded state dict using key: '{target_key}'")
            else:
                # Key missing, list available keys and try fallbacks
                available_keys = list(checkpoint.keys())
                logger.warning(
                    f"Key '{target_key}' not found. Available keys: {available_keys}"
                )

                # Fallback logic
                if "state_dict" in checkpoint:
                    noise_pred_net.load_state_dict(checkpoint["state_dict"])
                    logger.success("Fallback: Loaded using 'state_dict'")
                elif "model_state_dict" in checkpoint:
                    noise_pred_net.load_state_dict(checkpoint["model_state_dict"])
                    logger.success("Fallback: Loaded using 'model_state_dict'")
                else:
                    logger.error(
                        "Could not find a valid model state dictionary in checkpoint."
                    )
                    raise KeyError(
                        f"Expected '{target_key}' or similar, found {available_keys}"
                    )
        else:
            # Checkpoint is the model object itself
            logger.warning(
                "Checkpoint is not a dictionary. Attempting to use as full model object."
            )
            noise_pred_net = checkpoint

    except Exception as e:
        logger.exception("Failed to load checkpoint.")
        raise e
    # --------------------------------------------------------------------------

    noise_pred_net = noise_pred_net.to(device).eval()

    diffusion_sampler_small_resnet = DiffusionSampler(
        noise_pred_net,
        noise_scheduler,
        env_id,
        policy="flow_matching",
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        prediction_type=prediction_type,
        obs_history=obs_history,
        action_history=action_history,
        goal_conditioned=True,
        num_diffusion_iters=num_diffusion_iters,
        local_map_size=local_map_size,
    ).eval()

    os.makedirs("benchmark_results", exist_ok=True)

    # Load the scenarios from the CSV file
    with open(scenarios_file, mode="r") as scenarios_csv:
        scenarios_reader = csv.reader(scenarios_csv)
        next(scenarios_reader)  # Skip the header

        # Iterate through all scenarios
        for scenario in scenarios_reader:
            if "car" in env_id.lower():
                (
                    scenario_name,
                    maze_name,
                    start_row,
                    start_col,
                    start_deg,
                    goal_row,
                    goal_col,
                ) = scenario
            else:
                scenario_name, maze_name, start_row, start_col, goal_row, goal_col = (
                    scenario
                )

            # Load the corresponding maze
            maze_path = os.path.join(mazes_dir, f"{maze_name}.csv")
            if not os.path.exists(maze_path):
                logger.warning(f"Maze file {maze_path} not found, skipping scenario.")
                continue

            maze_data = np.loadtxt(maze_path, delimiter=",")

            # Create a new environment for each maze with the maze data
            if "pointmaze" in env_id.lower():
                env = gym.make(
                    "PointMaze_Large-v3", maze_map=maze_data, render_mode=render_mode
                )
                start_xy = env.maze.cell_rowcol_to_xy(
                    np.array([int(start_row), int(start_col)])
                )
                goal_xy = env.maze.cell_rowcol_to_xy(
                    np.array([int(goal_row), int(goal_col)])
                )
                start = np.array([start_xy[0], start_xy[1], 0.0, 0.0])
                goal = np.array([goal_xy[0], goal_xy[1], 0.0, 0.0])
            elif "antmaze" in env_id.lower():
                env = gym.make(
                    "AntMaze_Large-v4", maze_map=maze_data, render_mode=render_mode
                )
                start_xy = env.maze.cell_rowcol_to_xy(
                    np.array([int(start_row), int(start_col)])
                )
                goal_xy = env.maze.cell_rowcol_to_xy(
                    np.array([int(goal_row), int(goal_col)])
                )
                start = np.zeros(obs_dim)
                start[:2] = start_xy
                start[2] = 0.75
                start[3] = 1.0
                goal = np.zeros(obs_dim)
                goal[:2] = goal_xy
            elif "dronemaze" in env_id.lower():
                # Assuming DroneEnv is imported or available in namespace, though not in original imports provided.
                # If needed, ensure 'from drone_env import DroneEnv' is at top
                try:
                    from drone_env import DroneEnv

                    env = DroneEnv(maze_map=maze_data, collision_checking=False)
                except ImportError:
                    logger.error("DroneEnv not found. Skipping drone scenario.")
                    continue

                start_xy = env.cell_rowcol_to_xy(
                    np.array([int(start_row), int(start_col)])
                )
                goal_xy = env.cell_rowcol_to_xy(
                    np.array([int(goal_row), int(goal_col)])
                )
                start = np.array(
                    [start_xy[0], start_xy[1], 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                )
                goal = np.array(
                    [goal_xy[0], goal_xy[1], 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                )
            elif "car" in env_id.lower():
                env = CarEnv(maze_map=maze_data, collision_checking=False)
                start_xy = env.cell_rowcol_to_xy(
                    np.array([int(start_row), int(start_col)])
                )
                start_rad = np.deg2rad(float(start_deg))
                goal_xy = env.cell_rowcol_to_xy(
                    np.array([int(goal_row), int(goal_col)])
                )
                start = np.array([start_xy[0], start_xy[1], start_rad, 0.0, 0.0, 0.0])
                goal = np.array([goal_xy[0], goal_xy[1], 0.0, 0.0, 0.0, 0.0])

            # Initialize Planners
            diffusion_RRT = RRT_Planner(
                start,
                goal,
                env_id=env_id,
                environment=env,
                sampler=diffusion_sampler_small_resnet,
                prediction_type=prediction_type,
                action_horizon=action_horizon,
                local_map_size=local_map_size,
                local_map_scale=local_map_scale,
                global_map_scale=s_global,
                goal_conditioning_bias=goal_conditioning_bias,
                prop_duration=prop_duration,
                time_budget=time_budget,
                max_iter=300,
                verbose=True,
            )

            Diffuser_MPC = MPC_Planner(
                start,
                goal,
                env_id=env_id,
                environment=env,
                sampler=diffusion_sampler_small_resnet,
                prediction_type=prediction_type,
                action_horizon=action_horizon,
                local_map_size=local_map_size,
                local_map_scale=local_map_scale,
                global_map_scale=s_global,
                time_budget=time_budget,
                verbose=True,
            )

            # Define which planners to run
            planners = [
                ("diffusion_RRT_PD64", diffusion_RRT),
                # ("Diffuser_MPC", Diffuser_MPC),
            ]

            for planner_name, planner in planners:
                logger.info(
                    f"Running scenario {scenario_name} with planner {planner_name}..."
                )

                # Prepare CSV output for each scenario and planner
                scenario_output_csv = (
                    f"benchmark_results/{scenario_name}_{planner_name}_{env_id}.csv"
                )

                existing_rows = 0
                if os.path.exists(scenario_output_csv):
                    with open(scenario_output_csv, mode="r", newline="") as file:
                        reader = csv.reader(file)
                        rows = list(reader)
                        if len(rows) > 1:  # Exclude the header
                            existing_rows = len(rows) - 1

                remaining_runs = total_runs - existing_rows

                if remaining_runs <= 0:
                    logger.info(
                        f"CSV already contains {existing_rows} rows. No additional runs needed."
                    )
                else:
                    logger.info(
                        f"CSV contains {existing_rows} rows. Running {remaining_runs} more iterations."
                    )

                    # If the file is empty, create it and add headers
                    if existing_rows == 0:
                        with open(scenario_output_csv, mode="w", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(
                                [
                                    "iteration",
                                    "success",
                                    "runtime",
                                    "trajectory_length",
                                    "trajectory_time",
                                    "avg_velocity",
                                    "num_states_in_tree",
                                    "num_RRT_iterations",
                                    "ctrl_effort_max",
                                    "ctrl_effort_mean",
                                    "ctrl_effort_std",
                                ]
                            )

                    # Run only the remaining required iterations
                    for i in range(existing_rows, existing_rows + remaining_runs):
                        logger.info(
                            f"Scenario: {scenario_name} | Iteration: {i + 1}/{total_runs}"
                        )

                        start_time = time.time()
                        planner.reset()
                        path_array, actions = planner.plan()
                        end_time = time.time()

                        # Collect statistics
                        planner.visualize_tree(path_array)
                        runtime = end_time - start_time
                        num_states_in_tree = planner.results.get("number_of_nodes", 0)
                        num_iterations = planner.results.get("iterations", 0)

                        # Variables to populate
                        success = 0
                        trajectory_length = -1
                        avg_velocity = -1
                        trajectory_time = 0
                        ctrl_effort_max = -1
                        ctrl_effort_mean = -1
                        ctrl_effort_std = -1

                        if path_array is not None:
                            # Save path trace
                            np.savetxt(
                                f"path_DP_{i}.csv",
                                path_array,
                                fmt="%.6f",
                                delimiter=",",
                            )
                            success = 1
                            trajectory_time = planner.results.get("path_time", 0)

                            try:
                                trajectory_length = calculate_trajectory_length(
                                    path_array
                                )
                                avg_velocity = calculate_average_velocity(path_array)
                                if actions is not None and len(actions) > 0:
                                    ctrl_effort = np.linalg.norm(actions, axis=1)
                                    ctrl_effort_max = np.max(ctrl_effort)
                                    ctrl_effort_mean = np.mean(ctrl_effort)
                                    ctrl_effort_std = np.std(ctrl_effort)
                                else:
                                    (
                                        ctrl_effort_max,
                                        ctrl_effort_mean,
                                        ctrl_effort_std,
                                    ) = 0, 0, 0
                            except Exception as ex:
                                success = -1
                                logger.error(
                                    f"Error calc metrics for {scenario_name}, planner {planner_name}, run {i}: {ex}"
                                )
                                # Defaults on error
                                trajectory_length = 0
                                avg_velocity = 0
                                ctrl_effort_max, ctrl_effort_mean, ctrl_effort_std = (
                                    0,
                                    0,
                                    0,
                                )
                        else:
                            success = 0

                        # Note: 'total_cc' was referenced in original code but not defined in scope.
                        # Removed the print line referring to total_cc to prevent NameError.

                        # Write results to CSV
                        with open(scenario_output_csv, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(
                                [
                                    i + 1,
                                    success,
                                    runtime,
                                    trajectory_length,
                                    trajectory_time,
                                    avg_velocity,
                                    num_states_in_tree,
                                    num_iterations,
                                    ctrl_effort_max,
                                    ctrl_effort_mean,
                                    ctrl_effort_std,
                                ]
                            )


def calculate_trajectory_length(path_array):
    return np.sum(np.linalg.norm(np.diff(path_array[:, :2], axis=0), axis=1))


def calculate_average_velocity(path_array):
    if path_array.shape[1] < 4:
        return 0.0
    velocities = np.sqrt(np.square(path_array[:, 2]) + np.square(path_array[:, 3]))
    return np.mean(velocities)


if __name__ == "__main__":
    ## Carmaze
    diffusion_sampler_checkpoints = {
        "resnet": "carmaze.pt",
        # 'resnet': 'antmaze.pt',
    }

    # scenario_file = 'experiments/test_scenarios_ant.csv'
    scenario_file = "experiments/test_scenarios_car.csv"

    # Use a try-except block for the main execution to catch unexpected crashes
    try:
        for cfg_file in ["carmaze"]:  # /'antmaze'
            evaluate_all_scenarios(
                "maps/mazes",
                scenario_file,  # test_scenarios.csv
                cfg_file=cfg_file,
                total_runs=10,
                time_budget=120,
                diffusion_sampler_checkpoints=diffusion_sampler_checkpoints,
            )
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
    except Exception as e:
        logger.exception("An unhandled exception occurred during execution.")
