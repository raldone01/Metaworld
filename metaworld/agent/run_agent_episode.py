from typing import Any, cast

import gymnasium as gym
import numpy as np

from metaworld.agent.agent import MetaworldAgent
from metaworld.sawyer_xyz_env import SawyerXYZEnv


def _compute_required_max_episode_steps_for_lingering(
    env: gym.Env, linger_time_after_success: float | None
) -> int | None:
    if linger_time_after_success is not None and linger_time_after_success < 0:
        raise ValueError("linger_time_after_success must be positive.")

    if linger_time_after_success is None:
        return None

    dt = cast(SawyerXYZEnv, env.unwrapped).dt
    extra_linger_step_count = int(linger_time_after_success / dt)
    return extra_linger_step_count


def run_agent_episode_with_env(
    env: gym.Env,
    agent: MetaworldAgent,
    max_episode_steps: int,
    record_keys: set[str] | None = None,
    linger_steps_after_success: int | None = None,
    tqdm: Any | None = None,
    tqdm_desc: str | None = None,
) -> dict:
    if linger_steps_after_success is not None and linger_steps_after_success < 0:
        raise ValueError("linger_steps_after_success must be positive.")

    if record_keys is None:
        record_keys = set()

    actual_max_episode_steps = max_episode_steps + (linger_steps_after_success or 0)

    # If we run N steps, we record N+1 observations (initial + N results).
    buffer_size = actual_max_episode_steps + 1

    # gym standard recordings
    rec_obs = "observations" in record_keys
    rec_rewards = "rewards" in record_keys
    rec_terminates = "terminates" in record_keys
    rec_truncates = "truncates" in record_keys
    # agent recordings
    rec_agent_actions = "agent_actions" in record_keys
    # metaworld specific recordings
    rec_info_grasp_reward = "info_grasp_reward" in record_keys
    rec_info_grasp_success = "info_grasp_success" in record_keys
    rec_info_in_place_reward = "info_in_place_reward" in record_keys
    rec_info_near_object = "info_near_object" in record_keys
    rec_info_obj_to_target = "info_obj_to_target" in record_keys
    rec_info_success = "info_success" in record_keys
    rec_info_unscaled_reward = "info_unscaled_reward" in record_keys

    agent.reset()
    obs, reset_info = env.reset()
    info = reset_info

    # Preallocate arrays (floats initialized to NaN, bools to False)
    # Observations
    if rec_obs:
        observations = np.zeros((buffer_size, *obs.shape), dtype=obs.dtype)
    if rec_rewards:
        rewards = np.full(buffer_size, np.nan, dtype=np.float64)
    if rec_terminates:
        terminates = np.zeros(buffer_size, dtype=bool)
    if rec_truncates:
        truncates = np.zeros(buffer_size, dtype=bool)
    # Actions (Note: buffer_size is enough, though actions will be 1 less than obs)
    if rec_agent_actions:
        agent_actions = np.zeros(
            (buffer_size, *cast(gym.spaces.Box, env.action_space).shape),
            dtype=env.action_space.dtype,
        )
    # Metaworld Infos
    if rec_info_grasp_reward:
        info_grasp_reward_stats = np.full(buffer_size, np.nan, dtype=np.float64)
    if rec_info_grasp_success:
        info_grasp_success_stats = np.zeros(buffer_size, dtype=bool)
    if rec_info_in_place_reward:
        info_in_place_reward_stats = np.full(buffer_size, np.nan, dtype=np.float64)
    if rec_info_near_object:
        info_near_object_stats = np.zeros(buffer_size, dtype=bool)
    if rec_info_obj_to_target:
        info_obj_to_target_stats = np.full(buffer_size, np.nan, dtype=np.float64)
    if rec_info_success:
        info_success_stats = np.zeros(buffer_size, dtype=bool)
    if rec_info_unscaled_reward:
        info_unscaled_reward_stats = np.full(buffer_size, np.nan, dtype=np.float64)

    agent_first_success_step = None
    any_terminated = False
    any_truncated = False

    done = False
    agent_step = 0
    if tqdm is not None:
        tqdm_desc = tqdm_desc or f"{cast(SawyerXYZEnv, env.unwrapped).env_name} seed={reset_info['seed']}"
        pbar = tqdm(total=actual_max_episode_steps, desc=tqdm_desc)
    while True:
        # Record Pre-Step Data
        if rec_obs:
            observations[agent_step] = obs

        if done:
            break

        agent_action = agent.get_action(
            env,
            obs,
            info,
        )

        if rec_agent_actions:
            agent_actions[agent_step] = agent_action

        obs, reward, terminate, truncate, info = env.step(agent_action)
        step_is_success = info.get("success", 0.0) >= 1.0

        if terminate:
            any_terminated = True

        if step_is_success and agent_first_success_step is None:
            agent_first_success_step = agent_step

        if agent_first_success_step is not None:
            if linger_steps_after_success is not None:
                # Continue stepping until linger time is over
                if agent_step >= (agent_first_success_step + linger_steps_after_success):
                    done = True
            else:
                done = True
        elif agent_step >= (max_episode_steps - 1):
            truncate = True
            done = True
        elif truncate:
            done = True

        if truncate:
            any_truncated = True

        # Record Post-Step Data
        if rec_rewards:
            rewards[agent_step] = reward
        if rec_terminates:
            terminates[agent_step] = terminate
        if rec_truncates:
            truncates[agent_step] = truncate

        # Record Info Data
        if rec_info_grasp_reward:
            info_grasp_reward_stats[agent_step] = info.get("grasp_reward", np.nan)
        if rec_info_grasp_success:
            info_grasp_success_stats[agent_step] = info.get("grasp_success", False)
        if rec_info_in_place_reward:
            info_in_place_reward_stats[agent_step] = info.get("in_place_reward", np.nan)
        if rec_info_near_object:
            info_near_object_stats[agent_step] = info.get("near_object", False)
        if rec_info_obj_to_target:
            info_obj_to_target_stats[agent_step] = info.get("obj_to_target", np.nan)
        if rec_info_success:
            info_success_stats[agent_step] = step_is_success
        if rec_info_unscaled_reward:
            info_unscaled_reward_stats[agent_step] = info.get("unscaled_reward", np.nan)

        agent_step += 1
        if tqdm is not None:
            pbar.update(1)

    if tqdm is not None:
        pbar.close()

    # Determine slice indices
    obs_slice = agent_step + 1
    trans_slice = agent_step

    ret = {}
    if rec_obs:
        ret["observations"] = observations[:obs_slice]
    if rec_rewards:
        ret["rewards"] = rewards[:trans_slice]
    if rec_terminates:
        ret["terminates"] = terminates[:trans_slice]
    if rec_truncates:
        ret["truncates"] = truncates[:trans_slice]

    if rec_agent_actions:
        ret["agent_actions"] = agent_actions[:trans_slice]

    if rec_info_grasp_reward:
        ret["info_grasp_reward"] = info_grasp_reward_stats[:trans_slice]
    if rec_info_grasp_success:
        ret["info_grasp_success"] = info_grasp_success_stats[:trans_slice]
    if rec_info_in_place_reward:
        ret["info_in_place_reward"] = info_in_place_reward_stats[:trans_slice]
    if rec_info_near_object:
        ret["info_near_object"] = info_near_object_stats[:trans_slice]
    if rec_info_obj_to_target:
        ret["info_obj_to_target"] = info_obj_to_target_stats[:trans_slice]
    if rec_info_success:
        ret["info_success"] = info_success_stats[:trans_slice]
    if rec_info_unscaled_reward:
        ret["info_unscaled_reward"] = info_unscaled_reward_stats[:trans_slice]

    ret["env_name"] = env.unwrapped.env_name
    ret["env_seed"] = reset_info["seed"]
    ret["agent_first_success_step"] = agent_first_success_step
    ret["total_episode_steps"] = agent_step
    ret["any_terminated"] = any_terminated
    ret["any_truncated"] = any_truncated

    return ret


def run_agent_episode(
    env_name,
    seed,
    agent: MetaworldAgent,
    max_episode_steps: int,
    record_keys: set[str] | None = None,
    reward_function_version: str = "v2",
    tqdm: Any | None = None,
    tqdm_desc: str | None = None,
    linger_time_after_success: float | None = None,
    linger_steps_after_success: int | None = None,
) -> dict:
    if linger_time_after_success is not None and linger_steps_after_success is not None:
        raise ValueError("Only one of linger_time_after_success or linger_steps_after_success may be specified.")

    env = gym.make(
        "Meta-World/MT1-v3",
        env_name=env_name,
        seed=seed,
        reward_function_version=reward_function_version,
        max_episode_steps=max_episode_steps,
        num_tasks_per_env=1,
    )

    if linger_time_after_success is not None:
        linger_steps_after_success = _compute_required_max_episode_steps_for_lingering(env, linger_time_after_success)

    episode_results = run_agent_episode_with_env(
        env=env,
        agent=agent,
        max_episode_steps=max_episode_steps,
        record_keys=record_keys,
        linger_steps_after_success=linger_steps_after_success or 0,
        tqdm=tqdm,
        tqdm_desc=tqdm_desc,
    )
    env.close()
    return episode_results
