from metaworld.agent import RandomMetaworldAgent, run_agent_episode

if __name__ == "__main__":
    env_name = "reach-v3"
    seed = 42

    agent = RandomMetaworldAgent(seed=seed)

    max_episode_steps = 500
    num_episodes = 50

    for episode in range(num_episodes):
        run_agent_episode(
            env_name=env_name,
            seed=seed,
            agent=agent,
            max_episode_steps=max_episode_steps,
        )
