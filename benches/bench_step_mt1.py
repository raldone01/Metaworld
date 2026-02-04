from pytest_benchmark.fixture import BenchmarkFixture

from metaworld.agent import RandomMetaworldAgent, run_agent_episode


def test_bench_step_mt1(benchmark: BenchmarkFixture):
    env_name = "reach-v3"
    seed = 42

    max_episode_steps = 500
    num_episodes = 100
    agent = RandomMetaworldAgent(seed=seed)

    def bench_x_episodes() -> None:
        for _ in range(num_episodes):
            run_agent_episode(
                env_name=env_name,
                seed=seed,
                agent=agent,
                max_episode_steps=max_episode_steps,
            )

    benchmark(bench_x_episodes)
