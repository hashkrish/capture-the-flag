import random
import numpy as np
import gym
from multiprocessing import Process, Manager


# Create a custom gym environment
class CaptureTheFlagEnv(gym.Env):
    def __init__(self):
        # Initialize game parameters
        self.territory_size = 10
        self.num_players_per_team = 1
        self.num_teams = 2
        self.flag_locations = [0, self.territory_size - 1]
        self.action_space = gym.spaces.Discrete(2)  # Move left or right
        self.observation_space = gym.spaces.Discrete(self.territory_size)
        self.state = None

    def reset(self):
        # Initialize game state
        self.state = [
            random.randint(0, self.territory_size - 1)
            for _ in range(self.num_teams * self.num_players_per_team)
        ]
        return self.state

    def step(self, action):
        # Update game state based on action
        new_state = self.state.copy()
        for i in range(self.num_teams * self.num_players_per_team):
            if action == 0:  # Move left
                new_state[i] = max(0, new_state[i] - 1)
            elif action == 1:  # Move right
                new_state[i] = min(self.territory_size - 1, new_state[i] + 1)

        # Determine rewards
        rewards = [0] * (self.num_teams * self.num_players_per_team)
        for i, location in enumerate(new_state):
            if location in self.flag_locations:
                rewards[i] = 1

        # Update state and return
        self.state = new_state
        return self.state, rewards, False, {}


# Create Q-learning agent
class QLearningAgent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.q_table = np.zeros((observation_space.n, action_space.n))

    def choose_action(self, state):
        return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        max_next_q = np.max(self.q_table[next_state, :])
        self.q_table[state, action] = (1 - alpha) * self.q_table[
            state, action
        ] + alpha * (reward + gamma * max_next_q)


def train_worker(agent, env, episode_range, alpha, gamma, result_list):
    for episode in episode_range:
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            actions = [agent.choose_action(s) for s in state]
            next_state, rewards, done, _ = env.step(actions)
            total_reward += sum(rewards)

            for i, reward in enumerate(rewards):
                agent.update_q_table(
                    state[i], actions[i], reward, next_state[i], alpha, gamma
                )

            state = next_state

        result_list[episode] = total_reward


if __name__ == "__main__":
    env = CaptureTheFlagEnv()
    agent = QLearningAgent(env.action_space, env.observation_space)
    num_episodes = 1000
    alpha = 0.1
    gamma = 0.99

    num_processes = 4  # Number of parallel processes
    episodes_per_process = num_episodes // num_processes

    with Manager() as manager:
        result_list = manager.list([0] * num_episodes)
        processes = []

        for i in range(num_processes):
            start_episode = i * episodes_per_process
            end_episode = start_episode + episodes_per_process
            process = Process(
                target=train_worker,
                args=(
                    agent,
                    env,
                    range(start_episode, end_episode),
                    alpha,
                    gamma,
                    result_list,
                ),
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        # Display results
        total_rewards = list(result_list)
        print("Total rewards per episode:", total_rewards)

    print("Training complete!")
