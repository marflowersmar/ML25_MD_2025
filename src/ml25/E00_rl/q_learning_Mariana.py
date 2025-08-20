import gymnasium as gym
import numpy as np


class RandomAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.n

        # Tabla estados x acciones
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        # Parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def act(self, observation):
        return self.action_space.sample()

    def step(self, state, action, reward, next_state):
        pass


class QLearningAgent(RandomAgent):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(env, alpha, gamma, epsilon)

    def act(self, observation):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  # Exploration
        else:
            return np.argmax(self.Q[observation])  # Exploitation

    # Update Q values using Q-learning
    def step(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        # Implementa la actualización de Q-learning
        self.Q[state][action] = self.Q[state][action] + self.alpha * (
            reward + self.gamma * self.Q[next_state][best_next_action] - self.Q[state][action]
        )


if __name__ == "__main__":
    env = gym.make("CliffWalking-v1", render_mode="human")

    n_episodes = 1000
    episode_length = 200
    
    # Cambié a QLearningAgent (era RandomAgent en el original)
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    for e in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0
        
        for i in range(episode_length):
            # take action (ahora usa Q-learning, no random)
            action = agent.act(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # update agent (ahora sí aprende)
            agent.step(obs, action, reward, next_obs)

            if done or truncated:
                break
                
            ep_return += reward
            obs = next_obs
            
            env.render()
        
        # Reducción simple de exploración (como pedía el TODO)
        agent.epsilon = 0.1 / (1 + e * 0.01)  # Se reduce con cada episodio

        print(f"Episode {e} return: {ep_return}, Epsilon: {agent.epsilon:.3f}")
    
    env.close()