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
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        super().__init__(env, alpha, gamma, epsilon)
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.episode_count = 0

    def act(self, observation):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  # Exploration
        else:
            return np.argmax(self.Q[observation])  # Exploitation

    # Update Q values using Q-learning
    def step(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        # Q-learning update formula
        self.Q[state][action] = self.Q[state][action] + self.alpha * (
            reward + self.gamma * self.Q[next_state][best_next_action] - self.Q[state][action]
        )

    def decay_epsilon(self):
        # Reduce exploration rate after each episode
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.episode_count += 1


if __name__ == "__main__":
    # Create environment
    env = gym.make("CliffWalking-v1", render_mode="human")
    
    # Hyperparameters
    n_episodes = 500
    episode_length = 200
    
    # Use QLearningAgent instead of RandomAgent
    agent = QLearningAgent(
        env, 
        alpha=0.1,       # Learning rate
        gamma=0.99,      # Discount factor (higher for long-term rewards)
        epsilon=1.0,     # Start with high exploration
        epsilon_decay=0.995,  # Decay rate for epsilon
        min_epsilon=0.01 # Minimum exploration rate
    )
    
    # Track performance
    returns = []
    
    for e in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0
        done = False
        
        for i in range(episode_length):
            # Get action from agent
            action = agent.act(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Update agent
            agent.step(obs, action, reward, next_obs)
            
            ep_return += reward
            obs = next_obs
            
            if done or truncated:
                break
        
        # Decay exploration rate after each episode
        agent.decay_epsilon()
        
        returns.append(ep_return)
        
        # Print progress
        if (e + 1) % 50 == 0:
            avg_return = np.mean(returns[-50:])
            print(f"Episode {e+1}/{n_episodes}, Epsilon: {agent.epsilon:.3f}, "
                  f"Return: {ep_return}, Avg Return (last 50): {avg_return:.2f}")
        else:
            print(f"Episode {e+1}: Return: {ep_return}, Epsilon: {agent.epsilon:.3f}")
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_episodes = 5
    for test_ep in range(test_episodes):
        obs, _ = env.reset()
        test_return = 0
        done = False
        
        for step in range(episode_length):
            action = np.argmax(agent.Q[obs])  # Always choose best action
            obs, reward, done, truncated, _ = env.step(action)
            test_return += reward
            env.render()
            
            if done or truncated:
                break
        
        print(f"Test Episode {test_ep+1}: Return: {test_return}")
    
    env.close()
    
    # Show final Q-table
    print("\nFinal Q-table (first few states):")
    for state in range(min(10, agent.Q.shape[0])):
        print(f"State {state}: {agent.Q[state]}")