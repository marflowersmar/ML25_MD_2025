import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


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
            return self.env.action_space.sample()  # Exploration
        else:
            return np.argmax(self.Q[observation])  # Exploitation

    # Update Q values using Q-learning
    def step(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        # Q-learning update equation
        self.Q[state][action] = self.Q[state][action] + self.alpha * (
            reward + self.gamma * self.Q[next_state][best_next_action] - self.Q[state][action]
        )

    def decay_epsilon(self):
        # Exponential decay of epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.episode_count += 1


if __name__ == "__main__":
    # Create environment
    env = gym.make("CliffWalking-v1", render_mode="human")
    
    # Hyperparameters
    n_episodes = 500
    episode_length = 200
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor (higher for long-term rewards)
    initial_epsilon = 1.0  # Start with high exploration
    epsilon_decay = 0.995  # Decay rate per episode
    min_epsilon = 0.01  # Minimum exploration rate
    
    # Initialize agent
    agent = QLearningAgent(env, alpha=alpha, gamma=gamma, 
                          epsilon=initial_epsilon, 
                          epsilon_decay=epsilon_decay, 
                          min_epsilon=min_epsilon)
    
    # Track performance
    returns = []
    epsilons = []
    
    for e in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0
        done = False
        
        for i in range(episode_length):
            # Take action
            action = agent.act(obs)
            next_obs, reward, done, _, _ = env.step(action)
            
            # Update agent
            agent.step(obs, action, reward, next_obs)
            
            if done:
                break
            
            ep_return += reward
            obs = next_obs
        
        # Decay epsilon after each episode
        agent.decay_epsilon()
        
        # Track metrics
        returns.append(ep_return)
        epsilons.append(agent.epsilon)
        
        print(f"Episode {e} return: {ep_return}, epsilon: {agent.epsilon:.4f}")
        
        # Early stopping if agent consistently performs well
        if e > 100 and np.mean(returns[-20:]) > -20:
            print(f"Early stopping at episode {e} - agent learned successfully!")
            break
    
    env.close()
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(returns)
    plt.title('Returns per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final Q-table statistics
    print(f"\nFinal Q-table statistics:")
    print(f"Max Q-value: {np.max(agent.Q):.4f}")
    print(f"Min Q-value: {np.min(agent.Q):.4f}")
    print(f"Mean Q-value: {np.mean(agent.Q):.4f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")