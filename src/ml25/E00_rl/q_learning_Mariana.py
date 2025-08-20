import gymnasium as gym
import numpy as np


class RandomAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.n

        # Q-table: estados x acciones
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

        # Hiperparámetros
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def act(self, observation):
        return self.action_space.sample()

    def step(self, state, action, reward, next_state, done=False):
        pass


class QLearningAgent(RandomAgent):
    def __init__(self, env, alpha=0.5, gamma=0.99, epsilon=1.0, min_epsilon=0.05, epsilon_decay=0.995):
        super().__init__(env, alpha, gamma, epsilon)
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def act(self, observation):
        # ε-greedy
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # ¡ojo! usar self.env
        else:
            return int(np.argmax(self.Q[observation]))

    # Update Q values using Q-learning
    def step(self, state, action, reward, next_state, done=False):
        # Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
        best_next = 0.0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def greedy_policy_from_Q(Q, n_rows=4, n_cols=12):
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}  # 0=UP,1=RIGHT,2=DOWN,3=LEFT
    policy = []
    for s in range(Q.shape[0]):
        a = int(np.argmax(Q[s]))
        policy.append(arrows[a])
    lines = []
    for r in range(n_rows):
        lines.append(" ".join(policy[r * n_cols:(r + 1) * n_cols]))
    return "\n".join(lines)


if __name__ == "__main__":
    # https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    # Para entrenar más rápido, usa render_mode=None. Para ver, usa "human".
    env = gym.make("CliffWalking-v1", render_mode=None)

    n_episodes = 1500
    max_steps = 200

    # Usa el agente QLearning (no el Random) para que aprenda
    agent = QLearningAgent(
        env,
        alpha=0.5,      # suele funcionar bien aquí
        gamma=0.99,
        epsilon=1.0,    # inicia explorando mucho
        min_epsilon=0.05,
        epsilon_decay=0.995
    )

    returns = []
    for e in range(1, n_episodes + 1):
        obs, _ = env.reset()
        ep_return = 0

        for t in range(max_steps):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(obs, action, reward, next_obs, done=done)

            ep_return += reward
            obs = next_obs
            if done:
                break

        # ↓↓↓ Reducción de exploración conforme aprende
        agent.decay_epsilon()

        returns.append(ep_return)
        if e % 100 == 0:
            print(f"Ep {e:4d} | Return medio (últ.100): {np.mean(returns[-100:]):6.2f} | ε={agent.epsilon:.3f}")

    # Muestra la política final aprendida (flechas)
    print("\nPolítica codiciosa aprendida:")
    print(greedy_policy_from_Q(agent.Q))

    # (Opcional) ver al agente ya entrenado:
    # env = gym.make("CliffWalking-v1", render_mode="human")
    # for _ in range(2):
    #     obs, _ = env.reset()
    #     done = False
    #     while not done:
    #         action = int(np.argmax(agent.Q[obs]))
    #         obs, _, terminated, truncated, _ = env.step(action)
    #         done = terminated or truncated
    #         env.render()

    env.close()
