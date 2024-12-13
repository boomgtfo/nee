import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define environment parameters
NUM_RESOURCES = 10
MAX_CAPACITY = 100

class CloudEdgeEnv:
    def __init__(self):
        self.state = np.random.randint(0, MAX_CAPACITY, NUM_RESOURCES)
        self.demand = np.random.randint(10, 50, NUM_RESOURCES)
        self.done = False

    def reset(self):
        self.state = np.random.randint(0, MAX_CAPACITY, NUM_RESOURCES)
        self.demand = np.random.randint(10, 50, NUM_RESOURCES)
        self.done = False
        return self.state

    def step(self, action):
        self.state = self.state.astype(np.float64)  # 将 self.state 转为 float64
        self.state -= self.demand * action  # 确保运算结果的数据类型一致
        reward = -np.sum(self.state ** 2)  # 示例奖励函数
        done = np.all(self.state <= 0)  # 示例终止条件
        return self.state, reward, done


# DQN Agent
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(64, input_dim=NUM_RESOURCES, activation='relu'),
            Dense(64, activation='relu'),
            Dense(NUM_RESOURCES, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.rand(NUM_RESOURCES)
        q_values = self.model.predict(state.reshape(1, -1))
        return q_values[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main Training Loop
env = CloudEdgeEnv()
agent = DQNAgent()
EPISODES = 500

for e in range(EPISODES):
    state = env.reset()
    total_reward = 0
    for time in range(200):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode {e + 1}/{EPISODES}, Reward: {total_reward}")
            break
    if len(agent.memory) > 32:
        agent.replay(32)
