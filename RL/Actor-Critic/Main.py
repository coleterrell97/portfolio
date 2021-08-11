from Agent import Agent
import gym


env = gym.make("CartPole-v0")
observation_dimensions = env.observation_space.shape[0]
action_dimensions = env.action_space.n
agent = Agent(observation_dimensions, 256, 256, action_dimensions, 0.0003, 0.99)

EPISODES = 2500
HISTORY = []

done = False

for episode in range(EPISODES):
    observation = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.learn(observation, action, reward, observation_, done)
        observation = observation_
        episode_reward += reward
    HISTORY.append(episode_reward)
    print(f"Episode {episode} -- Reward: {episode_reward}")
