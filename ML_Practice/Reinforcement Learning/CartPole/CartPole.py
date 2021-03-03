import CartPoleNN
import gym
import torch
from torch import nn
from collections import namedtuple
import numpy as np

BATCH_SIZE = 16
PERCENTILE = 70
env = gym.make("CartPole-v0")
model = CartPoleNN.Net(env.observation_space.shape[0], 128, env.action_space.n)
#env = gym.wrappers.Monitor(env, directory="mon", force=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
Episode = namedtuple("Episode", field_names = ["Reward", "Steps"])
EpisodeStep = namedtuple("EpisodeStep", field_names = ["Observation", "Action"])

def execute_episode(env, model):
    current_episode = []
    current_obs = env.reset()
    total_reward = 0.0
    done = False
    sm = nn.Softmax(dim=0)
    while done == False:
        current_obs_v = torch.FloatTensor(current_obs)
        action_probabilities = sm(model(current_obs_v))
        action_probabilities_np = action_probabilities.detach().numpy()
        action = np.random.choice(len(action_probabilities_np), p=action_probabilities_np)
        current_episode.append(EpisodeStep(Observation = current_obs, Action = action))
        current_obs, reward, done, _ = env.step(action)
        total_reward += reward
    return Episode(Reward = total_reward, Steps = current_episode)

def execute_batch(batch_size, env, model):
    batch_episodes = []
    for episode in range(batch_size):
        batch_episodes.append(execute_episode(env, model))
    return batch_episodes

def filter_elite_episodes(batch, percentile):
    rewards = list(map(lambda s: s.Reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    mean_reward = np.mean(rewards)
    train_obs = []
    train_acts = []
    for episode in batch:
        if episode.Reward < reward_bound:
            continue
        else:
            train_acts.extend(map(lambda s: s.Action, episode.Steps))
            train_obs.extend(map(lambda s: s.Observation, episode.Steps))
        return torch.FloatTensor(train_obs), torch.LongTensor(train_acts), mean_reward

def train(train_obs, train_acts, loss_fn, optimizer):
    predicted_actions = model(train_obs)
    optimizer.zero_grad()
    loss = loss_fn(predicted_actions, train_acts)
    loss.backward()
    optimizer.step()
    print(loss)

mean_reward = 0
while mean_reward < 200:
    batch = execute_batch(BATCH_SIZE, env, model)
    train_obs, train_acts, mean_reward = filter_elite_episodes(batch, PERCENTILE)
    train(train_obs, train_acts, loss_fn, optimizer)
    print(mean_reward)
