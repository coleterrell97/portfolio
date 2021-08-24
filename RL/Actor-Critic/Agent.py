from GenericNetwork import GenericNetwork
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch as T

class Agent():
    def __init__(self, observation_dimensions, layer1_dimensions, layer2_dimensions, action_dimensions, ALPHA, GAMMA):
        self.actor = GenericNetwork(observation_dimensions, layer1_dimensions, layer2_dimensions, action_dimensions, ALPHA)
        self.critic = GenericNetwork(observation_dimensions, layer1_dimensions, layer2_dimensions, 1, ALPHA)

        self.GAMMA = GAMMA
        self.log_probability = None

    def select_action(self, observation):
        logits = self.actor(observation)
        probabilities = F.softmax(logits, dim = 0)
        probability_distribution = Categorical(probabilities)
        action = probability_distribution.sample()
        self.log_probability = probability_distribution.log_prob(action)
        return action.item()

    def learn(self, observation, action, reward, new_observation, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        G = self.critic(observation)
        G_ = self.critic(new_observation)

        delta = (reward + self.GAMMA * G_ * (1-int(done))) - G
        critic_loss = delta**2
        actor_loss = -self.log_probability * delta
        loss = critic_loss + actor_loss
        loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
