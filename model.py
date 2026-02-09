import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_DIM = 40
NUM_ACTIONS = 12

class ActorCritic(nn.Module):
    def __init__(self, obs_dim = OBS_DIM, num_actions = NUM_ACTIONS):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        self.pi = nn.Linear(256, num_actions)

        self.v = nn.Linear(256, 1)

    def forward(self, obs):
        """
        obs: torch.float32 tensor, shape [obs_dim] or [B, obs_dim]
        returns:
          logits: [num_actions] or [B, num_actions]
          value:  scalar or [B]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.pi(x)
        value = self.v(x)
        return logits, value

