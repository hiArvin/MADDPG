import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.action_out = nn.Linear(16, args.action_shape[agent_id])


    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        actions = self.action_out(x)
        # actions = F.normalize(actions,p=2,dim=-1)
        # print(actions)
        # actions = actions / torch.sum(actions,dim=-1).unsqueeze(1)
        actions = torch.softmax(actions,dim=-1)

        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.q_out = nn.Linear(32, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value





