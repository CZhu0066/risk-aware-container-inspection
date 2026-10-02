import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional, Union
import numpy as np


class RewardModel(nn.Module):

    def __init__(self, state_dim: int = 6, hidden_dim: int = 64):
        super(RewardModel, self).__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:


        if action.dim() == 0:
            action = action.unsqueeze(0).unsqueeze(0)
        elif action.dim() == 1:
            action = action.unsqueeze(1)

        x = torch.cat([state.float(), action.float()], dim=-1)
        return self.net(x).squeeze(-1)

    def forward_pair(
            self,
            state: torch.Tensor,
            chosen_action: torch.Tensor,
            rejected_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        r_chosen = self.forward(state, chosen_action)
        r_rejected = self.forward(state, rejected_action)
        return r_chosen, r_rejected

    def compute_bt_loss(
            self,
            state: torch.Tensor,
            chosen_action: torch.Tensor,
            rejected_action: torch.Tensor,
            weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        r_chosen, r_rejected = self.forward_pair(state, chosen_action, rejected_action)


        raw_loss = -F.logsigmoid(r_chosen - r_rejected)

        if weight is not None:
            return (raw_loss * weight.float()).mean()
        return raw_loss.mean()

    def compute_reward_margin(
            self,
            state: torch.Tensor
    ) -> torch.Tensor:


        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.size(0)

        action_0 = torch.zeros(
            batch_size,
            device=state.device,
            dtype=torch.float32
        )
        action_1 = torch.ones(
            batch_size,
            device=state.device,
            dtype=torch.float32
        )

        r_0 = self.forward(state, action_0)
        r_1 = self.forward(state, action_1)
        # Reward Margin
        margin = r_1 - r_0
        return margin

    def compute_bounded_margin(
            self,
            state: torch.Tensor,
            temperature: float = 1.0
    ) -> torch.Tensor:

        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        # 原始 Reward Margin
        raw_margin = self.compute_reward_margin(state)
        # BT 偏好概率
        preference_prob = torch.sigmoid(
            raw_margin / temperature
        )
        # 映射到 [-1, 1]
        bounded_margin = 2.0 * preference_prob - 1.0
        return bounded_margin

    def compute_action_reward(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            temperature: float = 1.0
    ) -> torch.Tensor:

        bounded_margin = self.compute_bounded_margin(
            state,
            temperature=temperature
        )

        if action.dim() == 0:
            action = action.unsqueeze(0)
        action = action.view(-1).float()
        # action=1 -> +1
        # action=0 -> -1
        action_sign = 2.0 * action - 1.0

        reward = action_sign * bounded_margin
        return reward

class PPOActorCritic(nn.Module):


    def __init__(self, state_dim: int = 6, action_dim: int = 2):
        super(PPOActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim


        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim)
        )


        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.actor(state.float())
        return F.softmax(logits, dim=-1)

    def select_action(
            self,
            state: Union[torch.Tensor, np.ndarray],
            device: torch.device
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:

        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(device)

        probs = self.get_action_probs(state)
        dist = Categorical(probs)
        action = dist.sample()

        value = self.critic(state.float()).squeeze(-1)
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value

    def evaluate_actions(
            self,
            state: torch.Tensor,
            action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        probs = self.get_action_probs(state)
        dist = Categorical(probs)

        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        state_values = self.critic(state.float()).squeeze(-1)

        return log_probs, state_values, entropy


def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: float,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:

    advantages = []
    gae = 0.0

    vals = values.tolist()
    vals_ext = vals + [next_value]

    for step in reversed(range(len(rewards))):
        done = float(dones[step])
        delta = rewards[step] + gamma * vals_ext[step + 1] * (1.0 - done) - vals_ext[step]
        gae = delta + gamma * lam * (1.0 - done) * gae
        advantages.insert(0, gae)

    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    returns_tensor = advantages_tensor + values.cpu()
    return advantages_tensor, returns_tensor