import torch
import torch.nn as nn
from stable_baselines3.ppo.policies import MultiInputPolicy
import numpy as np


class MaskedMultiBinaryPolicy(MultiInputPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(MaskedMultiBinaryPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def compute_action_mask(self, obs, device):
        """
        Compute the action mask based on obs.
        obs is a dict of tensors with shapes (n_envs, ...)
        We'll assume:
          - obs['ms_trades'].shape = (n_envs, max_ms_trades, 5)
          - obs['exchange_trades'].shape = (n_envs, max_exchange_trades, 5)
          - processed_flag=1 means unprocessed (selectable), 0 means processed (not selectable).

        Action structure: [action_type(1 bit), ms_trades bits, exchange_trades bits]
        Let's say max_ms_trades=3 and max_exchange_trades=3 for example.
        """
        n_envs = obs['positions'].shape[0]
        action_dim = self.action_space.n  # total action bits

        mask = torch.ones((n_envs, action_dim), dtype=torch.bool, device=device)

        # Indices:
        # action_type: index 0
        # ms trades: index 1 to 1+max_ms_trades-1
        # exchange trades: following the ms trades
        max_ms_trades = obs['ms_trades'].shape[1]
        max_exchange_trades = obs['exchange_trades'].shape[1]
        ms_start = 1
        ms_end = ms_start + max_ms_trades
        ex_start = ms_end
        ex_end = ex_start + max_exchange_trades

        # Extract processed_flag (assuming it's at index 2 of each trade)
        # obs['ms_trades'][:,:,2] -> (n_envs, max_ms_trades)
        ms_unprocessed = (obs['ms_trades'][:, :, 2] == 1)
        ex_unprocessed = (obs['exchange_trades'][:, :, 2] == 1)

        # If processed_flag=1 means unprocessed (good), we allow these actions:
        # mask out processed (where ms_unprocessed=False)
        mask[:, ms_start:ms_end] = ms_unprocessed
        mask[:, ex_start:ex_end] = ex_unprocessed

        return mask

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        values = self.value_net(latent_vf)
        device = logits.device

        # Compute mask
        mask = self.compute_action_mask(obs, device)
        masked_logits = logits.clone()
        # Large negative value to invalid actions
        masked_logits[~mask] = -1e9

        # Create the distribution with masked logits
        with torch.no_grad():
            # Convert logits to probabilities with sigmoid
            probabilities = torch.sigmoid(masked_logits)
            # Set the distribution from these probabilities
            self.action_dist.proba_distribution(probabilities)

        if deterministic:
            actions = self.action_dist.mode()
        else:
            actions = self.action_dist.sample()
            # Ensure at least one trade is selected:
            # Re-sample if needed
            n_envs = actions.shape[0]
            attempts = 0
            while attempts < 10:
                # sum over trade bits except the first bit
                trades_selected = actions[:, 1:].sum(dim=1)
                invalid = (trades_selected == 0)
                if not invalid.any():
                    break
                # re-sample only for invalid envs
                new_actions = self.action_dist.sample()
                actions[invalid] = new_actions[invalid]
                attempts += 1

        return actions, values, self.action_dist.entropy()
