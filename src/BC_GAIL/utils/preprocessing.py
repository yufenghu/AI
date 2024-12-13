# utils/preprocessing.py

import torch


def preprocess_obs(obs):
    """
    Convert observations (dict of arrays) into a dictionary of tensors.

    obs is expected to be a dict like:
    {
      'ms_trades': np.array(... shape (n_envs, max_ms_trades, 5) ...),
      'exchange_trades': np.array(... shape (n_envs, max_exchange_trades, 5) ...),
      'positions': np.array(... shape (n_envs, 2) ...),
      'position_diff': np.array(... shape (n_envs, 1) ...),
    }
    """
    processed = {
        'ms_trades': torch.tensor(obs['ms_trades'], dtype=torch.float32),
        'exchange_trades': torch.tensor(obs['exchange_trades'], dtype=torch.float32),
        'positions': torch.tensor(obs['positions'], dtype=torch.float32),
        'position_diff': torch.tensor(obs['position_diff'], dtype=torch.float32),
    }
    return processed
