import numpy as np
import json
import os















def generate_synthetic_expert_data(num_samples, max_ms_trades, max_exchange_trades, action_dim):
    """
    Generate synthetic expert data for demonstration purposes.

    Parameters:
    - num_samples (int): Number of expert samples to generate.
    - max_ms_trades (int): Maximum number of MS trades.
    - max_exchange_trades (int): Maximum number of Exchange trades.
    - action_dim (int): Dimension of the action space.

    Returns:
    - expert_obs (dict): Dictionary containing observation components.
    - expert_actions (np.ndarray): Array of expert actions.
    """
    # Generate synthetic ms_trades and exchange_trades
    ms_trades = np.random.randn(num_samples, max_ms_trades, 5).astype(np.float32)
    exchange_trades = np.random.randn(num_samples, max_exchange_trades, 5).astype(np.float32)

    # Initialize processed_flag to 1 (unprocessed)
    ms_trades[:, :, 2] = 1.0
    exchange_trades[:, :, 2] = 1.0

    # Generate positions and position_diff
    ms_positions = np.random.uniform(5.0, 15.0, size=(num_samples, 1)).astype(np.float32)
    exchange_positions = np.random.uniform(10.0, 20.0, size=(num_samples, 1)).astype(np.float32)
    position_diff = ms_positions - exchange_positions

    # Generate synthetic actions
    action_type = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
    ms_trade_selections = np.random.randint(0, 2, size=(num_samples, max_ms_trades)).astype(np.float32)
    exchange_trade_selections = np.random.randint(0, 2, size=(num_samples, max_exchange_trades)).astype(np.float32)
    actions = np.hstack((action_type, ms_trade_selections, exchange_trade_selections))

    # Package observations
    expert_obs = {
        'ms_trades': ms_trades,  # Shape: (num_samples, max_ms_trades, 5)
        'exchange_trades': exchange_trades,  # Shape: (num_samples, max_exchange_trades, 5)
        'positions': np.hstack((ms_positions, exchange_positions)),  # Shape: (num_samples, 2)
        'position_diff': position_diff  # Shape: (num_samples, 1)
    }

    return expert_obs, actions


def save_expert_data_to_json(filepath, expert_obs, expert_actions):
    """
    Save expert data to a JSON file.

    Parameters:
    - filepath (str): Path to save the expert data.
    - expert_obs (dict): Dictionary containing observation components.
    - expert_actions (np.ndarray): Array of expert actions.
    """
    # Convert NumPy arrays to lists
    serializable_obs = {
        'ms_trades': expert_obs['ms_trades'].tolist(),
        'exchange_trades': expert_obs['exchange_trades'].tolist(),
        'positions': expert_obs['positions'].tolist(),
        'position_diff': expert_obs['position_diff'].tolist()
    }
    serializable_actions = expert_actions.tolist()

    # Combine into a single dictionary
    data = {
        'expert_obs': serializable_obs,
        'expert_actions': serializable_actions
    }

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write to JSON file
    with open(filepath, 'w') as f:
        json.dump(data, f)

    print(f"Expert data saved to {filepath}")


def load_expert_data_from_json(filepath):
    """
    Load expert data from a JSON file.

    Parameters:
    - filepath (str): Path to the expert data JSON file.

    Returns:
    - expert_obs (dict): Dictionary containing observation components as NumPy arrays.
    - expert_actions (np.ndarray): Array of expert actions.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expert data file not found at {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Convert lists back to NumPy arrays
    expert_obs = {
        'ms_trades': np.array(data['expert_obs']['ms_trades'], dtype=np.float32),
        'exchange_trades': np.array(data['expert_obs']['exchange_trades'], dtype=np.float32),
        'positions': np.array(data['expert_obs']['positions'], dtype=np.float32),
        'position_diff': np.array(data['expert_obs']['position_diff'], dtype=np.float32)
    }
    expert_actions = np.array(data['expert_actions'], dtype=np.float32)

    return expert_obs, expert_actions
