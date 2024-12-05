import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PositionBreakEnv(gym.Env):
    def __init__(self):
        super(PositionBreakEnv, self).__init__()

        # Number of trades for MS and Exchange
        self.num_ms_trades = 3
        self.num_exchange_trades = 3

        # Total possible actions: 2 * (2^num_ms_trades) * (2^num_exchange_trades)
        self.num_possible_actions = 2 * (2 ** self.num_ms_trades) * (2 ** self.num_exchange_trades)
        self.action_space = spaces.Discrete(self.num_possible_actions)

        # Observation space
        self.observation_space = spaces.Dict({
            "trades": spaces.Box(low=-9999, high=9999, shape=(12,), dtype=np.float32),
            # Flattened ms_trades + exchange_trades
            "positions": spaces.Box(low=-9999, high=9999, shape=(2,), dtype=np.float32),  # ms_position, exchange_position
        })

        self.reset()

    def reset(self, seed=None, options=None):
        # Initialize trades and positions
        self.ms_trades = np.array([
            [5, 100], [5, 102], [5, 101]
        ], dtype=np.float32)

        self.exchange_trades = np.array([
            [5, 98], [10, 99], [5, 101]
        ], dtype=np.float32)

        self.ms_position = 15.0
        self.exchange_position = 20.0

        self.unprocessed_ms_trades = self.ms_trades[:,
                                     0] > 0  # Boolean array tracking MS trades with non-zero quantities
        self.unprocessed_exchange_trades = self.exchange_trades[:,
                                           0] > 0  # Boolean array tracking Exchange trades with non-zero quantities

        return self._get_obs(), {}

    def _get_obs(self):
        # Flatten trades and combine with positions
        flattened_trades = np.concatenate((self.ms_trades.flatten(), self.exchange_trades.flatten()))
        positions = np.array([self.ms_position, self.exchange_position], dtype=np.float32)
        return {"trades": flattened_trades, "positions": positions}

    def step(self, action):
        # Decode the flattened action
        action_type, ms_indices, exchange_indices = self._decode_action(action)

        reward = 0
        terminated = False
        truncated = False  # No time limit, so truncated is always False

        if action_type == 0:  # Match operation
            reward = self._match(ms_indices, exchange_indices)
        elif action_type == 1:  # Balance operation
            reward = self._balance(ms_indices, exchange_indices)

        # Update unprocessed trades after the action
        self.unprocessed_ms_trades = self.ms_trades[:, 0] > 0
        self.unprocessed_exchange_trades = self.exchange_trades[:, 0] > 0

        # Check if the positions are balanced and no unprocessed trades are left
        if not np.any(
                self.unprocessed_ms_trades) and not np.any(self.unprocessed_exchange_trades):
            terminated = True  # Terminate if both positions are balanced and all trades have been processed

        return self._get_obs(), reward, terminated, truncated, {}

    def _match(self, ms_indices, exchange_indices):
        # Retrieve selected trades
        ms_trades = self.ms_trades[ms_indices]
        exchange_trades = self.exchange_trades[exchange_indices]

        ms_total_qty = ms_trades[:, 0].sum()
        exchange_total_qty = exchange_trades[:, 0].sum()

        if ms_total_qty != exchange_total_qty:
            return -10  # Quantity mismatch

        # Calculate average price
        ms_avg_price = np.sum(ms_trades[:, 0] * ms_trades[:, 1]) / ms_total_qty
        exchange_avg_price = np.sum(exchange_trades[:, 0] * exchange_trades[:, 1]) / exchange_total_qty

        price_diff = abs(ms_avg_price - exchange_avg_price)
        tolerance = 5  # Allowed price difference

        if price_diff > tolerance:
            return -5 * price_diff  # Excessive price difference penalty

        # Successful match: mark the trades as matched (no position adjustment)
        self.ms_trades[ms_indices] = 0
        self.exchange_trades[exchange_indices] = 0
        # Do not adjust positions here, as per the requirement
        return 100 - price_diff  # Reward for successful match

    def _balance(self, ms_indices, exchange_indices):
        """
        Balance the positions using selected MS and Exchange trades.
        The selected MS and Exchange trades must together bridge the gap between MS and Exchange positions.
        """
        # Calculate the position difference
        position_diff = abs(self.ms_position - self.exchange_position)

        # Select MS and Exchange trades based on the indices
        ms_trades = self.ms_trades[ms_indices]
        exchange_trades = self.exchange_trades[exchange_indices]

        ms_total_qty = ms_trades[:, 0].sum()
        exchange_total_qty = exchange_trades[:, 0].sum()

        # Total quantity of MS and Exchange trades should match the position difference
        total_qty = ms_total_qty + exchange_total_qty

        if total_qty != position_diff:
            return -10  # Penalty if the total quantity doesn't match the position difference

        # Successful balance: remove selected trades and adjust positions
        self.ms_trades[ms_indices] = 0  # Set MS trades to zero
        self.exchange_trades[exchange_indices] = 0  # Set selected Exchange trades to zero
        self.ms_position = 0  # MS position is now balanced
        self.exchange_position = 0  # Exchange position is now balanced

        return 50  # Reward for successfully balancing the positions

    def _encode_action(self, action_type, ms_indices, exchange_indices):
        """
        Encode the action into a single integer.
        """
        # Convert binary arrays to integers
        ms_binary = int("".join(map(str, ms_indices.astype(int))), 2)
        exchange_binary = int("".join(map(str, exchange_indices.astype(int))), 2)

        # Encode into a single integer
        return (action_type * (2 ** (self.num_ms_trades + self.num_exchange_trades))) + \
            (ms_binary * (2 ** self.num_exchange_trades)) + exchange_binary

    def _decode_action(self, action):
        """
        Decode the encoded action.
        """
        # Decode action_type
        action_type = action // (2 ** (self.num_ms_trades + self.num_exchange_trades))

        # Decode ms_indices binary
        remainder = action % (2 ** (self.num_ms_trades + self.num_exchange_trades))
        ms_binary = remainder // (2 ** self.num_exchange_trades)

        # Decode exchange_indices binary
        exchange_binary = remainder % (2 ** self.num_exchange_trades)

        # Convert integers back to binary array
        ms_indices = np.array(list(map(int, f"{ms_binary:03b}")), dtype=bool)
        exchange_indices = np.array(list(map(int, f"{exchange_binary:03b}")), dtype=bool)

        return action_type, ms_indices, exchange_indices
