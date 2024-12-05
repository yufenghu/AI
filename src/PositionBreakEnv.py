import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PositionBreakEnv(gym.Env):
    def __init__(self, ms_position=0, exchange_position=0, ms_trades=None, exchange_trades=None):
        """
        Initialize the environment with dynamic positions and trades.

        :param ms_position: Initial MS position (float)
        :param exchange_position: Initial Exchange position (float)
        :param ms_trades: MS trades as a 2D numpy array with shape (n_trades, 2), where each row is [qty, price]
        :param exchange_trades: Exchange trades as a 2D numpy array with shape (n_trades, 2), where each row is [qty, price]
        """
        super(PositionBreakEnv, self).__init__()

        # Parameters for initial positions
        self.ms_position = ms_position
        self.exchange_position = exchange_position

        # Set up the trades (if not provided, use random trades)
        self.ms_trades = ms_trades if ms_trades is not None else np.random.randint(1, 10, size=(10, 2)).astype(
            np.float32)
        self.exchange_trades = exchange_trades if exchange_trades is not None else np.random.randint(1, 10, size=(
        10, 2)).astype(np.float32)

        # Number of trades in MS and Exchange
        self.num_ms_trades = self.ms_trades.shape[0]
        self.num_exchange_trades = self.exchange_trades.shape[0]

        # Action space: Allow for actions to match or balance trades
        self.action_space = spaces.Discrete(
            2 ** (self.num_ms_trades + self.num_exchange_trades))  # Binary encoding for action space

        # Observation space: Flattened trades and positions
        self.observation_space = spaces.Dict({
            "trades": spaces.Box(low=-9999, high=9999, shape=(self.num_ms_trades * 2 + self.num_exchange_trades * 2,),
                                 dtype=np.float32),
            "positions": spaces.Box(low=-9999, high=9999, shape=(2,), dtype=np.float32),  # MS and Exchange positions
        })

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation.
        """
        # Use the current state for reset (or can add randomness if needed)
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Return the current observation consisting of flattened trades and positions.
        """
        flattened_trades = np.concatenate((self.ms_trades.flatten(), self.exchange_trades.flatten()))
        positions = np.array([self.ms_position, self.exchange_position], dtype=np.float32)
        return {"trades": flattened_trades, "positions": positions}

    def step(self, action):
        """
        Perform the step based on the given action and return the new observation, reward, and termination status.
        """
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

        # Check if both positions are balanced and all trades are processed
        if not np.any(
                self.unprocessed_ms_trades) and not np.any(self.unprocessed_exchange_trades):
            terminated = True  # Terminate if positions are balanced and trades are processed

        return self._get_obs(), reward, terminated, truncated, {}

    def _match(self, ms_indices, exchange_indices):
        """
        Match MS and Exchange trades and check if total qty matches.
        """
        ms_trades = self.ms_trades[ms_indices]
        exchange_trades = self.exchange_trades[exchange_indices]

        ms_total_qty = ms_trades[:, 0].sum()
        exchange_total_qty = exchange_trades[:, 0].sum()

        if ms_total_qty != exchange_total_qty:
            return -10  # Quantity mismatch penalty

        if ms_total_qty == 0 or exchange_total_qty == 0:
            return -10
        # Calculate average price for both MS and Exchange trades
        ms_avg_price = np.sum(ms_trades[:, 0] * ms_trades[:, 1]) / ms_total_qty
        exchange_avg_price = np.sum(exchange_trades[:, 0] * exchange_trades[:, 1]) / exchange_total_qty

        price_diff = abs(ms_avg_price - exchange_avg_price)
        tolerance = 5  # Allowed price difference tolerance

        if price_diff > tolerance:
            return -5 * price_diff  # Penalty for excessive price difference

        # Successful match: zero out the selected trades (no position adjustment)
        self.ms_trades[ms_indices] = 0
        self.exchange_trades[exchange_indices] = 0
        return 100 - price_diff  # Reward for successful match

    def _balance(self, ms_indices, exchange_indices):
        """
        Balance MS and Exchange positions by adjusting the positions using selected trades.
        """
        position_diff = abs(self.ms_position - self.exchange_position)
        ms_trades = self.ms_trades[ms_indices]
        exchange_trades = self.exchange_trades[exchange_indices]

        ms_total_qty = ms_trades[:, 0].sum()
        exchange_total_qty = exchange_trades[:, 0].sum()

        total_qty = ms_total_qty + exchange_total_qty

        if total_qty != position_diff:
            return -10  # Penalty if the total quantity doesn't match the position difference

        # Successful balance: zero out the trades and adjust positions
        self.ms_trades[ms_indices] = 0
        self.exchange_trades[exchange_indices] = 0
        self.ms_position = 0
        self.exchange_position = 0

        return 50  # Reward for balancing positions

    def _encode_action(self, action_type, ms_indices, exchange_indices):
        """
        Encode the action into a single integer.
        """
        ms_binary = int("".join(map(str, ms_indices.astype(int))), 2)
        exchange_binary = int("".join(map(str, exchange_indices.astype(int))), 2)

        return (action_type * (2 ** (self.num_ms_trades + self.num_exchange_trades))) + \
            (ms_binary * (2 ** self.num_exchange_trades)) + exchange_binary

    def _decode_action(self, action):
        """
        Decode the action back to its components dynamically based on the number of MS and Exchange trades.
        """
        # Total number of bits required to represent the action for both MS and Exchange trades
        total_bits = self.num_ms_trades + self.num_exchange_trades

        # Calculate the action type (the part of the action corresponding to the action type)
        action_type = action // (2 ** total_bits)

        # Get the remainder to work with the trade binary representations
        remainder = action % (2 ** total_bits)

        # Extract MS trade binary (we need to extract the first `num_ms_trades` bits)
        ms_binary = remainder // (2 ** self.num_exchange_trades)

        # Extract Exchange trade binary (we need to extract the last `num_exchange_trades` bits)
        exchange_binary = remainder % (2 ** self.num_exchange_trades)

        # Convert MS and Exchange binary representations to boolean arrays
        ms_indices = np.array(list(map(int, f"{ms_binary:0{self.num_ms_trades}b}")), dtype=bool)
        exchange_indices = np.array(list(map(int, f"{exchange_binary:0{self.num_exchange_trades}b}")), dtype=bool)

        # Return the decoded components: action type, MS trade selections, and Exchange trade selections
        return action_type, ms_indices, exchange_indices

