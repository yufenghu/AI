import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PositionBreakEnv(gym.Env):
    """
    Environment with:
    - multi-binary actions: first bit for action_type (0=match,1=balance),
      next bits for selecting MS and Exchange trades.
    - Trades have columns: [qty, price, processed_flag, matchgroup, balanceID].
      * processed_flag = 1 means unprocessed
      * processed_flag = 0 means processed

    Extra reward is given only if selected trades exactly match the full set of trades
    associated with the same matchgroup (for MATCH) or balanceID (for BALANCE).

    Parameters:
    - ms_position: initial MS position (float)
    - exchange_position: initial Exchange position (float)
    - ms_trades: numpy array of MS trades shape (n_ms_trades, 5)
    - exchange_trades: numpy array of Exchange trades shape (n_exchange_trades, 5)
    - max_ms_trades: maximum number of MS trades (for padding)
    - max_exchange_trades: maximum number of Exchange trades (for padding)
    """

    metadata = {"render_modes": []}

    def __init__(self, ms_position=10.0, exchange_position=15.0,
                 ms_trades=None, exchange_trades=None,
                 max_ms_trades=10, max_exchange_trades=20):
        super().__init__()

        self.initial_ms_position = float(ms_position)
        self.initial_exchange_position = float(exchange_position)
        self.position_diff = self.initial_ms_position - self.initial_exchange_position

        if ms_trades is None or exchange_trades is None:
            raise ValueError("ms_trades and exchange_trades must be provided")

        self.ms_trades_actual = ms_trades.astype(np.float32)
        self.exchange_trades_actual = exchange_trades.astype(np.float32)

        # Initialize processed_flag:
        # Set all trades to unprocessed (1 means unprocessed)
        self.ms_trades_actual[:, 2] = 1
        self.exchange_trades_actual[:, 2] = 1

        self.num_ms_trades = self.ms_trades_actual.shape[0]
        self.num_exchange_trades = self.exchange_trades_actual.shape[0]

        self.max_ms_trades = max_ms_trades
        self.max_exchange_trades = max_exchange_trades

        # Action space:
        # 1 bit for action_type + max_ms_trades bits + max_exchange_trades bits
        action_dim = 1 + self.max_ms_trades + self.max_exchange_trades
        self.action_space = spaces.MultiBinary(action_dim)

        # Observation space with fixed shapes for padding
        self.observation_space = spaces.Dict({
            "ms_trades": spaces.Box(low=-9999, high=9999,
                                    shape=(self.max_ms_trades, 5), dtype=np.float32),
            "exchange_trades": spaces.Box(low=-9999, high=9999,
                                          shape=(self.max_exchange_trades, 5), dtype=np.float32),
            "positions": spaces.Box(low=-9999, high=9999, shape=(2,), dtype=np.float32),
            "position_diff": spaces.Box(low=-9999, high=9999, shape=(1,), dtype=np.float32)
        })

        self.action_log = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ms_position = self.initial_ms_position
        self.exchange_position = self.initial_exchange_position
        self.position_diff = self.ms_position - self.exchange_position
        self.action_log = []
        return self._get_obs(), {}

    def _pad_trades(self, trades, max_count):
        """
        Pad or truncate trades to have shape (max_count, 5).
        """
        padded = np.zeros((max_count, 5), dtype=np.float32)
        count = min(trades.shape[0], max_count)
        padded[:count, :] = trades[:count, :]
        return padded

    def _get_obs(self):
        # Pad ms and exchange trades to fixed sizes
        padded_ms = self._pad_trades(self.ms_trades_actual, self.max_ms_trades)
        padded_ex = self._pad_trades(self.exchange_trades_actual, self.max_exchange_trades)

        return {
            "ms_trades": padded_ms,
            "exchange_trades": padded_ex,
            "positions": np.array([self.ms_position, self.exchange_position], dtype=np.float32),
            "position_diff": np.array([self.position_diff], dtype=np.float32)
        }

    def step(self, action):
        action_type = action[0]

        # Extract MS and Exchange indices
        ms_indices = action[1:1+self.max_ms_trades].astype(bool)
        exchange_indices = action[1+self.max_ms_trades:].astype(bool)

        # Truncate if fewer actual trades than max
        ms_indices = ms_indices[:self.num_ms_trades]
        exchange_indices = exchange_indices[:self.num_exchange_trades]

        ms_before = self.ms_trades_actual[ms_indices, 2].copy() if ms_indices.any() else np.array([])
        ex_before = self.exchange_trades_actual[exchange_indices, 2].copy() if exchange_indices.any() else np.array([])

        if action_type == 0:
            reward = self._match(ms_indices, exchange_indices)
            action_str = "MATCH"
        else:
            reward = self._balance(ms_indices, exchange_indices)
            action_str = "BALANCE"

        unprocessed_ms = np.any((self.ms_trades_actual[:, 2] == 1))
        unprocessed_exchange = np.any((self.exchange_trades_actual[:, 2] == 1))

        terminated = False
        truncated = False
        if not unprocessed_ms and not unprocessed_exchange:
            terminated = True

        ms_after = self.ms_trades_actual[ms_indices, 2] if ms_indices.any() else np.array([])
        ex_after = self.exchange_trades_actual[exchange_indices, 2] if exchange_indices.any() else np.array([])

        newly_processed = (ms_after < ms_before).any() if ms_before.size > 0 else False
        newly_processed = newly_processed or ((ex_after < ex_before).any() if ex_before.size > 0 else False)

        if newly_processed:
            initial_pos_diff = self.ms_position - self.exchange_position
            ms_selected_trades = self.ms_trades_actual[ms_indices].copy() if ms_indices.any() else np.empty((0,5), dtype=np.float32)
            exchange_selected_trades = self.exchange_trades_actual[exchange_indices].copy() if exchange_indices.any() else np.empty((0,5), dtype=np.float32)
            self.action_log.append({
                "step": len(self.action_log),
                "action_type": action_str,
                "ms_selected_indices": np.where(ms_indices)[0].tolist(),
                "exchange_selected_indices": np.where(exchange_indices)[0].tolist(),
                "ms_selected_trades": ms_selected_trades.tolist(),
                "exchange_selected_trades": exchange_selected_trades.tolist(),
                "reward": float(reward),
                "ms_position": float(self.ms_position),
                "exchange_position": float(self.exchange_position),
                "initial_pos_diff": float(initial_pos_diff),
                "position_diff": float(self.position_diff),
                "done": terminated
            })

        return self._get_obs(), reward, terminated, truncated, {}

    def _match(self, ms_indices, exchange_indices):
        if np.all(ms_indices == False) and np.all(exchange_indices == False):
            return -100.0

        # If any selected trade is already processed (0), return penalty
        if np.any(self.ms_trades_actual[ms_indices, 2] == 0) or np.any(self.exchange_trades_actual[exchange_indices, 2] == 0):
            return -100.0

        ms_sel = self.ms_trades_actual[ms_indices]
        ex_sel = self.exchange_trades_actual[exchange_indices]

        ms_qty = ms_sel[:, 0].sum() if ms_sel.size > 0 else 0
        ex_qty = ex_sel[:, 0].sum() if ex_sel.size > 0 else 0

        if ms_qty != ex_qty:
            self._mark_processed(ms_sel, ms_indices, ex_sel, exchange_indices)
            return -100.0

        # Mark processed trades as 0
        self._mark_processed(ms_sel, ms_indices, ex_sel, exchange_indices)
        reward = 50.0

        # Check exact matchgroup requirement, ignoring -1
        if ms_sel.size > 0 and ex_sel.size > 0:
            ms_mgroups = self.ms_trades_actual[ms_indices, 3].astype(int)
            ex_mgroups = self.exchange_trades_actual[exchange_indices, 3].astype(int)

            if (len(np.unique(ms_mgroups)) == 1 and
                len(np.unique(ex_mgroups)) == 1 and
                ms_mgroups[0] == ex_mgroups[0] and ms_mgroups[0] != 0):

                mg = ms_mgroups[0]
                all_ms_mg = np.where(self.ms_trades_actual[:, 3].astype(int) == mg)[0]
                all_ex_mg = np.where(self.exchange_trades_actual[:, 3].astype(int) == mg)[0]

                selected_ms_set = set(np.where(ms_indices)[0])
                selected_ex_set = set(np.where(exchange_indices)[0])
                all_ms_set = set(all_ms_mg)
                all_ex_set = set(all_ex_mg)

                if selected_ms_set == all_ms_set and selected_ex_set == all_ex_set:
                    reward += 40.0

        return reward





    def _balance(self, ms_indices, exchange_indices):
        if np.all(ms_indices == False) and np.all(exchange_indices == False):
            return -100.0

        # If any selected trade is already processed (0), return penalty
        if np.any(self.ms_trades_actual[ms_indices, 2] == 0) or np.any(self.exchange_trades_actual[exchange_indices, 2] == 0):
            return -100.0

        ms_sel = self.ms_trades_actual[ms_indices]
        ex_sel = self.exchange_trades_actual[exchange_indices]

        ms_qty = ms_sel[:, 0].sum() if ms_sel.size > 0 else 0
        ex_qty = ex_sel[:, 0].sum() if ex_sel.size > 0 else 0
        total_qty_diff = ms_qty - ex_qty
        self.position_diff = self.position_diff - total_qty_diff

        if self.position_diff != 0:
            self._mark_processed(ms_sel, ms_indices, ex_sel, exchange_indices)
            return -100.0

        # Mark processed trades as 0
        self._mark_processed(ms_sel, ms_indices, ex_sel, exchange_indices)
        self.ms_position = 0.0
        self.exchange_position = 0.0
        reward = 50.0

        # Check exact balanceID requirement, ignoring -1
        if ms_sel.size > 0 and ex_sel.size > 0:
            ms_bgroups = self.ms_trades_actual[ms_indices, 4].astype(int)
            ex_bgroups = self.exchange_trades_actual[exchange_indices, 4].astype(int)

            if (len(np.unique(ms_bgroups)) == 1 and
                len(np.unique(ex_bgroups)) == 1 and
                ms_bgroups[0] == ex_bgroups[0] and ms_bgroups[0] != 0):

                bg = ms_bgroups[0]
                all_ms_bg = np.where(self.ms_trades_actual[:, 4].astype(int) == bg)[0]
                all_ex_bg = np.where(self.exchange_trades_actual[:, 4].astype(int) == bg)[0]

                selected_ms_set = set(np.where(ms_indices)[0])
                selected_ex_set = set(np.where(exchange_indices)[0])
                all_ms_set = set(all_ms_bg)
                all_ex_set = set(all_ex_bg)

                if selected_ms_set == all_ms_set and selected_ex_set == all_ex_set:
                    reward += 80.0

        return reward


    def _mark_processed(self, ms_sel, ms_indices, ex_sel, exchange_indices):
        if ms_sel.size > 0:
            self.ms_trades_actual[np.where(ms_indices)[0], 2] = 0
        if ex_sel.size > 0:
            self.exchange_trades_actual[np.where(exchange_indices)[0], 2] = 0

    def render(self):
        print("MS Position:", self.ms_position, "Exchange Position:", self.exchange_position)
        print("MS Trades Actual:")
        print(self.ms_trades_actual)
        print("Exchange Trades Actual:")
        print(self.exchange_trades_actual)

    def get_action_summary(self):
        return self.action_log
