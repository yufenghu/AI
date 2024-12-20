import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os

class PositionBreakEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ms_position=10.0, exchange_position=15.0,
                 ms_trades=None, exchange_trades=None,
                 max_ms_trades=10, max_exchange_trades=20,
                 trade_label_dict_path="trade_label_dict.json",
                 processed_flag_index=-1):
        """
        ms_trades and exchange_trades are lists of dict, each:
        {
          "qty": float,
          "price": float,
          "matchgroup": int,
          "balanceID": int,
          "unique_id": str,
          "buy_sell": "B" or "S",
          "side": "MS" or "Exchange",
          "trade_label": string label
        }

        We'll produce final trade array with fields:
        [qty(0), price(1), group_index(2), buy_sell_id(3), side_id(4), trade_label_id(5), matchgroup(6), balanceID(7), processed_flag(8)]
        Observation excludes matchgroup and balanceID (indices 6,7).
        processed_flag_index can be -1 meaning last column is processed_flag.
        """
        super().__init__()

        self.initial_ms_position = float(ms_position)
        self.initial_exchange_position = float(exchange_position)

        if ms_trades is None or exchange_trades is None:
            raise ValueError("ms_trades and exchange_trades must be lists of dicts.")

        self.ms_trades_json = ms_trades
        self.exchange_trades_json = exchange_trades

        self.max_ms_trades = max_ms_trades
        self.max_exchange_trades = max_exchange_trades
        self.trade_label_dict_path = trade_label_dict_path
        self.trade_label_dict = self._load_or_create_trade_label_dict()
        self.processed_flag_index = processed_flag_index

        # Action space:
        # action_type: 0=match,1=balance,2=forceMatch
        # each trade bit: {0,1}
        action_dim = 1 + self.max_ms_trades + self.max_exchange_trades
        nvec = [3] + [2]*(self.max_ms_trades + self.max_exchange_trades)
        self.action_space = spaces.MultiDiscrete(nvec)

        self.action_log = []
        self.reset()

    def _load_or_create_trade_label_dict(self):
        if os.path.exists(self.trade_label_dict_path):
            with open(self.trade_label_dict_path, 'r') as f:
                return json.load(f)
        else:
            return {}

    def _save_trade_label_dict(self):
        with open(self.trade_label_dict_path, 'w') as f:
            json.dump(self.trade_label_dict, f)

    def _get_trade_label_id(self, label):
        if label not in self.trade_label_dict:
            if len(self.trade_label_dict) == 0:
                new_id = 0
            else:
                new_id = max(self.trade_label_dict.values()) + 1
            self.trade_label_dict[label] = new_id
            self._save_trade_label_dict()
        return self.trade_label_dict[label]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ms_position = self.initial_ms_position
        self.exchange_position = self.initial_exchange_position
        self.position_diff = self.ms_position - self.exchange_position

        # Extract unique_ids for group_index
        ms_unique_ids = [t["unique_id"] for t in self.ms_trades_json]
        ex_unique_ids = [t["unique_id"] for t in self.exchange_trades_json]

        all_ids = ms_unique_ids + ex_unique_ids
        unique_ids = list(set(all_ids))
        group_index_dict = {uid:i for i,uid in enumerate(unique_ids)}

        buy_sell_dict = {"B":0, "S":1}
        side_dict = {"MS":0, "Exchange":1}

        def process_trade(t):
            processed_flag = 1
            qty = float(t["qty"])
            price = float(t["price"])
            group_index = float(group_index_dict[t["unique_id"]])
            buy_sell_id = float(buy_sell_dict[t["buy_sell"]])
            side_id = float(side_dict[t["side"]])
            trade_label_id = float(self._get_trade_label_id(t["trade_label"]))

            # final order of fields (9 total):
            # qty(0), price(1), group_index(2), buy_sell_id(3), side_id(4), trade_label_id(5), matchgroup(6), balanceID(7), processed_flag(8)
            row = [qty, price, group_index, buy_sell_id, side_id, trade_label_id, processed_flag]
            return row

        ms_trades_array = np.array([process_trade(tr) for tr in self.ms_trades_json], dtype=np.float32)
        ex_trades_array = np.array([process_trade(tr) for tr in self.exchange_trades_json], dtype=np.float32)

        self.ms_trades_actual = ms_trades_array
        self.exchange_trades_actual = ex_trades_array

        self.num_ms_trades = self.ms_trades_actual.shape[0]
        self.num_exchange_trades = self.exchange_trades_actual.shape[0]

        self.num_columns = self.ms_trades_actual.shape[1] if self.num_ms_trades>0 else self.exchange_trades_actual.shape[1]

        # If processed_flag_index = -1 means last column
        self.processed_flag_index = self.num_columns - 1 - 2


        self.obs_indices = list(range(self.processed_flag_index + 1))

        obs_num_columns = len(self.obs_indices)
        self.observation_space = spaces.Dict({
            "ms_trades": spaces.Box(low=-9999, high=9999,
                                    shape=(self.max_ms_trades, obs_num_columns), dtype=np.float32),
            "exchange_trades": spaces.Box(low=-9999, high=9999,
                                          shape=(self.max_exchange_trades, obs_num_columns), dtype=np.float32),
            "positions": spaces.Box(low=-9999, high=9999, shape=(2,), dtype=np.float32),
            "position_diff": spaces.Box(low=-9999, high=9999, shape=(1,), dtype=np.float32)
        })

        self.action_log = []
        return self._get_obs(), {}

    def _pad_trades(self, trades, max_count):
        obs_data = trades[:, self.obs_indices]
        padded = np.zeros((max_count,obs_data.shape[1]), dtype=np.float32)
        count = min(obs_data.shape[0], max_count)
        padded[:count,:] = obs_data[:count,:]
        return padded

    def _get_obs(self):
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
        trade_bits = action[1:]
        ms_indices = trade_bits[:self.max_ms_trades].astype(bool)
        exchange_indices = trade_bits[self.max_ms_trades:].astype(bool)

        ms_indices = ms_indices[:self.num_ms_trades]
        exchange_indices = exchange_indices[:self.num_exchange_trades]

        processed_flag_index = self.processed_flag_index
        matchgroup_index = self.processed_flag_index + 1
        balanceID_index = self.processed_flag_index + 2

        ms_before = self.ms_trades_actual[ms_indices,processed_flag_index].copy() if ms_indices.any() else np.array([])
        ex_before = self.exchange_trades_actual[exchange_indices,processed_flag_index].copy() if exchange_indices.any() else np.array([])

        if action_type == 0:
            reward = self._match(ms_indices, exchange_indices, matchgroup_index, balanceID_index)
            action_str = "MATCH"
        elif action_type == 1:
            reward = self._balance(ms_indices, exchange_indices, matchgroup_index, balanceID_index)
            action_str = "BALANCE"
        else:
            reward = self._forcematch(ms_indices, exchange_indices, processed_flag_index)
            action_str = "FORCEMATCH"

        unprocessed_ms = np.any((self.ms_trades_actual[:,processed_flag_index]==1))
        unprocessed_exchange = np.any((self.exchange_trades_actual[:,processed_flag_index]==1))

        terminated = False
        truncated = False
        if not unprocessed_ms and not unprocessed_exchange:
            terminated = True

        ms_after = self.ms_trades_actual[ms_indices,processed_flag_index] if ms_indices.any() else np.array([])
        ex_after = self.exchange_trades_actual[exchange_indices,processed_flag_index] if exchange_indices.any() else np.array([])

        newly_processed = (ms_after < ms_before).any() if ms_before.size>0 else False
        newly_processed = newly_processed or ((ex_after < ex_before).any() if ex_before.size>0 else False)

        if newly_processed:
            initial_pos_diff = self.ms_position - self.exchange_position
            ms_selected_trades = self.ms_trades_actual[ms_indices].copy() if ms_indices.any() else np.empty((0,self.num_columns), dtype=np.float32)
            exchange_selected_trades = self.exchange_trades_actual[exchange_indices].copy() if exchange_indices.any() else np.empty((0,self.num_columns), dtype=np.float32)
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

    def _match(self, ms_indices, exchange_indices, matchgroup_index):
        if np.all(ms_indices == False) and np.all(exchange_indices == False):
            return 0

        processed_flag_index = self.processed_flag_index

        # If any selected trade is already processed (0), return no reward
        if np.any(self.ms_trades_actual[ms_indices, processed_flag_index] == 0) or \
           np.any(self.exchange_trades_actual[exchange_indices, processed_flag_index] == 0):
            return 0

        ms_sel = self.ms_trades_actual[ms_indices]
        ex_sel = self.exchange_trades_actual[exchange_indices]

        ms_qty = ms_sel[:, 0].sum() if ms_sel.size > 0 else 0
        ex_qty = ex_sel[:, 0].sum() if ex_sel.size > 0 else 0

        # If quantities differ, small penalty
        if ms_qty != ex_qty:
            self._mark_processed(ms_sel, ms_indices, ex_sel, exchange_indices, processed_flag_index)
            return -0.5

        # Quantities match, mark processed
        self._mark_processed(ms_sel, ms_indices, ex_sel, exchange_indices, processed_flag_index)
        reward = 10 * (ms_sel.size + ex_sel.size)

        # Check exact matchgroup requirement
        ms_mgroups = self.ms_trades_json[ms_indices, matchgroup_index].astype(int)
        ex_mgroups = self.exchange_trades_json[exchange_indices, matchgroup_index].astype(int)

        if (len(np.unique(ms_mgroups)) == 1 and
            len(np.unique(ex_mgroups)) == 1 and
            ms_mgroups[0] == ex_mgroups[0] and ms_mgroups[0] != 0):

            mg = ms_mgroups[0]
            all_ms_mg = np.where(self.ms_trades_json[:, matchgroup_index].astype(int) == mg)[0]
            all_ex_mg = np.where(self.exchange_trades_json[:, matchgroup_index].astype(int) == mg)[0]

            selected_ms_set = set(np.where(ms_indices)[0])
            selected_ex_set = set(np.where(exchange_indices)[0])
            all_ms_set = set(all_ms_mg)
            all_ex_set = set(all_ex_mg)

            # If exactly all trades of this matchgroup are selected
            if selected_ms_set == all_ms_set and selected_ex_set == all_ex_set:
                reward += 10 * (ms_sel.size + ex_sel.size)

        return reward

    def _balance(self, ms_indices, exchange_indices, balanceID_index):
        if np.all(ms_indices == False) and np.all(exchange_indices == False):
            return 0

        processed_flag_index = self.processed_flag_index

        # If any selected trade is already processed, no reward
        if np.any(self.ms_trades_actual[ms_indices, processed_flag_index] == 0) or \
           np.any(self.exchange_trades_actual[exchange_indices, processed_flag_index] == 0):
            return 0

        ms_sel = self.ms_trades_actual[ms_indices]
        ex_sel = self.exchange_trades_actual[exchange_indices]

        ms_qty = ms_sel[:, 0].sum() if ms_sel.size > 0 else 0
        ex_qty = ex_sel[:, 0].sum() if ex_sel.size > 0 else 0
        total_qty_diff = ms_qty - ex_qty
        self.position_diff = self.position_diff - total_qty_diff

        if total_qty_diff == 0:
            self._mark_processed(ms_sel, ms_indices, ex_sel, exchange_indices, processed_flag_index)
            return -100 * (ms_sel.size + ex_sel.size)

        if self.position_diff != 0:
            self._mark_processed(ms_sel, ms_indices, ex_sel, exchange_indices, processed_flag_index)
            return -0.5

        # Successfully balanced
        self._mark_processed(ms_sel, ms_indices, ex_sel, exchange_indices, processed_flag_index)
        reward = 11 * (ms_sel.size + ex_sel.size)

        # Check exact balanceID requirement
        ms_bgroups = self.ms_trades_json[ms_indices, balanceID_index].astype(int)
        ex_bgroups = self.exchange_trades_json[exchange_indices, balanceID_index].astype(int)

        if (len(np.unique(ms_bgroups)) == 1 and
            len(np.unique(ex_bgroups)) == 1 and
            ms_bgroups[0] == ex_bgroups[0] and ms_bgroups[0] != 0):

            bg = ms_bgroups[0]
            all_ms_bg = np.where(self.ms_trades_json[:, balanceID_index].astype(int) == bg)[0]
            all_ex_bg = np.where(self.exchange_trades_json[:, balanceID_index].astype(int) == bg)[0]

            selected_ms_set = set(np.where(ms_indices)[0])
            selected_ex_set = set(np.where(exchange_indices)[0])
            all_ms_set = set(all_ms_bg)
            all_ex_set = set(all_ex_bg)

            if selected_ms_set == all_ms_set and selected_ex_set == all_ex_set:
                reward += 10 * (ms_sel.size + ex_sel.size)

        return reward

    def _forcematch(self, ms_indices, exchange_indices, processed_flag_index):
        if np.all(ms_indices == False) and np.all(exchange_indices == False):
            return 0

        processed_flag_index = self.processed_flag_index

        # If any selected trade is already processed, no reward
        if np.any(self.ms_trades_actual[ms_indices, processed_flag_index] == 0) or \
                np.any(self.exchange_trades_actual[exchange_indices, processed_flag_index] == 0):
            return 0

        ms_sel = self.ms_trades_actual[ms_indices]
        ex_sel = self.exchange_trades_actual[exchange_indices]

        if self.position_diff != 0:
            self._mark_processed(ms_sel, ms_indices, ex_sel, exchange_indices, processed_flag_index)
            return -100 * (ms_sel.size + ex_sel.size)

        # Successfully forced matched
        self._mark_processed(ms_sel, ms_indices, ex_sel, exchange_indices, processed_flag_index)
        reward = 0.5

        return reward

    def _mark_processed(self, ms_sel, ms_indices, ex_sel, exchange_indices, processed_flag_index):
        # set processed_flag=0
        if ms_sel.size > 0:
            self.ms_trades_actual[np.where(ms_indices)[0], processed_flag_index] = 0
        if ex_sel.size > 0:
            self.exchange_trades_actual[np.where(exchange_indices)[0], processed_flag_index] = 0

    def render(self):
        print("MS Position:", self.ms_position, "Exchange Position:", self.exchange_position)
        print("MS Trades Actual:")
        print(self.ms_trades_actual)
        print("Exchange Trades Actual:")
        print(self.exchange_trades_actual)

    def get_action_summary(self):
        return self.action_log
