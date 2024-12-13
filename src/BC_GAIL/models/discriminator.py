import torch
import torch.nn as nn


class GAILDiscriminator(nn.Module):
    def __init__(self, obs_spaces, action_dim, device=None):
        super(GAILDiscriminator, self).__init__()
        # Store device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Example: Suppose obs_spaces is a dictionary of spaces like:
        # {
        #   "ms_trades": Box(..., shape=(max_ms_trades,5)),
        #   "exchange_trades": Box(..., shape=(max_exchange_trades,5)),
        #   "positions": Box(..., shape=(2,)),
        #   "position_diff": Box(..., shape=(1,))
        # }

        # Flatten and combine features from each observation component.
        # Adjust as needed based on your input shapes.

        ms_trades_dim = obs_spaces['ms_trades'].shape[0] * obs_spaces['ms_trades'].shape[1]  # max_ms_trades * 5
        exchange_trades_dim = obs_spaces['exchange_trades'].shape[0] * obs_spaces['exchange_trades'].shape[
            1]  # max_exchange_trades * 5
        positions_dim = obs_spaces['positions'].shape[0]  # 2
        position_diff_dim = obs_spaces['position_diff'].shape[0]  # 1

        # A simple MLP to process each component
        self.ms_trades_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ms_trades_dim, 128),
            nn.ReLU()
        )
        self.exchange_trades_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(exchange_trades_dim, 128),
            nn.ReLU()
        )
        self.positions_net = nn.Sequential(
            nn.Linear(positions_dim, 64),
            nn.ReLU()
        )
        self.position_diff_net = nn.Sequential(
            nn.Linear(position_diff_dim, 32),
            nn.ReLU()
        )

        # Combine all obs features + action into a single network
        # action_dim is known from your environment (binary vector)
        combined_input_dim = 128 + 128 + 64 + 32 + action_dim

        self.combined_net = nn.Sequential(
            nn.Linear(combined_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, obs, action):
        # obs is a dict: {"ms_trades": tensor, "exchange_trades": tensor, "positions": tensor, "position_diff": tensor}
        # action is a tensor of shape (batch_size, action_dim)

        # Ensure obs and actions are on self.device
        # If they are not, you can do something like:
        # obs['ms_trades'] = obs['ms_trades'].to(self.device)
        # ... and similarly for each component and action.
        # Or ensure they are already moved to device outside this function.

        ms_trades_out = self.ms_trades_net(obs['ms_trades'])
        exchange_trades_out = self.exchange_trades_net(obs['exchange_trades'])
        positions_out = self.positions_net(obs['positions'])
        position_diff_out = self.position_diff_net(obs['position_diff'])

        # Concatenate all features and actions
        combined_input = torch.cat([ms_trades_out, exchange_trades_out, positions_out, position_diff_out, action],
                                   dim=1)

        return self.combined_net(combined_input)

# Example usage:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# discriminator = GAILDiscriminator(obs_spaces, action_dim, device=device)
# discriminator.to(device)
