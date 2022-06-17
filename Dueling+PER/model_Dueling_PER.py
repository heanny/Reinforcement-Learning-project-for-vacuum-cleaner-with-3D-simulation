import torch
import torch.nn as nn
import torch.nn.functional as F
 
class QNetwork(nn.Module):
    """Actor (Policy) Model."""
 
    def __init__(self, state_size, action_size, seed, hidden_units = [100, 50, 20]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.hidden_units = hidden_units
        self.seed = torch.manual_seed(seed)
        self.seq = nn.Sequential(
           
            nn.Linear(510, 265),
            nn.ReLU(inplace=True),
            
            nn.Linear(265, hidden_units[0]),
            nn.ReLU(inplace=True),
           
            nn.Linear(hidden_units[0], hidden_units[0]),
            nn.ReLU(inplace=True),
           
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(inplace=True),
        )
        self.adv_1 = nn.Sequential(nn.Linear(hidden_units[1], hidden_units[2]), nn.ReLU(inplace=True))
        self.val_1 = nn.Sequential(nn.Linear(hidden_units[1], hidden_units[2]), nn.ReLU(inplace=True))
        
        self.adv_2 = nn.Linear(hidden_units[2], action_size)
        self.val_2 = nn.Linear(hidden_units[2], out_features=1)
       
    def forward(self, state):
        """Build a network that maps state -> action values."""
        h = self.seq(state)
        adv_1 = self.adv_1(h)
        val_1 = self.val_1(h)
        
        adv = self.adv_2(adv_1)
        val = self.val_2(val_1)

        if state.size(0) > 1:
            out = val + adv - adv.mean(dim=0).expand(state.size(0), self.action_size)
        else:
            out = val + adv - adv.mean(dim=1).expand(state.size(0), self.action_size).unsqueeze(1)
        return out

