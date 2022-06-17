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
           
            nn.Linear(hidden_units[1], hidden_units[2]),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(hidden_units[2], action_size)
       
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        h = self.seq(state)
        out = self.out(h)
       
        return out