import torch
import torch.nn as nn
import torch.nn.functional as F
 
class QNetwork(nn.Module):
    """Actor (Policy) Model."""
 
    def __init__(self, state_size, action_size, seed, hidden_units = [512, 256, 128]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5),  # 32@84*84*3
            nn.BatchNorm2d(32, momentum=1,affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32@40*40*3
 
            nn.Conv2d(32, 64, 5), # 64@37*37*3
            nn.BatchNorm2d(64, momentum=1,affine=True),
            nn.ReLU(inplace=True),    
            nn.MaxPool2d(2),   # 64@18*18*3
           
            nn.Conv2d(64, 64, 5), # 64@15*15*3
            nn.BatchNorm2d(64, momentum=1,affine=True),
            nn.ReLU(inplace=True),    
            nn.MaxPool2d(2),   # 64@7*7*3
            
            nn.Conv2d(64, 128, 5), # 128@10*10*3
            nn.BatchNorm2d(128, momentum=1,affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.seq = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            
            nn.Linear(2048, 1048),
            nn.ReLU(inplace=True),
           
            nn.Linear(1048, hidden_units[0]),
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
#         print("~~~~~~~~~~~~~~~~~~~~~")
        x = self.conv(state)
        x = x.view(x.size()[0], -1)
        h = self.seq(x)
        out = self.out(h)
       
        return out