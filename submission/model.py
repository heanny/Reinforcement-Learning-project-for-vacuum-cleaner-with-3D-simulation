import math
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
    
class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))

        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)
            noise_v = torch.mul(eps_in, eps_out).detach()
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)

class NoisyNetwork_Dueling_PER(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_units = [100, 50, 20]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(NoisyNetwork_Dueling_PER, self).__init__()
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
            
            nn.Linear(hidden_units[1], hidden_units[2]),
            nn.ReLU(inplace=True),
        )
        self.adv_1_noisy = nn.Sequential(NoisyFactorizedLinear(hidden_units[2], hidden_units[2]), nn.ReLU(inplace=True))
        self.val_1_noisy = nn.Sequential(NoisyFactorizedLinear(hidden_units[2], hidden_units[2]), nn.ReLU(inplace=True))

        self.adv_2_noisy = NoisyFactorizedLinear(hidden_units[2], action_size)
        self.val_2_noisy = NoisyFactorizedLinear(hidden_units[2], out_features=1)

       
    def forward(self, state):
        """Build a network that maps state -> action values."""
        h = self.seq(state)
        h2 = F.relu(self.adv_1_noisy(h))
        
        adv_1_noisy = self.adv_1_noisy(h2)
        val_1_noisy = self.val_1_noisy(h2)
        
        adv_noisy = self.adv_2_noisy(adv_1_noisy)
        val_noisy = self.val_2_noisy(val_1_noisy)
        
        if state.size(0) > 1:
            out = val_noisy + adv_noisy - adv_noisy.mean(dim=0).expand(state.size(0), self.action_size)
        else:
            out = val_noisy + adv_noisy - adv_noisy.mean(dim=1).expand(state.size(0), self.action_size).unsqueeze(1)
        
        return out
    
    
class DuelingNetwork_PER(nn.Module):
    """Actor (Policy) Model."""
 
    def __init__(self, state_size, action_size, seed, hidden_units = [100, 50, 20]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingNetwork_PER, self).__init__()
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

class Dueling_Network(nn.Module):
    """Actor (Policy) Model."""
 
    def __init__(self, state_size, action_size, seed, hidden_units = [100, 50, 20]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Dueling_Network, self).__init__()
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
            out = val + adv - adv.mean(dim=2).expand(state.size(0), self.action_size).unsqueeze(1)
        return out


class NoisyNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_units = [100, 50, 20]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(NoisyNetwork, self).__init__()
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
        self.noisy1 = NoisyFactorizedLinear(hidden_units[2],hidden_units[2])
        self.noisy2 = NoisyFactorizedLinear(hidden_units[2],action_size)
       
    def forward(self, x):
        """Build a network that maps state -> action values."""
        h = self.seq(x)
        h2 = F.relu(self.noisy1(h))
        out = self.noisy2(h2)
        return out

class NDDQNetwork(nn.Module):
    """Actor (Policy) Model."""
 
    def __init__(self, state_size, action_size, seed, hidden_units = [100, 50, 20]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(NDDQNetwork, self).__init__()
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
        self.adv_1 = nn.Sequential(NoisyFactorizedLinear(hidden_units[1], hidden_units[2]), nn.ReLU(inplace=True))
        self.val_1 = nn.Sequential(NoisyFactorizedLinear(hidden_units[1], hidden_units[2]), nn.ReLU(inplace=True))
        
        self.adv_2 = NoisyFactorizedLinear(hidden_units[2], action_size)
        self.val_2 = NoisyFactorizedLinear(hidden_units[2], out_features=1)
       
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
            out = val + adv - adv.mean(dim=2).expand(state.size(0), self.action_size).unsqueeze(1)
        return out

class NoisyNetwork_PER(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_units = [100, 50, 20]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(NoisyNetwork_PER, self).__init__()
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
        self.noisy1 = NoisyFactorizedLinear(hidden_units[2],hidden_units[2])
        self.noisy2 = NoisyFactorizedLinear(hidden_units[2],action_size)
       
    def forward(self, x):
        """Build a network that maps state -> action values."""

        h = self.seq(x)
        h2 = F.relu(self.noisy1(h))
        out = self.noisy2(h2)
        return out
    
class QNetwork_PER(nn.Module):
    """Actor (Policy) Model."""
 
    def __init__(self, state_size, action_size, seed, hidden_units = [100, 50, 20]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork_PER, self).__init__()
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