import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork, NoisyNetwork_Dueling_PER, DuelingNetwork_PER, Dueling_Network, NoisyNetwork, NDDQNetwork, NoisyNetwork_PER, QNetwork_PER

from experience_class import NaivePrioritizedBuffer, ReplayBuffer 

import torch
import torch.autograd as autograd 
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network 
CAPACITY_PER = 100000   # Priortised experience replay buffer capacity
BETA = 0.2              # Priortised experience replay importance sample beta
PROB_ALPHA = 0.6        # how much prioritization is used, prob_alpha = 0 corresponding to the uniform case (i.e. without PER).


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_withPER():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, Dueling_flag, Noisy_flag, capacity, gamma, prob_alpha, beta, batch_size, update_every, tau, lr):
        """Initialize an Agent object with PER.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            capacity (int): Priortised experience replay buffer capacity
            seed (int): random seed
            gamma (float): discount factor
            update_every (int): how often to update the network 
            Dueling_flag (boolean): The flag of using Dueling DQN algorithm
            Noisy_flag (boolean):  The flag of using Noisy Net algorithm
            prob_alpha (float): the level of prioritization
            batch_size (int): minibatch size
            beta (float): Priortised experience replay importance sample beta
            tau (float): for soft update of target parameters
            LR (float): learning rate 
        """
        self.state_size = state_size
        self.action_size = action_size
        self.capacity = capacity
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.update_every = update_every
        self.Dueling_flag = Dueling_flag
        self.Noisy_flag = Noisy_flag
        self.prob_alpha = prob_alpha
        self.batch_size = batch_size
        self.beta = beta
        self.tau = tau
        self.lr = lr
        
        # All DQN Network variants with PER 
        
        # For DQN + Dueling + Noisy + PER = NoisyNetwork_Dueling_PER
        if self.Dueling_flag == True and self.Noisy_flag == True:
            self.qnetwork_local = NoisyNetwork_Dueling_PER(state_size, action_size, seed).to(device)
            self.qnetwork_target = NoisyNetwork_Dueling_PER(state_size, action_size, seed).to(device)
            
        # For DQN + Dueling + PER = DuelingNetwork_PER
        if self.Dueling_flag == True and self.Noisy_flag == False:
            self.qnetwork_local = DuelingNetwork_PER(state_size, action_size, seed).to(device)
            self.qnetwork_target = DuelingNetwork_PER(state_size, action_size, seed).to(device)
        
        # For DQN + Noisy + PER = NoisyNetwork_PER
        if self.Dueling_flag == False and self.Noisy_flag == True:
            self.qnetwork_local = NoisyNetwork_PER(state_size, action_size, seed).to(device)
            self.qnetwork_target = NoisyNetwork_PER(state_size, action_size, seed).to(device)
            
        # For DQN + PER = QNetwork_PER
        if self.Dueling_flag == False and self.Noisy_flag == False:
            self.qnetwork_local = QNetwork_PER(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork_PER(state_size, action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        replay_initial = self.capacity
        self.memory = NaivePrioritizedBuffer(replay_initial, prob_alpha = self.prob_alpha, batch_size = self.batch_size)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """
        Make a step of the agent to save experience in replay memory and learn every UPDATE_EVERY time steps.
        
        Params
        ======
            state : the current state
            action : the chosen action
            reward (float): reward of the chosen action
            next_state: the next state after propose the chosen action
            done (boolean): whether the episode ends.
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size, self.beta)
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(torch.Tensor(state))
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        state, action, reward, next_state, done, indices, weights = experiences 
        
        states      = torch.FloatTensor(state).squeeze(1)
        next_states = torch.FloatTensor(next_state).squeeze(1)
        actions     = torch.LongTensor(action).unsqueeze(1)
        rewards     = torch.FloatTensor(reward).unsqueeze(1)
        dones       = torch.FloatTensor(done).unsqueeze(1)
        indices     = torch.LongTensor(indices).unsqueeze(1)
        weights     = torch.FloatTensor(weights).unsqueeze(1)
        

        Q_target_next = self.qnetwork_target(torch.Tensor(next_states)).detach().max(1)[0].unsqueeze(1)
        
        Q_target = rewards + gamma * (Q_target_next * (1- dones))
        
        # Old value Q(state, action, w)
        Q_local = self.qnetwork_local(states).gather(1, actions)
        
        #compute the loss and minimize the loss
        loss  = (Q_local - Q_target.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss  = loss.mean()
                                             
        # update weights in local_network (optimizer object is linked to local_network parameters, see initialisations above)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priorities(indices, prios.detach().numpy())
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                  

        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
class Agent_withoutPER():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, batch_size, buffer_size, Dueling_flag, Noisy_flag, update_every, gamma, tau, lr):
        """Initialize an Agent object without PER.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            gamma (float): discount factor
            update_every (int): how often to update the network 
            Dueling_flag (boolean): The flag of using Dueling DQN algorithm
            Noisy_flag (boolean):  The flag of using Noisy Net algorithm
            batch_size (int): minibatch size
            tau (float): for soft update of target parameters
            LR (float): learning rate 
            buffer_size (int): replay buffer size
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.Dueling_flag = Dueling_flag
        self.Noisy_flag = Noisy_flag
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
           
        # All DQN Network variants without PER 
        
        # For DQN + Dueling + Noisy = NDDQNetwork
        if self.Dueling_flag == True and self.Noisy_flag == True:
            self.qnetwork_local = NDDQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = NDDQNetwork(state_size, action_size, seed).to(device)
            
        # For DQN + Dueling = Dueling_Network
        if self.Dueling_flag == True and self.Noisy_flag == False:
            self.qnetwork_local = Dueling_Network(state_size, action_size, seed).to(device)
            self.qnetwork_target = Dueling_Network(state_size, action_size, seed).to(device)
        
        # For DQN + Noisy = NoisyNetwork
        if self.Dueling_flag == False and self.Noisy_flag == True:
            self.qnetwork_local = NoisyNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = NoisyNetwork(state_size, action_size, seed).to(device)
            
        # For vanille DQN = QNetwork
        if self.Dueling_flag == False and self.Noisy_flag == False:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.batch_size, self.buffer_size, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """
        Make a step of the agent to save experience in replay memory and learn every UPDATE_EVERY time steps.
        
        Params
        ======
            state : the current state
            action : the chosen action
            reward (float): reward of the chosen action
            next_state: the next state after propose the chosen action
            done (boolean): whether the episode ends.
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size :
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).squeeze(2).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # TD_target (R + gamma * max_a(Q(next_state, actions , w`))) 
        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_target = rewards + gamma * (Q_target_next * (1- dones))

        # Old value Q(state, action, w)
        Q_local = self.qnetwork_local(states).gather(1, actions)
        
        #loss
        loss = F.mse_loss(Q_target, Q_local)
                                             
        # update weights in local_network (optimizer object is linked to local_network parameters, see initialisations above)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                  

        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

