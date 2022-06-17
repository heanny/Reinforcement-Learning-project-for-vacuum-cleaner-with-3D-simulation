import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.autograd as autograd 
import torch.nn.functional as F
import torch.optim as optim
# from NaivePER import NaivePrioritizedBuffer

# BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network 
CAPACITY_PER = 100000   # Priortised experience replay buffer capacity
BETA = 0.4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        replay_initial = CAPACITY_PER
        self.memory = NaivePrioritizedBuffer(replay_initial)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE, BETA)
                self.learn(experiences, GAMMA)

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
#         print("state_shape in learn before",state.shape)
        
        states      = torch.FloatTensor(state).squeeze(1)
        next_states = torch.FloatTensor(next_state).squeeze(1)
        actions     = torch.LongTensor(action).unsqueeze(1)
        rewards     = torch.FloatTensor(reward).unsqueeze(1)
        dones       = torch.FloatTensor(done).unsqueeze(1)
        indices     = torch.LongTensor(indices).unsqueeze(1)
        weights     = torch.FloatTensor(weights).unsqueeze(1)
        
#         print("next_state_size in learn",next_states.shape)
#         print("state_shape in learn",states.shape)
#         print("actions_shape in learn",actions.shape)
#         print("rewards_shape in learn",rewards.shape)
#         print("weights_shape in learn",weights.shape)
#         print("indicess_shape in learn",indices.shape)
#         print("dones_shape in learn",dones.shape)
        Q_target_next = self.qnetwork_target(torch.Tensor(next_states)).detach().max(1)[0].unsqueeze(1)
#         print("Q_target_next in learn",Q_target_next.shape)
        
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
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                  

        
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


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = deque(maxlen=capacity)
        self.batch_size = BATCH_SIZE
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
#         print("next_state.ndim",next_state.ndim)
        assert state.ndim == next_state.ndim
        
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        batch       = list(zip(*samples))
        
        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_states = np.concatenate(batch[3])
        dones       = batch[4]
        
        return (states, actions, rewards, next_states, dones, indices, weights)
    
    def update_priorities(self, batch_indices, batch_priorities):
#         print("update_priorities", batch_indices.shape, batch_priorities.shape)
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)