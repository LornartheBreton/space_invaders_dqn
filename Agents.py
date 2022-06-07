import random, copy
import torch
import numpy as np
from collections import deque
from torch import nn


class ShipNet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        channels, _, _ = input_dim

        self.primary = nn.Sequential(  # Neural network of the original q value
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.primary)  # Network of the target Q value

        # Q target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "primary":
            return self.primary(input)
        elif model == "target":
            return self.target(input)


class Ship:
    def __init__(self,
                 state_dim,
                 action_dim,
                 save_dir,
                 exploration_rate=1,
                 exploration_rate_decay=0.99999975,
                 exploration_rate_min=0.1,
                 batch_size=32,
                 gamma=0.9,
                 lr=0.00025,
                 experiences_before_training=10000,
                 learn_every=3,
                 sync_every=10000):
        self.state_dim = state_dim #Tuple containing the dimensions of each state
        self.action_dim = action_dim #Int containing the actions to take
        self.save_dir = save_dir #Directory to save Checkpoints to

        self.use_cuda = torch.cuda.is_available()  # If available, use CUDA.
        self.lr = lr  # The learning rate

        self.net = ShipNet(self.state_dim, self.action_dim).float()  # Initialize the neural network
        if self.use_cuda:  # Move if to the GPU if available
            self.net = self.net.to(device="cuda")

        self.exploration_rate = exploration_rate  # Epsilon
        self.exploration_rate_decay = exploration_rate_decay  # Every step, Epsilon = Epsilon * exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min  # Minimmal value Epsilon can take
        self.curr_step = 0  # The current step

        self.replay_buffer = deque(maxlen=1000)  # Start up replay_buffer with a queue length 1000
        self.batch_size = batch_size  # Number of experiences to retrieve from memory
        self.Q = None  # The current primary Q value
        self.gamma = gamma  # The discount factor used in the Bellman equation
        self.loss = 0  # The current loss
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean') #The loss function,
                                                # used this because it uses mean as a criterion and
                                                # its less sensisitve to outliers (which is common in Atari games),
                                                # unlike MSE
                                                # (https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)

        self.experiences_before_training = experiences_before_training  # min. experiences before training
        self.learn_every = learn_every  # number of experiences between updates to primary Q
        self.sync_every = sync_every  # number of experiences before the target and primary Q values are synced

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        :parameter: state(LazyFrame): A single observation of the current state, dimension is (state_dim)

        :returns:
        In a tuple:
            action_to_take (int): An integer representing which action to take
            action_values (np.array): An array representing the q values of all the actions
        """
        action_values = torch.tensor([[1, 1, 1, 1, 1, 1]])
        # EXPLORE - Take a random action
        if np.random.rand() < self.exploration_rate:
            action_to_take = np.random.randint(self.action_dim)

        # EXPLOIT - Use the neural network to determine the action to take
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="primary")
            action_to_take = torch.argmax(action_values, axis=1).item()

        # Update exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # Update step count
        self.curr_step += 1
        return (action_to_take, action_values.cpu().detach().numpy())

    def push_to_memory_queue(self, state, next_state, action, reward, done):
        """
        Store the experience to self.replay_buffer.

        :parameter: state (LazyFrame): The current state of the game,
        :parameter: next_state (LazyFrame): The next state of the game,
        :parameter: action (int): The integer representation of the action to take,
        :parameter: reward (float): The reward obtained,
        :parameter: done(bool): Whether the episode is done executing
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.replay_buffer.append((state, next_state, action, reward, done,))

    def remember(self):
        """
        Retrieve a batch of experiences from memory of size self.batch_size.
        """
        batch = random.sample(self.replay_buffer, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """
        Estimate the primary Q value.

        :param state: The current state of the environment
        :param action: The action taken
        :return: The current primary Q
        """
        current_Q = self.net(state, model="primary")[
            np.arange(0, self.batch_size), action
        ]  # Q_primary(s,a)
        return current_Q

    @torch.no_grad()  # WE DON'T WANT TO UPDATE THE TARGET Q JUST YET
    def td_target(self, reward, next_state, done):
        """
        Estimate the target Q value.

        :param reward: The reward obtenied on the next state.
        :param next_state: The next state.
        :param done: Whether the episode is done.
        :return: The estimated target Q value
        """
        next_state_Q = self.net(next_state, model="primary")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_primary_Q(self, td_estimate, td_target):
        """
        Backpropagate and update the primary Q neural network.

        :param td_estimate: The function to estimate the primary Q.
        :param td_target: The function to estimate the target Q.
        :return: The current loss.
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target_Q(self):
        self.net.target.load_state_dict(self.net.primary.state_dict())

    def learn(self):
        """Update primary action value (Q) function with a batch of experiences"""

        if self.curr_step % self.sync_every == 0:
            self.sync_target_Q()

        if self.curr_step < self.experiences_before_training:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.remember()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_primary
        self.loss = self.update_primary_Q(td_est, td_tgt)
        self.Q = td_est.mean().item()

        return self.Q, self.loss

    def save(self):
        """
        Save the model parameters to a Pytorch .chkpt file in the self.save_dir directory.

        :return: save_path (String): The full path to the saved file.
        """
        save_path = (
                self.save_dir / f"space-invaders_net_{self.curr_step}.chkpt"
        )
        torch.save(
            dict(net=self.net.state_dict(),
                 exploration_rate=self.exploration_rate,
                 exploration_rate_decay=self.exploration_rate_decay,
                 exploration_rate_min=self.exploration_rate_min,
                 batch_size=self.batch_size,
                 gamma=self.gamma,
                 lr=self.lr,
                 burn_in=self.experiences_before_training,
                 learn_every=self.learn_every,
                 sync_every=self.sync_every,
                 curr_step=self.curr_step,
                 memory=self.replay_buffer
                 ),
            save_path,
        )
        print(f"Agent saved to {save_path} at step {self.curr_step}")
        return save_path

    def load(self, load_path):
        """
        Loads a model from a Pytorch .chkpt file.

        :param load_path (String): The path to the .chkpt file to load.
        :return:
        """
        checkpoint = torch.load(load_path)

        self.net.load_state_dict(checkpoint['net'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.exploration_rate_decay = checkpoint['exploration_rate_decay']
        self.exploration_rate_min = checkpoint['exploration_rate_min']
        self.batch_size = checkpoint['batch_size']
        self.gamma = checkpoint['gamma']
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=checkpoint['lr'])
        self.experiences_before_training = checkpoint['burn_in']
        self.learn_every = checkpoint['learn_every']
        self.sync_every = checkpoint['sync_every']
        self.curr_step = checkpoint['curr_step']
        self.replay_buffer = checkpoint['memory']
