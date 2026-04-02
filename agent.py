import os
import torch
import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch.nn as nn
import torch.optim as optim
import random
import argparse

# Defining the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        # Hyperparameters
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss() # Loss function for training the DQN
        self.optimizer = None # This will be initialized after creating the DQN model in the run method

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}_log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")


    def run(self, is_training=True, render=False):
        env = gym.make('FlappyBird-v0', render_mode='human' if render else None, use_lidar=False)

        num_states = env.observation_space.shape[0] # i/p dim
        num_actions = env.action_space.n # o/p dim

        policy_dqn = DQN(num_states, num_actions).to(device)

        
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device) # Target network is only needed while training
            #copy the wt. and bias vals from policy => target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0

            # Initialize the optimizer after creating the DQN model
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha) 

            best_reward = float('-inf') # Initialize best reward to negative infinity

        else:
            # best policy => load the model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, weights_only=True))
            policy_dqn.eval() # Set the model to evaluation mode

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_reward = 0
            terminated = False

            while (not terminated and episode_reward < self.reward_threshold):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()  # Explore: select a random action
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()  # Exploit: select the best action

                # Processing: terminated => done
                next_state, reward, terminated, _, _ = env.step(action.item())

                episode_reward += reward

                #Create tensors for next_state and reward
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                                
                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                state = next_state

            if is_training:
                print(f"Episode: {episode+1} with Total Reward: {episode_reward} and Epsilon: {epsilon}")
            else:
                break 


            if is_training:
                # Epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if episode_reward > best_reward:
                    log_msg = f"best reward = {episode_reward} for episode={episode+1}"

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE) # Save the model after each episode (you can modify this to save only the best model)
                    best_reward = episode_reward # Update best reward if current episode's reward is higher

            if is_training and len(memory) > self.mini_batch_size:
                # get sample
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Sync target network with policy network
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0

    # env.close() -> this is comment out bcoz we want to manually stop our code.

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get experience from mini_batch => batch train 
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # Compute target Q values - if terminations=true => zero
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # Compute current Q values
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # loss calculation
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad() # Clear the gradients
        loss.backward() # Backpropagation
        self.optimizer.step() # Update the weights



if __name__ == "__main__":
    # Parse the command line inputs
    parser = argparse.ArgumentParser(description="Train or Test Model")
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters) # DQN learning

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)

# Command to run the code:
# For training: python agent.py flappybirdv0 --train