import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import argparse
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)
        
        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                              lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory, alg_type="full", reinforce_lambda=0.0, ppo_lambda = 1.0):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        #for _ in range(self.K_epochs):
            # Evaluating old actions and values :
        ### Tab backwards as it's no longer in for loop
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
        
        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - old_logprobs.detach())
            
        # Finding Surrogate Loss:
        advantages = rewards - state_values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

        ###################### REINFORCE loss ##################### Added in 24/sep/2019
        reinforce_loss = - (logprobs * rewards) / len(rewards)
        ###########################################################
        
        if alg_type == "full":
            loss = ppo_lambda * (- torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy) + reinforce_lambda * reinforce_loss
        elif alg_type == "clipped-only":
            loss = ppo_lambda * (-torch.min(surr1, surr2) + reinforce_lambda * reinforce_loss)
        elif alg_type == "clipped-value":
            loss = ppo_lambda * (-torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards)) + reinforce_lambda * reinforce_loss
        else:
            raise Exception("Unkown option ({}) passed to this function".format(alg_type))
        
        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        ### End of tab backward
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def read_args():
    parser = argparse.ArgumentParser(description="Configuring the algorithm")
    parser.add_argument("--ppo_algorithm", default="full", help="clipped-only/clipped-value/full (clipped-value-entroy)")
    parser.add_argument("--reinforce_lambda", default=0.0, help="Shows the proportion of REINFORCE")
    parser.add_argument("--ppo_lambda", default = 1.0, help="Show the proportion of PPO")
    return parser.parse_args()
# reinforce_lambda=0.0, ppo_lambda = 1.0
def main():
    
    ############## Hyperparameters ##############
    ###
    args = read_args()
    alg_type = args.ppo_algorithm
    reinforce_lambda = float(args.reinforce_lambda)
    ppo_lambda = float(args.ppo_lambda)
    ###
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    solved_reward = 600         # stop training if avg_reward > solved_reward
    #solved_reward = 230   # MAIN      # stop training if avg_reward > solved_reward
    log_interval = 50           # print avg reward in the interval
    max_episodes = 20000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    ####
    d = datetime.datetime.now()
    formatted_date = "{}-{}-{}_{}.{}.{}".format(d.year,d.month,d.day,d.hour,d.minute,d.second)
    log_file_name = "PPO_{0}_{1}_ppo-{2}_REINF-{3}_{4}.log".format(alg_type, env_name,ppo_lambda,reinforce_lambda,formatted_date) ##Added by Fred
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    #print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            # Saving reward:
            memory.rewards.append(reward)
            
            # update if its time
            #if timestep % update_timestep == 0:
            if done == True: # instead of updating in update_timesteps, update it every episde, this makes it ready for REINFORCE too
                ppo.update(memory, alg_type, reinforce_lambda=reinforce_lambda, ppo_lambda = ppo_lambda)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward
            if render:
                env.render()
            if done:
                break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            #print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            with open(log_file_name, "a+") as fh:
                fh.write('Episode {} \t avg length: {} \t reward: {}\n'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
