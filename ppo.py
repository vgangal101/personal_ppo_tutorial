from network import FeedForwardNN
from torch.distributions import MultivariateNormal
import torch 
import numpy as np
import gym


"""
Tutorial location: 
https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
"""


class PPO: 

    def __init__(self,env):
        self._init_hyperparameters()
        
        self.env = env 
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim,1)

        #self.timesteps_per_batch = timesteps_per_batch
        #self.max_timesteps_per_episode = max_timesteps_per_episode

        self.cov_var = torch.full(size=(self.act_dim,),fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(),lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),lr=self.lr)

    def  _init_hyperparameters(self):
        # default values for hyperparameters 
        self.timesteps_per_batch = 4800 # timesteps per batch 
        self.max_timesteps_per_episode = 1600 #timesteps per episode 
        self.gamma = 0.95 # value of gamma for experiments 
        self.n_updates_per_iteration = 5
        self.clip = 0.2 # recommended by paper 
        self.lr = 0.005 


    def get_action(self,obs):
        mean = self.actor(obs)
        
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()


    def rollout(self):
        batch_obs = []      # storage list for obs, 
        batch_acts = []     # storage list for actions 
        batch_log_probs = []    # storage list for log probs 
        batch_rews = []         # storage list for rewards 
        batch_rtgs = []         # storage list for rewards to go 
        batch_lens = []         # storage list for ep lens , 

        # obs = self.env.reset()
        # done = False 
        # for ep_t in range(self.max_timesteps_per_episode):

        #     action = self.env.action_sample.sample()
        #     obs, rew, done, _ = self.env.step(action)

        #     if done: 
        #         break 
        t = 0 
        while t < self.timesteps_per_batch:
            # rewards this episode 
            ep_rews = []

            obs = self.env.reset()
            done = False 

            for ep_t in range(self.max_timesteps_per_episode):
                # increment timesteps ran this batch so far 
                t += 1 

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs) # need to implement get_action 
                obs, rew, done, _ = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done: 
                    break 
                    
            # collect episode length and batch rewards
            batch_lens.append(ep_t+1)
            batch_rews.append(ep_rews)

        # batch everything as pytorch tensors 
        batch_obs = torch.tensor(batch_obs,dtype=torch.float)
        batch_acts = torch.tensor(batch_acts)
        batch_log_probs = torch.tensor(batch_log_probs)

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


    def compute_rtgs(self,batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):

            discounted_rewards = 0 

            for rew in reversed(ep_rews):
                discounted_rewards = rew + discounted_rewards * self.gamma
                batch_rtgs.insert(0,discounted_rewards)
            
        batch_rtgs = torch.tensor(batch_rtgs,dtype=torch.float)
        return batch_rtgs


    def evaluate(self,batch_obs,batch_acts):
        # Query critic network for a value V for each obs in batch_obs 
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return V, log_probs




    def learn(self,total_timesteps):
        t_so_far = 0

        while t_so_far < total_timesteps:
            
            batch_obs , batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            
            # compute advantages 
            V, _ = self.evaluate(batch_obs,batch_acts)
            A_k = batch_rtgs - V.detach()

            # normalize the advatages 
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # for numerical stability

            for i in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs,batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1+ self.clip) * A_k

                actor_loss = (-1 * torch.min(surr1,surr2)).mean()
                critic_loss = torch.nn.MSELoss()(V,batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True) # WHY ???
                self.actor_optim.step()
            
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            
            t_so_far += np.sum(batch_lens)




            




if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    algo = PPO(env)
    algo.learn(10000)



