import torch
import gym
from bin.core.utils import convert_listofrollouts
from ddpg_model import Actor, Critic
import numpy as np
from random_process import OrnsteinUhlenbeckProcess
from normalized_env import NormalizedEnv
from copy import deepcopy
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class DDPGPolicy(object):
	def __init__(self,env_name, policy_config,device = 'cpu'):
		self.device = device
		self.env = gym.make(env_name) #仅仅用于设置observation
		self.obs_dim = self.env.observation_space.shape[0]
		if isinstance(self.env.action_space, gym.spaces.Box):
			self.action_dim = self.env.action_space.shape[0]
		elif isinstance(self.env.action_space, gym.spaces.Discrete):
			raise TypeError('Unsupported action type')
		else:
			raise ValueError('unsupport action ', type(self.action_dim))

		self.action_limit = self.env.action_space.high[0]
		self.lr = policy_config['lr']
		self.actor = Actor(self.obs_dim, self.action_dim).to(device)
		self.critic = Critic(self.obs_dim, self.action_dim).to(device)
		self.actor_target = deepcopy(self.actor)
		self.critic_target = deepcopy(self.critic)

		hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
		hard_update(self.critic_target, self.critic)

		self.actor_optim =  torch.optim.Adam(params=self.actor.parameters(), lr=0.001)
		self.critic_optim =  torch.optim.Adam(params=self.critic.parameters(), lr=0.001)
		self.discount_factor = policy_config['discount_factor']
		self.tau = 0.005


	def train_on_batch(self,rollouts_batch):
		# loss = r+q(st) - q (st+1), minimums loss
		obs, acs, next_obs, dones, r, un_r, summed_r = convert_listofrollouts(paths=rollouts_batch)
		acs = torch.tensor(acs).float().to(self.device)
		obs = torch.FloatTensor(obs).to(self.device)
		next_obs = torch.FloatTensor(next_obs).to(self.device)
		#acs_one_hot = torch.eye(2).to(self.device).index_select(0,acs)# to one hot discrete action space
		dones = torch.IntTensor(dones).to(self.device)
		r = torch.FloatTensor(r).to(self.device)

		# update critic
		self.critic_optim.zero_grad()
		act_target = self.actor_target(next_obs).to(self.device)
		q_target = r + self.discount_factor * self.critic_target(next_obs,act_target)* (1-dones)
		q_pred = self.critic(obs,acs)
		critic_loss = torch.nn.functional.mse_loss(q_pred,q_target)
		critic_loss.backward()
		self.critic_optim.step()

		#update actor
		self.actor_optim.zero_grad()
		actor_loss = -torch.mean(self.critic(obs,self.actor(obs)))
		actor_loss.backward()
		self.actor_optim.step()

		info = {'loss': actor_loss.cpu().detach().numpy(),  # scale
				'model_out': q_target,  # torch.tensor [sum(batch), ac_dim],
				}
		return info

	def update_target_network(self):
		soft_update(self.actor_target, self.actor, self.tau)
		soft_update(self.critic_target, self.critic, self.tau)

	def get_weights(self):
		#TODO: actor and critic parameters
		return {k:v for k,v in self.actor.state_dict().items()}

	def set_weights(self,weights):
		self.actor.load_state_dict(weights)

	def compute_actions(self, obs, noise_scale): #通过noise来判断是否需要增加噪声，如果是在eval中noise为0
		obs = obs.to(self.device)
		actions = self.actor(obs).cpu().detach().numpy()
		actions += noise_scale * np.random.rand(self.action_dim)
		actions = np.clip(actions, -self.action_limit, self.action_limit)

		return actions

	def reset(self):#在env.reset的同时需要reset random_process
		self.random_process.reset_states()
