import torchvision
import torch
import gym
from bin.core.model import FCModel
from bin.core.utils import convert_listofrollouts
from bin.core.utils import ActionDist
import numpy as np
class DQNPolicy(object):
	def __init__(self,env_name, policy_config,device = 'cpu'):
		self.device = device
		self.env = gym.make(env_name)
		self.obs_dim = self.env.observation_space.shape[0]
		if isinstance(self.env.action_space, gym.spaces.Box):
			raise TypeError('Unsupported action type')
		elif isinstance(self.env.action_space, gym.spaces.Discrete):
			self.action_dim = self.env.action_space.n
		else:
			raise ValueError('unsupport action ', type(self.action_dim))

		self.model = FCModel(self.obs_dim, self.action_dim).to(device)
		self.dist = ActionDist(self.env)
		self.target_model = FCModel(self.obs_dim, self.action_dim).to(device)
		self.discount_factor = policy_config['discount_factor']
		self.lr = policy_config['lr']
		self.is_adv_normlize = policy_config['is_adv_normlize']
		self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr,momentum=0.9)

	def clip_val(self,x,min=-100,max=100):
		x = torch.clamp(x,min,max)
		return x

	def train_on_batch(self,rollouts_batch):
		# loss = r+q(st) - q (st+1), minimums loss
		obs, acs, next_obs, dones, r, un_r, summed_r = convert_listofrollouts(paths=rollouts_batch)
		acs = torch.tensor(acs).long().to(self.device)
		obs = torch.FloatTensor(obs).to(self.device)
		next_obs = torch.FloatTensor(next_obs).to(self.device)
		acs_one_hot = torch.eye(2).to(self.device).index_select(0,acs)# to one hot
		dones = torch.IntTensor(dones).to(self.device)
		r = torch.FloatTensor(r).to(self.device)

		self.optimizer.zero_grad()

		# remove softmax
		q_t = self.model(obs)
		#q_t = self.clip_val(q_t)
		q_tp1 = self.target_model(next_obs)
		target = r + self.discount_factor * torch.max(q_tp1,dim=1).values * (1-dones)
		q_vals = torch.sum(q_t * acs_one_hot,dim=1)

		loss = torch.nn.functional.mse_loss(target,q_vals).to(self.device)

		loss.backward()
		self.optimizer.step()
		info = {'loss': loss.cpu().detach().numpy(),  # scale
				'model_out': q_t,  # torch.tensor [sum(batch), ac_dim],
				}
		return info

	def update_target_network(self):
		self.target_model.load_state_dict(self.model.state_dict())

	def get_weights(self):
		return {k:v for k,v in self.model.state_dict().items()}

	def set_weights(self,weights):
		self.model.load_state_dict(weights)

	def compute_actions(self, obs):
		obs = obs.to(self.device)
		exploration_rate = 0.1
		if np.random.random() < exploration_rate :
			return self.env.action_space.sample()
		else:
			return torch.softmax(self.model(obs),dim=0).argmax().cpu().detach().numpy()