from bin.core.utils import Path
import gym
import torch
from normalized_env import NormalizedEnv

class RolloutWorker(object):
	def __init__(self, env_name,worker_config,policy):
		'''

		:param worker_config:
		'''
		self.env_name = env_name
		self.env = gym.make(env_name)
		self.is_render = worker_config['is_render']
		self.policy = policy

		#self.policy.discount_factor

	def collect_one_step(self,noise_scale=0):#独立的，这个是拥有自己的env
		env = gym.make(self.env_name)
		ob = env.reset()
		while True:
			ob_t = torch.FloatTensor(ob)
			action = self.policy.compute_actions(ob_t,noise_scale)
			ob_next, reward, done, info = env.step(action)
			yield (ob,action,reward,ob_next,done)
			ob = ob_next
			if done:
				ob = env.reset()

	def collect_one_traj(self,noise_scale=0):

		ob = self.env.reset()
		if self.is_render:
			self.env.render()
		obs, actions, rewards, obs_next, dones, summed_rewards = [], [], [], [], [], []
		summed_reward = 0
		while True:
			ob_t = torch.FloatTensor(ob)
			action = self.policy.compute_actions(ob_t,noise_scale)
			ob_next, reward, done, info = self.env.step(action)

			if self.is_render:
				self.env.render()

			obs.append(ob)
			rewards.append(reward)
			dones.append(done)
			actions.append(action)
			obs_next.append(ob_next)
			ob = ob_next

			summed_reward = reward + self.policy.discount_factor * summed_reward
			summed_rewards.append(summed_reward)

			if done:
				break
		summed_rewards.reverse()
		if self.is_render:
			self.env.close()
		return Path(obs, actions, rewards, obs_next, dones, summed_rewards)

	def random_collect_one_traj(self):

		ob = self.env.reset()
		if self.is_render:
			self.env.render()
		obs, actions, rewards, obs_next, dones, summed_rewards = [], [], [], [], [], []
		summed_reward = 0
		while True:
			action = self.env.action_space.sample()
			ob_next, reward, done, info = self.env.step(action)

			if self.is_render:
				self.env.render()

			obs.append(ob)
			rewards.append(reward)
			dones.append(done)
			actions.append(action)
			obs_next.append(ob_next)
			ob = ob_next

			summed_reward = reward + self.policy.discount_factor * summed_reward
			summed_rewards.append(summed_reward)

			if done:
				break
		summed_rewards.reverse()
		if self.is_render:
			self.env.close()
		return Path(obs, actions, rewards, obs_next, dones, summed_rewards)

	def collect_trajs(self, num_trajs, noise_scale):
		paths = []
		for i in range(num_trajs):
			path = self.collect_one_traj(noise_scale)
			paths.append(path)
		return paths

	def random_collect_trajs(self, num_trajs):
		paths = []
		for i in range(num_trajs):
			path = self.random_collect_one_traj()
			paths.append(path)
		return paths


	def set_weights(self,weights):
		self.policy.set_weights(weights)

	def get_weights(self):
		return self.policy.get_weights()



