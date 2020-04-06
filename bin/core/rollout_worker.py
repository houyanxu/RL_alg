from bin.core.utils import Path
import gym
import torch


class RolloutWorker(object):
	def __init__(self, env_name,worker_config,policy):
		'''

		:param worker_config:
		'''
		self.env = gym.make(env_name)
		self.is_render = worker_config['is_render']
		self.policy = policy

	def collect_one_traj(self):

		ob = self.env.reset()
		if self.is_render:
			self.env.render()
		obs, actions, rewards, obs_next, dones = [], [], [], [], []
		while True:
			ob_t = torch.FloatTensor(ob)
			action = self.policy.compute_actions(ob_t)
			ob_next, reward, done, info = self.env.step(action)

			if self.is_render:
				self.env.render()

			obs.append(ob)
			rewards.append(reward)
			dones.append(done)
			actions.append(action)
			obs_next.append(ob_next)
			ob = ob_next
			if done:
				break

		if self.is_render:
			self.env.close()
		return Path(obs, actions, rewards, obs_next, dones)

	def collect_trajs(self, num_trajs):
		paths = []
		for i in range(num_trajs):
			path = self.collect_one_traj()
			paths.append(path)
		return paths

	def set_weights(self,weights):
		self.policy.set_weights(weights)

	def get_weights(self):
		return self.policy.get_weights()



