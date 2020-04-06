import gym
from bin.core.model import FCModel
# TODO: policy parent class
class Policy(object):
	def __init__(self,env_name,policy_config):
		self.env = gym.make(env_name)
		self.model = (FCModel if policy_config['model']==None else policy_config['model'])
	def get_weights(self):
		pass