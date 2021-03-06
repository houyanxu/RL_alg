from collections import deque
from bin.core.utils import convert_listofrollouts,Path
import numpy as np
#TODO : BATCH CLASS

class Batch(object):
	def __init__(self):
		REWARDS = 'cur_rewards'
		CUR_OBS = 'cur_obs'
		NEXT_OBS = 'next_obs'
		DONES = 'dones'
		ACTIONS = 'actions'

		self.expiences_pool = deque(maxlen=int(1e6)) #TODO：add to config
	def concat_paths(self,paths):

		obs, acs, next_obs, dones, r, un_r, summed_r = convert_listofrollouts(paths=paths)
		for i in range(len(r)):
			self.expiences_pool.append(np.array((obs[i], acs[i], next_obs[i], dones[i], r[i], summed_r[i])))

	def sample_random_batch(self,batch_size):
		rand_indices = np.random.permutation(len(self.expiences_pool))[:batch_size]
		batch_list = np.array(self.expiences_pool)[rand_indices]
		# obs = np.array(batch_list[:,0],dtype=np.float32)
		# acs = np.array(batch_list[:,1].tolist(),dtype=np.float)
		# next_obs = np.array(batch_list[:,2].tolist(),dtype=np.float)
		# dones = np.array(batch_list[:,3].tolist(),dtype=np.float)
		# rewards = np.array(batch_list[:,4].tolist(),dtype=np.float)
		# summed_rewards = np.array(batch_list[:,5].tolist(),dtype=np.float)

		obs = np.stack(batch_list[:,0])
		acs = np.stack(batch_list[:,1])
		next_obs = np.stack(batch_list[:,2])
		dones = np.stack(batch_list[:,3])
		rewards = np.stack(batch_list[:,4])
		summed_rewards = np.stack(batch_list[:,5])

		return [Path(obs,acs,rewards,next_obs,dones,summed_rewards)]

	def sample_recent_batch(self,batch_size):
		batch_list = np.array(self.expiences_pool)[-batch_size:]
		obs = np.array(batch_list[:,0].tolist(),dtype=np.float)
		acs = np.array(batch_list[:,1].tolist(),dtype=np.float)
		next_obs = np.array(batch_list[:,2].tolist(),dtype=np.float)
		dones = np.array(batch_list[:,3].tolist(),dtype=np.float)
		rewards = np.array(batch_list[:,4].tolist(),dtype=np.float)
		summed_rewards = np.array(batch_list[:,5].tolist(),dtype=np.float)

		return [Path(obs,acs,rewards,next_obs,dones,summed_rewards)]

	def __getitem__(self, index):
		return self.expiences_pool[index]
