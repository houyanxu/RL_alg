from collections import deque
import numpy as np
from bin.core.sample_batch import Batch

class ReplayBuffer(object):
    def __init__(self,maxlen):
        '''
        replay buffer
        :param maxlen:
        '''
        self.paths = deque(maxlen=maxlen)
        self.max_size = maxlen
        self.batch = Batch()

    def add_rollouts(self,paths):
        for path in paths:
            self.paths.append(path)
        self.batch.concat_paths(paths)

    def sample_random_rollouts(self, num_rollouts): # for off policy
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return np.array(self.paths)[rand_indices]

    def sample_recent_rollouts(self,num_rollouts):
        return np.array(self.paths)[-num_rollouts:]

    def sample_random_batch(self,batch_size):
        batch = self.batch.sample_random_batch(batch_size)
        return batch

    def sample_recent_batch(self,batch_size):
        batch = self.batch.sample_recent_batch(batch_size)
        return batch

    def __len__(self):
        return len(self.paths)