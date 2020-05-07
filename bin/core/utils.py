import numpy as np
import torch
import gym
class ActionDist(object):
    def __init__(self,env):
        self._sigma = torch.FloatTensor(np.log([1]))
        self.env = env

    def get_act_dist(self,params):
        if isinstance(self.env.action_space, gym.spaces.Box):
            dist = torch.distributions.normal.Normal(loc=params, scale=torch.tensor([self._sigma]))
        elif isinstance(self.env.action_space, gym.spaces.Discrete):
            dist = torch.distributions.categorical.Categorical(logits=params)
        else:
            raise ValueError('unsupport action', type(self.action_dim))
        return dist

def Path(obs, acs, rewards, next_obs, terminals,summed_reward = None):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
        (hyx) a completely trajectory
        Path(obs,actions,rewards,obs_next,dones)
    """

    return {"observation" : np.array(obs, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32),
            "summed_reward": np.array(summed_reward, dtype=np.float32),
            }

def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths]).astype(np.float32)
    actions = np.concatenate([path["action"] for path in paths]).astype(np.float32)
    next_observations = np.concatenate([path["next_observation"] for path in paths]).astype(np.float32)
    terminals = np.concatenate([path["terminal"] for path in paths]).astype(np.float32)
    concatenated_rewards = np.concatenate([path["reward"] for path in paths]).astype(np.float32)
    unconcatenated_rewards = [path["reward"] for path in paths]
    concatenated_summed_rewards = np.concatenate([path["summed_reward"] for path in paths]).astype(np.float32)
    
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards, concatenated_summed_rewards

