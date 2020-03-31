import torch
import gym
import numpy as np
from bin.core.utils import convert_listofrollouts
from bin.core.model import FCModel

class GaussianDistAction(object):
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


class PGPolicy(object):
    def __init__(self,env_name,policy_config):
        self.env = gym.make(env_name)
        self.policy = GaussianDistAction(self.env)
        self.obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space,gym.spaces.Box):
            self.action_dim = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space,gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
        else:
            raise ValueError('unsupport action ',type(self.action_dim))

        self.model = FCModel(self.obs_dim,self.action_dim)

        self.discount_factor = policy_config['discount_factor']
        self.lr = policy_config['lr']
        self.is_adv_normlize = policy_config['is_adv_normlize']

        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr)

    def calculate_q_vals(self,r_list,discount_factor=0.99):
        '''

        :param r_list: un_concat rewards, type: class list, size: [N,episode_len]
        :param discount_factor:
        :return: q_vals : type: torch.tensor, size:[N,episode_len]
        '''

        dist_list = [discount_factor**i for i in range(200)]
        q_vals = []
        for r in r_list:
            q_vals.append([np.sum(r*dist_list[:len(r)])]*len(r)) # stable versions
        q_vals = torch.FloatTensor(np.concatenate(q_vals))
        return q_vals

    def compute_actions(self,obs):
        model_out = self.model(obs)
        act_dist = self.policy.get_act_dist(model_out)
        actions = act_dist.sample()
        return actions.numpy()

    def calculate_log_pi(self,obs,acs):

        obs_t = torch.FloatTensor(obs)
        acs_t = torch.FloatTensor(acs)

        model_out = self.model(obs_t)
        act_dist = self.policy.get_act_dist(model_out)
        log_pi = act_dist.log_prob(acs_t)
        return log_pi,model_out

    def get_weights(self):
        return {k:v for k,v in self.model.state_dict().items()}

    def set_weights(self,weights):
        self.model.load_state_dict(weights)

    def train_on_batch(self,rollouts_batch):
        '''
            return {"observation" : np.array(obs, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}
        :param batch:
        :return:
        '''
        #rollouts_batch
        # 1. calculate q_vals
        # 2. calculate log_pi
        # 3. train on batch
        obs,acs,next_obs,dones,r,un_r = convert_listofrollouts(paths=rollouts_batch)
        # 1. calculate q_vals
        q_vals= self.calculate_q_vals(un_r,self.discount_factor)# TODO: only one traj for now, extend to multi traj in the future
        adv_n = q_vals

        if self.is_adv_normlize:
            adv_n = (adv_n - torch.mean(adv_n))/(torch.std(adv_n) + 1e-8)

        # 2. calculate log_pi
        log_pi,model_out = self.calculate_log_pi(obs,acs)

        # 3. train on batch
        self.optimizer.zero_grad()
        loss = -torch.sum( log_pi * adv_n)
        loss.backward()
        self.optimizer.step()

        info = {'loss': loss.detach().numpy(), # scale
                'model_out':model_out, #torch.tensor [sum(batch), ac_dim],
                }
        return info