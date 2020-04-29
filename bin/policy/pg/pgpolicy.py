import torch
import gym
import numpy as np
from bin.core.utils import convert_listofrollouts,ActionDist
from bin.core.model import FCModel,FCCriticModel


class PGPolicy(object):
    def __init__(self,env_name,policy_config,device='cpu'):
        self.device = device
        self.env = gym.make(env_name)
        self.dist = ActionDist(self.env)
        self.obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space,gym.spaces.Box):
            self.action_dim = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space,gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
        else:
            raise ValueError('unsupport action ',type(self.action_dim))

        self.model = FCModel(self.obs_dim,self.action_dim).to(device)
        self.critic_model = FCCriticModel(self.obs_dim).to(device)
        self.discount_factor = policy_config['discount_factor']
        self.lr = policy_config['lr']
        self.is_adv_normlize = policy_config['is_adv_normlize']
        self.baseline = policy_config['baseline']
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        #lr_lambda = lambda epoch: 0.95 ** epoch
        #self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min')
        #self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda = lr_lambda)
        self.critic_optim = torch.optim.Adam(params=self.critic_model.parameters(),lr=1e-2)

    def calculate_q_vals(self,r_list,discount_factor=0.99):
        '''
        :param r_list: un_concat rewards, type: class list, size: [N,episode_len]
        :param discount_factor:
        :return: q_vals : type: torch.tensor, size:[N,episode_len]
        '''

        dist_list = [discount_factor**i for i in range(200)]
        q_vals = []
        for r in r_list:
            q_vals.append([np.sum(r*dist_list[:len(r)])]*len(r)) # a stable version
        q_vals = torch.FloatTensor(np.concatenate(q_vals))
        return q_vals

    def compute_actions(self,obs):
        obs = obs.to(self.device)
        model_out = self.model(obs).to(self.device)
        logits = torch.log_softmax(model_out,-1).to(self.device)
        act_dist = self.dist.get_act_dist(logits)
        actions = act_dist.sample()
        return actions.cpu().numpy()

    def calculate_log_pi(self,obs,acs):


        model_out = self.model(obs)
        logits = torch.log_softmax(model_out, -1)
        act_dist = self.dist.get_act_dist(logits)
        log_pi = act_dist.log_prob(acs)
        return log_pi,model_out

    def get_weights(self):
        return {k:v for k,v in self.model.state_dict().items()}

    def set_weights(self,weights):
        self.model.load_state_dict(weights)

    def update_target_network(self):
        pass
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
        obs,acs,next_obs,dones,r,un_r, summed_r = convert_listofrollouts(paths=rollouts_batch)
        obs_t = torch.FloatTensor(obs).to(self.device)
        acs_t = torch.FloatTensor(acs).to(self.device)


        #adv_n = target.cpu().detach().numpy() - vf_t_pred.cpu().detach().numpy()

        if self.baseline == 'mean_q':
            b = np.mean(summed_r)
        elif self.baseline == 'vf':
            vf_t_pred = self.critic_model(obs_t)
            b =  vf_t_pred.cpu().detach().numpy()
        else:
            b = 0
        q_vals = summed_r  # baseline
        adv_n = q_vals - b

        if self.is_adv_normlize:
            adv_n = (adv_n - np.mean(adv_n))/(np.std(adv_n) + 1e-8)

        adv_n = torch.FloatTensor(adv_n).to(self.device)

        # 2. calculate log_pi
        log_pi,model_out = self.calculate_log_pi(obs_t,acs_t)

        # 3. train on batch
        self.optimizer.zero_grad()
        loss = -torch.mean(log_pi * adv_n).to(self.device)
        loss.backward()
        self.optimizer.step()
        #self.lr_scheduler.step()
        if self.baseline == 'vf':
            self.critic_optim.zero_grad()
            summed_r = torch.FloatTensor(summed_r).unsqueeze(-1).to(self.device)
            vf_loss = torch.nn.functional.mse_loss(vf_t_pred, summed_r)
            vf_loss.backward()
            self.critic_optim.step()
            #print('vf_loss',vf_loss.cpu().detach().numpy())

        info = {'loss': loss.cpu().detach().numpy(), # scale
                'model_out':model_out, #torch.tensor [sum(batch), ac_dim],
                }
        return info