from utils import Path
from agent import Agent
import gym
import torch

class Worker(object):
    def __init__(self, worker_config):
        self.env = gym.make(worker_config['env'])
        self.is_render = worker_config['is_render']
        self.agent = Agent(self.env, worker_config['agent_config'])

    def collect_one_traj(self):

        ob = self.env.reset()
        if self.is_render:
            self.env.render()
        obs, actions, rewards, obs_next, dones = [], [], [], [], []
        while True:
            ob_t = torch.FloatTensor(ob)
            action = self.agent.compute_actions(ob_t)
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

    def update(self, batch):
        info = self.agent.train_on_batch(batch)
        return info
