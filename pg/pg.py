import torch.nn as nn
import gym
import torch
import numpy as np
import time
from replaybuffer import ReplayBuffer
from worker import Worker
from logger import Logger


class PGTrainer(object):

    def __init__(self,config):
        self.epoch = config['epoch']
        self.iter_sgd_per_epoch = config['iter_sgd_per_epoch'] # sgd training times per epoch
        self.train_batch_size = config['train_batch_size'] #sgd train batch
        self.num_sample_trajs = config['num_sample_trajs'] # sample how many trajs per loop
        self.worker = Worker(config['worker_config']) # worker config
        self.buffer = ReplayBuffer(config['max_buffer_len'])  # buffer maximum length
        self.sample_before_train = config['sample_before_train'] # sample how many trajectories before training
        self.logger = Logger((config['logger_config'])) # logger's config
        self.num_log_steps = config['num_log_step'] # how many steps per log
        self.is_evaluation = config['is_evaluation'] # whether to evaluation
        self.num_eval_trajs = config['num_eval_trajs'] # the number of trajectories for evaluation
        self.stop_mean_reward = config['stop_config']['stop_mean_reward']

    def train(self):
        # 1.sample
        # 2.add to buffer
        # 3.update
        if len(self.buffer) < self.sample_before_train:
            print('Sampling for initializing buffer')
            self.buffer.add_rollouts(self.worker.collect_trajs(self.sample_before_train))
            print('Sampling Done length of buffer {}'.format(len(self.buffer)))

            for steps in range(self.epoch):
                #1.sample
                paths = self.worker.collect_trajs(self.num_sample_trajs)

                #2.add to buffer
                self.buffer.add_rollouts(paths)
                for j in range(self.iter_sgd_per_epoch):
                    samples_rollouts = self.buffer.sample_recent_rollouts(self.train_batch_size)
                    info = self.worker.update(samples_rollouts)

                self.logger.writer.add_scalar('running/reward', np.sum(samples_rollouts[0]['reward']), steps)
                self.logger.writer.add_scalar('running/loss', info['loss'], steps)
                self.logger.writer.add_scalar('running/model_out', info['model_out'][0][0], steps)
                print('epoch {}, loss {}, reward {}'.format(steps, info['loss'], np.sum(samples_rollouts[0]['reward'])))

                if steps % self.num_log_steps == 0:
                    # eval work policy
                    rewards = []
                    eval_mean_rewards = 0
                    if self.is_evaluation:

                        eval_episode_len = 0
                        for i in range(self.num_eval_trajs):
                            eval_traj = self.worker.collect_one_traj()
                            rewards.append(np.sum(eval_traj['reward']))
                            eval_episode_len += len(eval_traj['reward'])
                        eval_mean_rewards = np.mean(rewards)
                        eval_mean_rewards_std = np.std(rewards)
                        eval_max_rewards = np.max(rewards)
                        eval_min_rewards = np.min(rewards)
                        eval_episode_len = eval_episode_len / self.num_eval_trajs

                        self.logger.writer.add_scalar('eval/eval_mean_rewards', eval_mean_rewards, steps)
                        self.logger.writer.add_scalar('eval/eval_max_rewards', eval_max_rewards, steps)
                        self.logger.writer.add_scalar('eval/eval_min_rewards', eval_min_rewards, steps)
                        self.logger.writer.add_scalar('eval/eval_mean_rewards_std', eval_mean_rewards_std, steps)
                        self.logger.writer.add_scalar('eval/eval_episode_len', eval_episode_len, steps)
                        print('Eval: eval_mean_rewards {}, eval_mean_rewards_std {}'.format(eval_mean_rewards,eval_mean_rewards_std))


                    # wether it satisfied break condition
                    if self.stop_mean_reward:
                        if eval_mean_rewards >= self.stop_mean_reward:
                            print('Eval: eval_mean_rewards ',eval_mean_rewards)
                            break

if __name__ == '__main__':
    TIME = time.strftime('%Y%m%d%H%M%S')
    ENV = 'CartPole-v0'
    config = {
        'max_buffer_len': 10000,
        'epoch': 1000,
        'iter_sgd_per_epoch':1,
        'num_sample_trajs' : 1,
        'sample_before_train':4,
        'train_batch_size' : 100,

        'num_log_step' : 10,
        'is_evaluation': True,
        'num_eval_trajs':10,
        'stop_config':{
            'stop_mean_reward' : 200,
            },
        'worker_config': {
            'env': ENV,
            'is_render':False,
            'agent_config':
                {
                    'lr': 5e-3,
                    'discount_factor':0.99,
                    'is_adv_normlize': True,
                }
            },
        'logger_config':{
            'log_dir' :'./log/log_' + ENV + '_' + TIME + '/',
        },
        }
    trainer = PGTrainer(config)
    trainer.train()

