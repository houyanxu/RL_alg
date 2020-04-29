import numpy as np
import time
from bin.core.replaybuffer import ReplayBuffer
from bin.core.logger import Logger
from bin.core.rollout_worker import RolloutWorker
from bin.policy.pg.pgpolicy import PGPolicy
from bin.policy.dqn.dqn_policy import DQNPolicy
from bin.policy.dqn.dqn_config import DQN_DEFAULT_CONFIG
from bin.policy.pg.pg_config import PG_VF_CONFIG,PG_MC_CONFIG
from bin.policy.a2c.a2c import A2CPolicy
from bin.core.utils import convert_listofrollouts,Path
import torch
import gym
def get_policy_class(policy_class):
    if isinstance(policy_class,str) == False:
        raise TypeError('Need str type to initialize policy.Such as PG')
    if policy_class == 'PG':
        return PGPolicy
    if policy_class == 'DQN':
        return DQNPolicy
    if policy_class == 'A2C':
        return A2CPolicy
    else:
        raise TypeError('Unsupported RL algorithms')


def collect_one_step(env,policy):

    t = 0
    ob = env.reset()
    while True:
        ob_t = torch.FloatTensor(ob)
        action = policy.compute_actions(ob_t)
        ob_next, reward, done, info = env.step(action)
        t += 1
        print(t)
        print(done)
        yield (ob, action, reward, ob_next, done)
        ob = ob_next
        if done:
            ob = env.reset()
            t = 0

class Runner(object):

    def __init__(self,config):
        self.device = torch.device(config['device']) # assign device
        self.env = gym.make(config['env'])
        self.epoch = config['epoch']
        self.policy_class = config['policy_class']
        self.iter_sgd_per_epoch = config['iter_sgd_per_epoch'] # sgd training times per epoch
        self.train_batch_size = config['train_batch_size'] #sgd train batch
        self.num_sample_trajs = config['num_sample_trajs'] # sample how many trajs per loop
        self.policy = get_policy_class(config['policy_class'])(config['env'],config['policy_config'],self.device)
        self.rolloutworker = RolloutWorker(config['env'],config['worker_config'],self.policy) # worker config, workers is used to collect trajectories
        self.buffer = ReplayBuffer(config['max_buffer_len'])  # buffer maximum length
        self.sample_before_train = config['trajs_before_train'] # sample how many trajectories before training
        self.logger = Logger((config['logger_config'])) # logger's config
        self.num_log_steps = config['num_log_step'] # how many steps per log
        self.is_evaluation = config['is_evaluation'] # whether to evaluation
        self.num_eval_trajs = config['num_eval_trajs'] # the number of trajectories for evaluation
        self.stop_mean_reward = config['stop_config']['stop_mean_reward']
        self.num_update_target_network = config['num_update_target_network']
        self.train_in_episode =  config['train_in_episode']

    def evaluation(self,steps):
        # eval work policy
        rewards = []
        eval_mean_rewards = 0
        if self.is_evaluation:

            eval_episode_len = 0
            for i in range(self.num_eval_trajs):
                eval_traj = self.rolloutworker.collect_one_traj()
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

            print(
                'Eval: eval_mean_rewards {}, eval_mean_rewards_std {}'.format(eval_mean_rewards, eval_mean_rewards_std))
        # wether it satisfied break condition

        return eval_mean_rewards

    def sample(self):
        if self.policy_class == 'PG':
            return self.buffer.sample_recent_batch(self.train_batch_size)
        if self.policy_class == 'DQN':
            return self.buffer.sample_random_batch(self.train_batch_size)
        if self.policy_class == 'A2C':
            return self.buffer.sample_recent_batch(self.train_batch_size)

    def train(self):
        # 1.sample
        # 2.add to buffer
        # 3.update

        if len(self.buffer) < self.sample_before_train:
            print('Sampling for initializing buffer')
            self.buffer.add_rollouts(self.rolloutworker.collect_trajs(self.sample_before_train))
            print('Sampling Done length of buffer {}'.format(len(self.buffer)))
            collector = self.rolloutworker.collect_one_step()

            for steps in range(self.epoch):
                #1.sample
                paths = self.rolloutworker.collect_trajs(self.num_sample_trajs)

                #2.add to buffer
                self.buffer.add_rollouts(paths)
                if self.train_in_episode:
                    # # for i in range(10):
                    # while True:

                    sample_step = next(collector)
                    ob, action, reward, ob_next, done = sample_step
                    #print(done)
                    ob = np.expand_dims(ob, 0)
                    ac = np.expand_dims(action, 0)
                    next_ob = np.expand_dims(ob_next, 0)
                    done = np.expand_dims(done, 0)
                    r_ = np.expand_dims(reward, 0)
                    summed_r_ = np.expand_dims(reward, 0)
                    single_rollout = [Path(ob, ac, r_, next_ob, done,summed_r_)]
                    info = self.policy.train_on_batch(single_rollout)
                        # else:
                        #     break

                    # print('sample_step',sample_step)
                    # samples_rollouts = self.sample()
                    # obs, acs, next_obs, dones, r, un_r, summed_r = convert_listofrollouts(paths=samples_rollouts)
                    #
                    # for i in range(len(r)):
                    #     ob = np.expand_dims(obs[i],0)
                    #     ac = np.expand_dims(acs[i],0)
                    #     next_ob = np.expand_dims(next_obs[i],0)
                    #     done = np.expand_dims(dones[i],0)
                    #     r_ = np.expand_dims(r[i],0)
                    #     summed_r_ = np.expand_dims(summed_r[i],0)
                    #     single_rollout = [Path(ob,ac,r_,next_ob,done,summed_r_)]
                    #     info = self.policy.train_on_batch(single_rollout)

                else:
                    for iter in range(self.iter_sgd_per_epoch):
                        samples_rollouts = self.sample()
                        info = self.policy.train_on_batch(samples_rollouts)

                        if iter % self.num_update_target_network == 0:

                            self.policy.update_target_network()

                #3. boardcast to rolloutworker
                weights = self.policy.get_weights()
                self.rolloutworker.set_weights(weights)

                # log to tensorboard
                self.logger.writer.add_scalar('running/reward', np.sum(paths[0]['reward']), steps)
                self.logger.writer.add_scalar('running/loss', info['loss'], steps)
                self.logger.writer.add_scalar('running/model_out', info['model_out'][0][0], steps)
                print('epoch {}, loss {}, reward {}'.format(steps, info['loss'], np.sum(paths[0]['reward'])))

                if steps % self.num_log_steps == 0:
                    eval_mean_rewards = self.evaluation(steps)
                    if self.stop_mean_reward:
                        if eval_mean_rewards >= self.stop_mean_reward:
                            print('Eval: eval_mean_rewards ', eval_mean_rewards)
                            break

def run():
    TIME = time.strftime('%Y%m%d%H%M%S')
    ENV = 'CartPole-v0'
    POLICY_CLASS = 'A2C'
    config = {
        'policy_class': POLICY_CLASS,
        'env': ENV,

        'max_buffer_len': 1000,
        'epoch': 10000,
        'iter_sgd_per_epoch': 2,
        'num_sample_trajs': 1,
        'trajs_before_train': 1,
        'train_batch_size': 200,
        'num_update_target_network': 2,
        'num_log_step': 10,
        'is_evaluation': True,
        'num_eval_trajs': 3,
        'stop_config': {
            'stop_mean_reward': 200,
        },
        'device':'cuda:0',
        'policy_config':
            {
                'lr': 1e-3,
                'discount_factor': 0.99,
                'is_adv_normlize': True,
                'baseline':'vf', # mean_q(need summed reward) or vf(value function estimator)
            },

        'worker_config': {
            'is_render': False,
        },
        'train_in_episode':True,
        'logger_config': {
            'log_dir': './log/log_' + POLICY_CLASS+'_'+ ENV + '_' + TIME + '/',
        },
    }
    runner = Runner(config)
    runner.train()

if __name__ == '__main__':
    run()

