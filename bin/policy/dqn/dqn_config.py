import time
TIME = time.strftime('%Y%m%d%H%M%S')
ENV = 'CartPole-v0'
POLICY_CLASS = 'DQN'
DQN_DEFAULT_CONFIG = {
    'policy_class': POLICY_CLASS,
    'env': ENV,

    'max_buffer_len': 1000,
    'epoch': 10000,
    'iter_sgd_per_epoch': 32,
    'num_sample_trajs': 1,
    'trajs_before_train': 2,
    'train_batch_size': 256,
    'num_update_target_network': 2,
    'num_log_step': 10,
    'is_evaluation': True,
    'num_eval_trajs': 3,
    'stop_config': {
        'stop_mean_reward': 200,
    },
    'device': 'cuda:0',
    'policy_config':
        {
            'lr': 5e-4,
            'discount_factor': 0.99,
            'is_adv_normlize': True,
            'baseline': 'vf',  # mean_q(need summed reward) or vf(value function estimator)
        },

    'worker_config': {
        'is_render': False,
    },
    'train_in_episode': False,
    'logger_config': {
        'log_dir': './log/log_' + POLICY_CLASS + '_' + ENV + '_' + TIME + '/',
    },
}