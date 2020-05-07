### 20200426
1. 解决每次输出一步的生成器的问题，问题在于生成器中应独立拥有一个env，不能与class共用
2. 修改了PG中的vf,mean_q,PG算法实际上还是采用了Monte Carlo方法，其中就算有critic模型(value function)也只能作为baseline使用。目前这两种方法都能在200次以内迭代成功。
3. 增加了A2C方法，增加了td_error，但是不收敛。
4. TODO:需要把DQN方法改过来。
5. TODO:需要把runner类更好的包装起来使用多种方法。如果on-policy和off-policy之间差距过大，那么就考虑写两个runner.

### 20200427
1. fixed dqn, reach convergence at 90-120 epoch, lr = 5e-4, num_sgd_train = 8
2. 增加了dqn，pg_mc,pg_vf的 default_config
3. A2C还是不收敛

### 20200502
1. 可能是找到了A2C不收敛的问题，问题在set_weights时对model没有修改
2. 需要修改rollout_worker里面,以使用ddpg

### 20200507
1. 修改了ob->obs一些小的bug。ddpg 在pendulum-v0,OK，参考借鉴了spinup