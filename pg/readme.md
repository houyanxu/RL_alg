# Policy Gradient 
*by Hou*

take tons of code from [CS285](http://rail.eecs.berkeley.edu/deeprlcourse/) hw 

## Theory
equation 

## Algorithm
>repeat
>
>    1. following $\pi(a|s)$ collect trajs $(s',a,s,done)$ 
>
>    2. update $\pi(a|s)$ using trajs

## structure 

### *class Trainer* 
* description 
    * implement policy iteration loop 
        * collect samples 
        * update worker 
        * log infos

### *class Worker*
* do_rollouts
* update agent
* TODO: extent to parallel worker in the future

### *class Agent*
* interact with environment 
* monte carlo calculate q_vals $\sum_{t=0}^T(r(s,a))$
* calculate log_pi $log\pi(a|s)$
* compute actions $a_t$ according to $s_t$
### *class AgentModel* 
* subclass of *torch.nn.Module*
* predict policy params

### *class Policy*
* discrete action -> *categorical*
* continue action -> *Normal*

### *class ReplayBuffer*
* add trajs to buffer
* sample trajs from buffer 
    * sample random
    * sample recent 

### *class Logger* 
* using tensorboardX to log infos 

## Results


test for vcs
    

    
    



