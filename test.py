import gym
def collect_one_step(env):

    t = 0
    ob = env.reset()
    while True:
        action = env.action_space.sample()
        ob_next, reward, done, info = env.step(action)
        t += 1
        print(t)
        print(done)
        yield (ob, action, reward, ob_next, done)
        ob = ob_next
        if done:
            ob = env.reset()
            t = 0

def generator(num = 10):
    for i in range(num):
        yield i


if __name__ == '__main__':
    env =gym.make('CartPole-v0')
    a = collect_one_step(env)
    while True:
        i = next(a)
        print(i)
