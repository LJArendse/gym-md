import gym
import gym_md
import random

env = gym.make('md-collab-gene_1-v0')
#env = gym.make('md-collab-test-v0')

# Currently the only md-collab-simple-v0 env supports the dual action_type option
action_type = 'path'
action_type = 'directional'
env = gym.make('md-collab-simple-v0', action_type=action_type)

LOOP: int = 1000
TRY_OUT: int = 1

for _ in range(TRY_OUT):
    observation = env.reset()
    reward_sum = 0
    for i in range(LOOP):
        env.render(mode='human')
        actions = [random.random() for _ in range(7)]
        actions = [0.98,0.0,1.0,0.99,0.0,0.0,0.95]
        c_1actions = [0.98,0.0,1.0,0.99,0.0,0.0,0.95]
        # direction action input
        # DIRECTIONAL_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
        c_1actions = [1.0, 1.0, 0.0, 0.0] 

        from pprint import pprint
        pprint(env.grid.g)
        if action_type=='directional':
            observation, reward, done, info = env.step([c_1actions, c_1actions])
        else:
            observation, reward, done, info = env.step([actions, actions])
        print(observation, reward, done, info)

        reward_sum += reward

        if done:
            env.render()
            break

    print(reward_sum)
