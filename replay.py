from pathlib import Path

import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from agent import Mario
from wrappers import CutObservation, SkipFrame

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
env = JoypadSpace(env, [["right"], ["right", "A"]])

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = CutObservation(env)
env = TransformObservation(env, f=lambda x: x / 255.0)
env = FrameStack(env, num_stack=4)

checkpoint = Path("checkpoints/trained_mario.chkpt")
mario = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    checkpoint=checkpoint,
)
mario.exploration_rate = mario.exploration_rate_min

episodes = 10
total_reward = 0.0

for e in range(episodes):
    state = env.reset()
    while True:
        env.render()
        action = mario.act(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        mario.cache(state, next_state, action, reward, done)
        state = next_state
        if done or info["flag_get"]:
            break

print(total_reward / episodes)
