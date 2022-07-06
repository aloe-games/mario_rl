import datetime
from pathlib import Path

import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from agent import Mario
from metrics import MetricLogger
from wrappers import ResizeObservation, SkipFrame

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
env = JoypadSpace(env, [["right"], ["right", "A"]])

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.0)
env = FrameStack(env, num_stack=4)

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

checkpoint = None
mario = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    checkpoint=checkpoint,
)

logger = MetricLogger(save_dir)

episodes = 40000

for e in range(episodes):
    state = env.reset()
    while True:
        action = mario.act(state)
        next_state, reward, done, info = env.step(action)
        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn()
        logger.log_step(reward, loss, q)
        state = next_state
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
