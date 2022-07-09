import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from wrappers import CutAndScaleObservation, SkipFrame


def build_env(name):
    env = gym_super_mario_bros.make(name)
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = CutAndScaleObservation(env)
    env = TransformObservation(env, f=lambda x: x / 255.0)
    env = FrameStack(env, num_stack=4)

    return env
