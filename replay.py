from pathlib import Path

from agent import Mario
from env import build_env

env = build_env()

checkpoint = Path("checkpoints/trained_mario.chkpt")
mario = Mario(
    state_dim=(4, 21, 21),
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
