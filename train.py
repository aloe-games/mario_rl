import datetime
from pathlib import Path

from agent import Mario
from env import build_env
from metrics import MetricLogger

env = build_env()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(
    state_dim=(4, 21, 21),
    action_dim=env.action_space.n,
    save_dir=save_dir,
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
