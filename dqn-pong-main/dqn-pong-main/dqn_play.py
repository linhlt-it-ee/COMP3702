#!/usr/bin/env python3
import gym
import time
import argparse
import numpy as np

import torch

from lib import wrappers
from lib import dqn_model

import collections

from utils import load_hyperparams

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="trained_models/PongNoFrameskip-v4-best.dat", help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-n", "--network", default='duelling-dqn', help="DQN type - one of dqn, double-dqn and duelling-dqn")
    parser.add_argument("-c", "--config_file", default="config/dqn.yaml", help="Config file with hyper-parameters")

    args = parser.parse_args()

    params = load_hyperparams(args)

    env = wrappers.make_env(args.env, render_mode='human')
    env.reset()
    env.render('rgb_array')

    if args.network == 'duelling-dqn':
        net = dqn_model.DuellingDQN(env.observation_space.shape, params["hidden_size"], env.action_space.n)
    else:
        net = dqn_model.DQN(env.observation_space.shape, params["hidden_size"], env.action_space.n)

    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)

    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()

