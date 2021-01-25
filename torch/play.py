#!/usr/bin/env python3

import gym
import numpy as np
import torch
import sys
from totorch import TorchNet

def t(s):
    return torch.from_numpy(s).float()

class Player:
    def __init__(self):
        self.env = gym.make("LunarLanderContinuous-v2")
        self.actor = torch.load('actor_lunar.pth', map_location=torch.device('cpu'))
        self.actor.eval()

    def play(self):
        done = False
        total_rew = 0
        s = self.env.reset()

        with torch.no_grad():
            while not done:
                self.env.render()
                a = self.actor(t(s)).detach()
                s, rew, done, _= self.env.step(a.numpy())
                s = s
                total_rew += rew
        self.env.close()
        print(total_rew)

def main():
    Player().play()

if __name__ == "__main__":
    main()


