#!/usr/bin/env bash
import pickle
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from net import Net
import numpy as np

def dfs(i, out):
    q = out.children[:]
    if i in q:
        return True
    if q == []:
        return False
    for node in q:
        if dfs(i, node):
            return True
    return False


def main():
    weights = []
    with open(f"../models/best_net_BipedalWalker-v3.pickle", "rb") as f:
        n = pickle.load(f)
    for out in n.outputs:
        temp_w = [n.best_w for c in out.children]
        weights.append(temp_w)
        for i in n.inputs:
            if not dfs(i, out):
                out.children.append(i)
                weights[-1].append(np.random.normal(0,0.1))
    with open(f'../models/best_net_BipedalWalker-v3_added_connections.pickle', 'wb') as f:
        pickle.dump(n, f)
    with open('out_weights.txt', 'w+') as f:
        for line in weights:
            f.write(str(line)[1:-1]+'\n')


if __name__ == '__main__':
    main()
