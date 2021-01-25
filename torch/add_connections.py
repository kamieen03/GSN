#!/usr/bin/env bash
import pickle
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from net import Net

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
    with open(f"../models/best_net_BipedalWalker-v3.pickle", "rb") as f:
        n = pickle.load(f)
    for out in n.outputs:
        for i in n.inputs:
            if not dfs(i, out):
                out.children.append(i)
    with open(f'../models/best_net_BipedalWalker-v3_added_connections.pickle', 'wb') as f:
        pickle.dump(n, f)


if __name__ == '__main__':
    main()
