# Weight Agnostic Neural Networks

This repository contains an implementation of [Weight Agnostic Neural Networks](https://arxiv.org/abs/1906.04358).
WANN is a type of neural network with fixed weights, all set to the same constant value. WANNs aren't trained with gradient-based methods but with combinatorial optimization algorithms. Starting with a small single-layer network,
their role is adding new nodes and connections and modifying them. We benchmark the networks using 
4 reinforcement learning tasks available in [OpenAi gym](https://gym.openai.com), so the goal of optimization is maximization of expected reward.


For each environment 4 results are shown:
* Graph of the final network. The first layer is input - state of environment. The last output layer contains agent's controls.
* Plots of evolution. Agent (network) during optimization is evaluated on three criteria. The first two are maximized,
  the last one minimized. Plots show the evolution of these criteria for the best network (blue),
  the worst network (green) and average across all networks in population (orange). Criteria are:
  - average fitness - how well given architecture performs across couple of different values of weights
  - max fitness - fitness on the best weight
  - number of connections
 

* Fitness of the final selected architecture across many values of weights. The best networks are invariant to change of weight
  (like in CartPole case); their behavior is determined only by architecture and irrelevant of weights.
* GIF showing behavior of the final selected network with weights set to best value maximizing fitness.


### CartPole-v1
<p float="left">
  <img src="/pics/cartpole/net.jpg" width="300" />
  <img src="/pics/cartpole/plot.jpg" width="300" /> 
</p>
<p float="left">
  <img src="/pics/cartpole/scores.jpg" width="300" />
  <img src="/pics/cartpole/render.gif" width="300" /> 
</p>

### LunarLanderContinuous-v2
<p float="left">
  <img src="/pics/lander/net.jpg" width="300" />
  <img src="/pics/lander/plot.jpg" width="300" /> 
</p>
<p float="left">
  <img src="/pics/lander/scores.jpg" width="300" />
  <img src="/pics/lander/render.gif" width="300" /> 
</p>

### MountainCar-v0
<p float="left">
  <img src="/pics/mountain/net.jpg" width="300" />
  <img src="/pics/mountain/plot.jpg" width="300" /> 
</p>
<p float="left">
  <img src="/pics/mountain/scores.jpg" width="300" />
  <img src="/pics/mountain/render.gif" width="300" /> 
</p>

### BipdealWalker-v3
<p float="left">
  <img src="/pics/walker/net.jpg" width="300" />
  <img src="/pics/walker/plot.jpg" width="300" /> 
</p>
<p float="left">
  <img src="/pics/walker/scores.jpg" width="300" />
  <img src="/pics/walker/render.gif" width="300" /> 
</p>
