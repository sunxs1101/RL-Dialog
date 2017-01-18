# chatbot from scratch
[Three challenges you’re going to face when building a chatbot](https://blog.infermedica.com/three-challenges-youre-going-to-face-when-building-a-chatbot/)

## steps
按照simpleDS和Deng li‘s paper做
 1. make clear the SimpleDS， deep reinforcement learning
 2. RL-based

## SimpleDS: use deep Q-learning
A Simple Deep Reinforcement Learning Dialogue System，参考[github](https://github.com/cuayahuitl/SimpleDS)中的步骤，进行train,test， This system 
runs using a client-server architecture, where the 'server' is your system and the 'client' is the Deep Reinforcement Learner.

The SimpleDS learning agent is based on the [ConvNetJS](http://cs.stanford.edu/people/karpathy/convnetjs/) which implements the algorithm `Deep Q-Learning with experience replay`, it's a Javascript library for training Deep Learning models (Neural Networks) entirely in your browser. 
. SimpleDS extended this tool to support multi-threaded and client-server processing with constrained search spaces.

```
Cuayáhuitl (2016) proposed SimpleDS, which uses a multi-layer feed-forward network to directly map environment states
to agent actions. The network is trained using Q-learning and a simulated user; however it does not interact with a structured database, leaving that task to a server,
which may be suboptimal as we show in our experiments below.
```
~/work/chatbot/SimpleDS/resources/english/SlotValues.txt  SysResponses.txt  UsrResponses.txt， the difference between SysResponses.txt
and UsrResponse.txt 
```
Salutation
Request
Provide
ImpConfirm
ExpConfirm
Confirm
```
### other toolkit
[OpenAI Gym](https://github.com/openai/gym) is a toolkit for developing and comparing reinforcement learning algorithms.
## Simple RL & End-to-End
[End-to-End Reinforcement Learning of Dialogue Agents for Information Access](https://arxiv.org/pdf/1609.00777v2.pdf)

