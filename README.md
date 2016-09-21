# Deep reinforcement learning and its application in dialogue system

## reinforcement learning
karpathy给出了js版[reinforcement learning](http://cs.stanford.edu/people/karpathy/reinforcejs/)和[rnn/lstm](http://cs.stanford.edu/people/karpathy/recurrentjs/)。其中RNN用于字符生成的demo，将[Paul Graham](http://www.paulgraham.com/articles.html)的诗集编码成RNN的权重，

### deep reinforcement learning
[Mnih etal.](Playing Atari with Deep Reinforcement Learning)的Atari Game Playing游戏很好的描述Deep Reinforcement Learning的作用，用卷积神经网络对action-value函数Q(s,a)建模，

在Oxford的[lecture12](http://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/lecture12.pdf)和UCL的[lecture6](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/FA.pdf)都讲到了DQN，

### Policy Gradients
 - a stochastic policy gradient Agent利用REINFORCE和LSTMs学习actor policy和value function baseline。
 - [Deterministic Policy Gradients](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Publications_files/deterministic-policy-gradients.pdf)
 - 
 
## 测试
参考[基于tensorflow的DQN](https://github.com/devsisters/DQN-tensorflow)，在

## 参考
 - Playing Atari with Deep Reinforcement Learning论文[Human-level control through deep reinforcement learning](http://home.uchicago.edu/%7Earij/journalclub/papers/2015_Mnih_et_al.pdf)

## 强化学习学习资料

 1. [Oxford reinforcement learning lecture](http://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)
 2. [David Silver's RL class](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
 3. [Udacity RL class](https://classroom.udacity.com/courses/ud600/lessons/4676850295/concepts/46733448110923)

## 算法实现
[OpenAI Gym](https://gym.openai.com/)提供了一个Reinforcement Learning的toolkit，

对话论文中



