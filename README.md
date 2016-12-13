# Deep reinforcement learning and its application in dialogue system

## reinforcement learning
karpathy给出了js版[reinforcement learning](http://cs.stanford.edu/people/karpathy/reinforcejs/)和[rnn/lstm](http://cs.stanford.edu/people/karpathy/recurrentjs/)。其中RNN用于字符生成的demo，将[Paul Graham](http://www.paulgraham.com/articles.html)的诗集编码成RNN的权重，

http://it.sohu.com/20161202/n474728555.shtml
### deep reinforcement learning

 - [workshop](http://rll.berkeley.edu/deeprlworkshop/)
 - [zhihu整理](https://zhuanlan.zhihu.com/p/23600620)
 - [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/pdf/1604.06778v3.pdf)
 - [Deep Reinforcement Learning in Large Discrete Action Spaces](http://101.96.8.165/tx.technion.ac.il/~danielm/icml_workshop/12.pdf)
 - []()
 
[Mnih etal.](Playing Atari with Deep Reinforcement Learning)的Atari Game Playing游戏很好的描述Deep Reinforcement Learning的作用，用卷积神经网络对action-value函数Q(s,a)建模，

在Oxford的[lecture12](http://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/lecture12.pdf)和UCL的[lecture6](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/FA.pdf)都讲到了DQN，

两条主线PG和Q-learning，分别有两个知名例子，
Atari games：Q Learning with function approximation，
AlphaGo ：uses policy gradients with Monte Carlo Tree Search (MCTS)

### Policy Gradients
karpathy的[这篇文章](http://karpathy.github.io/2016/05/31/rl/)详细介绍了Policy Gradient，用PG学习Atari游戏。
 - Stochastic policy gradient Agent利用REINFORCE和LSTMs学习actor policy和value function baseline，在karpathy的文章中就是UP/DOWN的概率。
 - [Deterministic Policy Gradients](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Publications_files/deterministic-policy-gradients.pdf)
 
### Value Function Approximation
DQN的例子

karpathy在文章中说更多人倾向于用Policy Gradient，而不是Q-learning，因为PG是end-to-end，当调参好时，PG比Q-learning效果好。
### critic-Actor算法
参考[]()以Cart-pole为例，构建一个控制器，系统状态(theta,w,x,v)，critic记载reward v(theta,w,x,v)，然后Actor u=u(theta,w,x,v)+rn，F=Fmax(u)施加一个F给environment，通过获取大的V(theta,w,x,v)值，得到摆的直立状态，直立状态获取最大的reward。然而，V(theta,w,x,v)是未知的值，必须做函数近似来获取V。
 
 - critic来估计V(theta,w,x,v)
 - Actor

在对话系统中的应用
http://www.maluuba.com/blog/2016/11/23/deep-reinforcement-learning-in-dialogue-systems

[Policy Networks with Two-Stage Training for Dialogue Systems](https://arxiv.org/pdf/1606.03152v4.pdf)

## Implementations
参考[基于tensorflow的DQN](https://github.com/devsisters/DQN-tensorflow)，在

https://github.com/dennybritz/reinforcement-learning

## 展望
[John Schulman:nuts-and-bolts](http://101.96.8.165/rll.berkeley.edu/deeprlcourse/docs/nuts-and-bolts.pdf)


## 参考
 - Playing Atari with Deep Reinforcement Learning论文[Human-level control through deep reinforcement learning](http://home.uchicago.edu/%7Earij/journalclub/papers/2015_Mnih_et_al.pdf)

这篇论文中讲到当一个非线性函数近似比如神经网络用于表示Q函数时，RL通常不稳定，甚至发散，DQN用experience replay和fixed Q-targets来增加稳定性。

 - experience replay
 

## 强化学习学习资料

 1. [Oxford reinforcement learning lecture](http://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)
 2. [David Silver's RL class](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
 3. [Udacity RL class](https://classroom.udacity.com/courses/ud600/lessons/4676850295/concepts/46733448110923)
 4. [berkeley deep RL course](http://rll.berkeley.edu/deeprlcourse/) 这个课程是最全的，包括policy-gradient，action-value approximation等。

## dqn算法实现
参考[devsisters代码](https://github.com/devsisters/DQN-tensorflow)，这个代码依赖
 - Python 
 - gym
 - tqdm
 - OpenCV2
 - TensorFlow

 1. [OpenCV2安装](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)
 按流程将opencv git clone到/home/crawler/tensorflow/dqn/gym/DQN-tensorflow/opencv ，但这么做很繁琐，直接用命令apt-get install python-opencv 就可以安装完成，然后import cv2测试。
 2. gym安装
[OpenAI Gym](https://gym.openai.com/)是一个Reinforcement Learning算法的toolkit，对agent结构没有假设，并且兼容tensorflow和theano。它包括两部分：1.gym开源库；2.OpenAI Gym service。

/home/crawler/tensorflow/dqn/gym路径下，
 - case.py
 - case1.py
这个cart-pole的例子，是用actor-critic的方式，actor-critic是Policy Gradient的一种方法，[这篇文章](http://brain.cc.kogakuin.ac.jp/~kanamaru/NN/CPRL/)比较好的介绍了这个方法，参考论文 [A Survey of Actor-Critic Reinforcement Learning:Standard and Natural Policy Gradients](http://busoniu.net/files/papers/ivo_smcc12_survey.pdf)。例子中cart-pole就是environment，gym库的主要目的是提供environment的集合，通过下面的命令查看
```
from gym import envs
print(envs.registry.all())
```
在~/tensorflow/dqn/gym/DQN-tensorflow路径下，执行main.py程序，测试dqn。

### zhihu上的例子
https://zhuanlan.zhihu.com/p/21477488?refer=intelligentunit



### 搭建博客--未完

两种方法

 - JekyII机制：username.github.io，访问这个时，JekyII会解析username用户下，username.github.io项目的master分支
 - 阮：username/blog 的 gh-pages 分支。cxwangyi.github.com
阮的文章中详细记录每个文件每个目录的作用，但jekyll new命令可以直接生成这些文件和目录。

 - [jekyllrb quickstart](https://jekyllrb.com/docs/quickstart/) 

实现用localhost访问，没有结合github
```
~ $ gem install jekyll bundler
~ $ jekyll new myblog
~ $ cd myblog
新建一个名为Gemfile的文件，内容如下
source 'https://rubygems.org'
gem 'github-pages', group: :jekyll_plugins
~/myblog $ bundle install
~/myblog $ bundle exec jekyll serve
# => Now browse to http://localhost:4000
```

1. 按https://pages.github.com/设置io
2. Jekyll是一个静态地址生成器，先安装sudo apt install ruby，按照https://jekyllrb.com/docs/quickstart/安装Jekyll

Jekyll的好处：
1. 你可以用MD而不是HTML，
2. 添加Jekyll theme


https://www.zhihu.com/question/28123816

https://pages.github.com/
