## reinforcement learning在nlp中的应用

http://blog.dennybritz.com/2015/09/11/reimagining-language-learning-with-nlp-and-reinforcement-learning/

[Language Understanding for Text-based Games using Deep Reinforcement Learning](http://people.csail.mit.edu/karthikn/pdfs/mud-play15.pdf)
笔记：

论文为text-based games学习控制策略，采用deep reinforcement learning架构来学习状态表示和动作策略，游戏的rewards作为反馈。这个架构使我们可以将
文本映射为向量表征，这样会捕捉游戏状态的语义。

选择状态表示：bag of words忽略词顺序。本文采用reinforcement learning架构，将游戏序列组成MDP，智能体学习一个策略，这个策略是以action-value函数Q(s,a)
的形式表示，Q函数表示长期的收益，用deep rnn网络来拟合Q函数，rnn有两个模块，第一个将文本转化为向量表示，这部分是用LSTM执行的；第二个模块
在给定第一个计算出来的向量表示的情况下，对动作评分。


