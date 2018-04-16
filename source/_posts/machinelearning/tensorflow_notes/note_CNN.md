---
title: Convolutional Nerual Networks for Visual Recongnition
date: 2018-04-12 22:58:00
tags: [CNN,DeepLearning]
---

# Convolutional Nerual Networks (CNNs/ConvNets)

## Over view
### 引入
首先为啥会有CNN这个东西呢？  
一个普通的神经网络的示意图如下所示  
![](./images/tf/simple_neural_net.jpeg)
可见，这种神经网络层与层之间是全连接的，对于minist这种数据集使用，假设输入图像为32*32*3 = 3072个节点，勉强可以处理。但是对于更大的输入图像，200*200*3 = 120000个神经元节点，这种神经网络处理起来就比较费力。  
很显然，这种时候全连接就显得比较无用和浪费，大量的参数不仅难以优化，而且会快速的导致网络的过拟合。  

### Architecture
卷积神经网络同样由许多层构成，其中主要的有三种：
1. Concoluntional Layer (卷积层)
2. Polling Layer ()
3. Full-Connected Layer ()

一个较为典型的架构是：[INPUT-CONV-RELU-POOL-FC]
* INPUT: 3-d [width * height * color-channels]
* CONV : 卷积层
* RELU : Rectified Linear Unit (线性整流函数) 常用的有斜坡函数(max(0,x))
* POOL : 对输入进行下降抽样（输出向量的前两位小于输入）
* FC: 全连接的层  

其中只有CONV层和FC层是包含所需要优化的参数的。

## Layers
### Convolutional Layer (卷积层)
我们知道在高维度的输入下，全连接不太实际。取而代之的是，可以对每个节点和输入的某个局部的区域连接。而如何选择这个区域，由一组超参数（hyperparameter）决定，这被称为神经元的（receptive field），也就是filter size.

#### 输出维度（spatial-arrangement）
输出的空间维度由三个超参数决定：
* Depth: 输出的深度等于用到的filter的个数。可以理解为：不同的filter试图在数据里面找到不同的特征。
* Stride: Stride可以理解为对filter滑动的间距。当Stride较大的时候，输出的维度较小。（通常情况下1、2）
* Zero-padding: 有时候在特定Stride值下，不能整除的时候周围输入就要补零。

$W =$ input volume size  
$P =$ receptive field size of the conv layer nerons  
$S =$ stride  
$P =$ amount of zero-padding  
则有：  
$(W-F+2P)/S + 1$则为一个filter所对应的CONV Layer的节点数。

#### 参数共享（parameter sharing）
在Conv Layer Local connectivity的情况下，假设输入向量大小为[a* b * c], 输出[x * y* z], filter [n * m * c]。那么Conv Layer一共有xyz个节点，每一个节点有nmc个参数，一共有xyznmc个参数，取x = 55, y = 55, z = 96,n = 11,m = 11, c = 3。这种数量级仍然是难以接受的。

可以通过一个合理的假设大量减少参数的数量，可以认为如果某个特征在某一点是有效的，那么在其它点其是同样有效的。也就是说，限制Conv layer在每一个filter（depth）下的神经元使用同样的参数和bias，总的参数数量可以快速减少到zmnc。（在back propogation当中，同意深度下使用相同参数的神经元的贡献会被相加）

Conv Layer 的计算过程如图所示：  
![](./images/tf/convolution.png)

#### Two key insights：
关于CONV Layer的两个关键点  
1) Features are hierarchical
Composing high-complexity features out of low-complexity features is more
efficient than learning high-complexity features directly.
e.g.: having an “circle” detector is useful for detecting faces… and basketballs
2) Features are translationally invariant
If a feature is useful to compute at (x, y) it is useful to compute that feature at
(x’, y’) as well

ps: 为何叫卷积层呢：因为其与两个信号的卷积类似。  



### Pooling Layer (不知道咋翻译..)
Pooling Layer常被加在连续的Conv Layer当中，它的主要作用是逐步减少空间大小来减少参数的数量，从而控制过拟合。  

Pooling层独立的作用于各个depth slice。

一个常见的例子是使用2*2的filter，stride为2,使用max function，取四激励中最大的，从而忽略掉75%的激励

当然还有一些其它pooling的方法，如average pooling和L2-norm pooling在此mark以后深入研究。

