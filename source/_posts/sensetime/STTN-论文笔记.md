---
title: STTN-论文笔记
date: 2018-09-15 18:28:09
tags: [deeplearning, computer vision]
categories: 论文笔记
---
# 论文笔记-STTN ECCV 2018 
## 前要
刚来sensetime的第一天，在工位上不知所措，下午挑了几篇单目深度估计的文章来看，然后...宇哥晚上跟我说来来来你把这篇文章看一看吧，然后就有了第一个任务。  
## 简介
这篇文章是ECCV2018的一篇文章 [Spatio-Temporal Transformer Network for Video Restoration.](http://openaccess.thecvf.com/content_ECCV_2018/html/Tae_Hyun_Kim_Spatio-temporal_Transformer_Network_ECCV_2018_paper.html)  
现在state of art的Video Restoration Method 通常使用了optical flow network来优化视频中帧与帧之间的短时的信息。然而这些方法大多数只关注相邻的一对帧之间的联系，从而忽略了视频中较长距离的帧之间的联系。这篇文章提出了一种网络结构（Spatio-Temporal Transformer Network）能够一次性处理多帧，从而解决了视频中的遮挡问题，也可以应用于视频超分辨率和视频去模糊等其它问题。

## Main Idea
这篇文章的Inspiration来自于Google的一篇文章[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)   
STN网络的实质就是训练了一个Grid Generator 来对原图进行变化，或者说对原图重新Sample  
见下图与对应公式，这样做的好处在于，弥补了神经网络对空间不变性的缺陷(spatial invariant),比如说对于下图的手写数字，重新采样后的图片一定程度上恢复了数字的旋转压缩，这让后面神经网络的准确率大大增加。

STTN采用了STN的思想，把二维扩展到了三维。原STN是通过预测一个二维的grid generator来生成采样点，而STTN则是通过预测多帧之间的Flow(可以理解为光流？)来确定一个在多帧之间的采样点。
有关STN 可以参考这里：https://kevinzakka.github.io/2017/01/18/stn-part2/

## Architect Detail
STTN network 的网络结构如下图所示
![STTN 网络结构](/images/sttn/architecture.png)
### spatio-Temporal Flow Estimation Network
传统的预测光流的方法常用相邻两张图像比较，比较多次之后得到结果，一是计算耗时，二是不可靠。  
STTN使用了一种[U-net](https://arxiv.org/abs/1505.04597v1)的网络结构，将所有帧stack到一起（H*W*C*T）作为网络的输入，输出(u,v,w)->(H*W*3)的光流
U-net的网络结构如下所示

### Differentiable Spatio-Temporal Sampler
这一块和STN中的Grid Generator相同，根据得到的Optical Flow对原图进行采样。公式如下
![](/images/sttn/formu_1.png)
这个公式看着唬人，实际上比STN的思想还要暴力简单,展开之后
![](/images/sttn/formu_2.png)
实际上想一想，就是把每一个点(x,y,t) 分别加上(u,v,w)的偏移量之后得到的新点，根据到其空间内最近四个点的距离加权求和

### Image Processing part
原图给了一个Video restoration的例子，使用的了Resblock*9? sttn结构这个东西好像可以配合各种网络用上，如下图所示。
![](/images/sttn/architecture2.png)

## 补充
这一篇文章目前好像还没有放出官方代码，数据集也不见踪影,更坑爹的是loss function和各种Test data给的十分不详细...然鹅宇哥让我实现一下(- ▽ -)"...  
Tensorflow实现与分析见另一篇文章 [传送门](../STTN-tf-Implementation/)