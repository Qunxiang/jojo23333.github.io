---
title: STTN tf Implementation
date: 2018-09-17 17:03:36
tags: [tensorflow, deeplearning, computer vision]
categories:
---
# STTN tensorflow 实现
## 前要
这篇文章是论文 [Spatio-Temporal Transformer Network for Video Restoration.](http://openaccess.thecvf.com/content_ECCV_2018/html/Tae_Hyun_Kim_Spatio-temporal_Transformer_Network_ECCV_2018_paper.html)的实现，之前有写过这篇论文的解析 [传送门](../STTN-论文笔记)
第一次尝试用tensorflow复现论文..（；´д｀）ゞ而且还是从头写，虽然之前也片片段段的写过一点，不过从头写感觉就是不一样啊,这里先贴一点我用到的资料吧。  
* 我发现tensorflow所有文档里面除了api以外..[guide](https://tensorflow.google.cn/programmers_guide/)最好用..
* STN --> STNN STN Offical Repo: https://github.com/kevinzakka/spatial-transformer-network
* 尝试理解一个构造模式也不错，由于原论文refer到了我参考了VDSR 的 tensorflow-repo: https://github.com/Jongchan/tensorflow-vdsr

## network architecture
虽然network architecture在之前一篇文章里面也有提到过，这里就提一些细节吧，整体的网络结构和
* U—net 包括下采样层和上采样层两个部分，与U-net论文里提到的U-net不同，这里的U-net下采样通过stride=2的卷积层来做到，而不是池化层。而上采样使用的是双线性插值。（这里面的两点区别有待仔细考虑，例如用反卷积会怎么样？）
* Spatio-Temporal Sampler 实现类似于STN（其实觉得这种能自动算gradient很神奇，有时间仔细观察一下tensorboard理解一下这里咋back propoagate）
* Image proccesing part 我就简单的使用了他video restoration network 里面的resnet-9

tensorboard 可视化之后结果大概这样：  

| unet部分 | res9部分 |
| :-  | :-  |
| ![](/images/sttn/unet.png) | ![](/images/sttn/res9.png) |


## 有关数据集
### Self-made VideoScenes 数据集
这个论文既不公开代码也不公开数据集=。=。我参考了这篇论文refer的一些论文用的数据集，发现很多都是通过下载网站上的HD Video来搞的。然后同样是参考了这篇论文refer的论文的一些数据。发现它download的一些数据集大多是一些运动类的高清数据集。于是我也采用了类似的数据集（不过video还是自己手动一个一个去download的）。  
下好20多个时长在5~10分钟不等的video后，我发现一个严重的问题，那就是这些视频大多数由多个连续镜头剪辑在一起形成。而且有的还有片头片尾，我必须要考虑如下的一些事情：
1. **场景分割**。 不能简单地把视频转化为帧，必须要考虑场景和场景之间在哪切分，不然训练到场景切换之间地数据就有问题
幸运的是，python有一个scenedetect的工具来处理这个问题，具体涉及到一些阈值的处理方式啥的可以参考[文档](https://pyscenedetect.readthedocs.io/en/latest/)

2. **分割间隔**。 视频取帧怎么设置间隔？  
我觉得这是个很难说的问题，因为有缓慢地视角移动也有快速的视角移动，但是按理来说，STTN的optical flow estimation Network应该是要能学到光流中包含的这些信息的，暂时来说我是取了0.1s为间隔

3. **数据清洗**  
有一些渐暗的视频切换效果比较难鉴别到，也有一些制作人表之类的Scene需要自己去掉

4. **数据加噪**
宇哥跟我建议先加高斯噪声试试..我就先每个frame resize之后加的高斯噪声..我觉得之后需要加上blur和down sample弄复杂一点..

意思意思贴一点数据处理的代码，估计以后还用的上PySceneDetect这个工具
```python
def split_video_by_scene(video):
    video_name = video.split('.')[-2].split('/')[-1]
    os.system("scenedetect --input \"" + video +  "\" list-scenes detect-content -t 25")

    scene = pd.read_csv(video_name + "-Scenes.csv", skiprows=1)
    scene_begin = list(scene["Start Time (seconds)"])
    scene_end = list(scene["End Time (seconds)"])
    end_frame = scene_end[-1]
    save_scene_images(video, scene_begin, scene_end, end_frame)
```

### GoPR 数据集
这个数据集是CVPR2017的一篇论文用到的数据集，具体戳 https://github.com/SeungjunNah/DeepDeblur_release
大概有2k张连续帧，1k多张测试用，其实很小了..
更大的数据集估计能有1w张清晰图，之后参考那个


## 实现当中踩过的一些坑和问题
### 

## Train过程中遇到的问题汇总
1. 一开始Train不起来 :(   
可能太年轻，一个1280* 720 * 3 的tensor,其对应的一个conv2d没想到竟然会占到450M的显存，分分钟爆掉11G的1080Ti....,batch_size=1才勉强能跑。无奈只好downsize输入，1280 * 720 -> 640 * 360。   
为了保证训练的数据集像素不变，每次训练随机地crop 640*360大小的图片出来然后feed进去。由于数据集本身不是很大，这种方法应该也从某种意义上扩大了数据集？（在原数据集上random crop）
2. 能训练了之后，针对自己的数据集训练的第一版效果不是很好，很多有噪声的地方被模糊化了，估计是optical flow network没有训练好。用tensorboard加上了一些输出信息，观察到正则loss占了很大部分比重，减小正则项，同时使用decayrate 来减少有时候出现的diverge现象。以上均属于hyperparameter finetuning 过程。
3. 用并行输入优化，尝试加快训练过程。由于现在仍是在单卡上训练，后续考虑tensorflow的多卡数据并行训练。
4. 怀疑之前的问题很大程度是光流网络输出的问题。尝试visualize光流网络的训练结果。同时考虑替换后续image processing network为densenet。
5. 有关loss, 论文里没明确给loss的表达式子，现在是joint train两个loss ,xt~ yt~相对于ground_truth的loss ,xt~ yt~如下图
![](/images/sttn/architecture2.png)

## 训练结果记录
### GoPR数据集第一批的训练结果
psnr基本在25上下，下面是一个结果截图
![一个效果一般的结果](/images/sttn/result_analyze1.png)
1. 效果一般，但是可以看到网络似乎是学到了通过预测光流用其它帧来补偿模糊(人脚那一块的模糊)
2. 考虑改善后处理网络
3. 我plot了一下total loss 和每训练一个epoch得到的test loss如下图所示  

| train_mse | test_mse |
|:-|:-|
| ![](/images/sttn/total_mse.png)| ![](/images/sttn/test_mse.png) |


