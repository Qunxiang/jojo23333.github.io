---
title: TensorFlow 学习笔记 #1
date: 2018-03-26 10:10:32
tags: [tensorflow]
---
# TensorFlow 学习笔记 #1
## TensorFlow Basics
### Graphs and Sessions
首先，需要和普通python程序区别出来的是，tensorflow将计算的定义和具体执行过程分离开来，个人认为这有点像函数式中的求值运算。  

不过tensorflow的特点在于其把所依赖的所有计算转换成一个数据流图（dag）

![数据流图](./img/tensor_data_flow_graph.png)

1. 根据输入构成数据流图
2. 创建会话，执行操作

用图的优点有如下几个：  
1. 能够保存计算结果。只会运行你所期望得到值的子图。
2.  易于分布任务，进行分布式的计算
3.  Break computation into small, differential pieces to facilitate auto-differentiation
4.  Many common machine learning models are taught and visualized as directed graphs

**何为TensorFlow?**  
**Tensor**: An n-dimensional array  
0-d tensor: scalar (number)   
1-d tensor: vector  
2-d tensor: matrix  

## Tensorflow ops
### TensorBorad
Tensorborad 使用通过将图的节点信息和图中的操作记入event files当中来完成整个流程的可视化，使用如下代码创建event files以及停止记录
```python
# use tf.get_default_graph() to get default graph
writer = tf.summary.FileWriter([logdir], [graph])
# ...
writer.close()
```
之后运行python代码并打开tensorboard
```bash
$ python3 [my_program.py] 
$ tensorboard --logdir="./graphs" --port 6006
```
但是我们此时看到的图每个节点我们无法对上名字，这就要在定义图的时候给出其的名字
```python
a = tf.constant(2, name="a")
b = tf.constant(2, name="b")
x = tf.add(a, b, name="add")
```

### Some useful tricks
#### 查看protobuf
常数存储在函数的定义当中，通过查看图的protobuf(protocol buffer)来查看图定义当中的内容。
```python
import tensorflow as tf

my_const = tf.constant([1.0, 2.0], name="my_const")
print(tf.get_default_graph().as_graph_def())
```
Output :
```json
node {
  name: "my_const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\200?\000\000\000@"
      }
    }
  }
}
versions {
  producer: 24
}
```

#### 变量的声明和初始化
为了变量共享的方便 官方推荐使用tf.get_variable方法
```python
s = tf.get_variable("scalar", initializer=tf.constant(2)) 
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())
```
同时可以较简单的初始化变量
```python
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
```

#### Assign a variable
观察如下程序
```python
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
	sess.run(W.initializer)
	print(W.eval()) # >> 10
```
结果输出是10，但为什么不是100呢。注意的是，之前也说过，tensorflow的声明和运行是分离的，W.assign(100)创建了一个assign操作，但是我们并没有运行它，所以应该按照如下写
```python
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
	sess.run(assign_op)
	print(W.eval()) # >> 100
```

#### Sessions
会话独自保存值，因而假如有两个不同的会话对同一个变量进行操作，其得到最终的值也有可能不相同。  
有时候为了方便可以使用interactive session来隐式地run session
```python
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
print(c.eval()) # we can use 'c.eval()' without explicitly stating a session
sess.close()
```
tf.get_default_session()返回当前进程的默认session

#### the trap of lazy loading
考虑如下代码有什么不好的地方
```
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('graphs/lazy_loading', sess.graph)
	for _ in range(10):
		sess.run(tf.add(x, y))
	print(tf.get_default_graph().as_graph_def()) 
	writer.close()
```
sess.run(tf.add(x, y))这一句会将tf.add(x,y)这个操作创建10次，造成网络的大量冗余
考虑解决方案：
1. 总是将操作的定义与执行分离开来
2. 使用python的@property来保证你的函数只被调用一次