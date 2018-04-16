---
title: TensorFlow 学习笔记2 #2
date: 2018-03-30 11:31:00
tags: [tensorflow]
---
# TensorFlow 学习笔记 #2
  
先来看一个简单的线性回归的代码例子，再来看在其基础上可以做出什么改进
```python
import tensorflow as tf

import utils

DATA_FILE = "data/birth_life_2010.txt"

# Step 1: read in data from the .txt file
# data is a numpy array of shape (190, 2), each row is a datapoint
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Step 4: construct model to predict Y (life expectancy from birth rate)
Y_predicted = w * X + b 

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
 
with tf.Session() as sess:
	# Step 7: initialize the necessary variables, in this case, w and b
	sess.run(tf.global_variables_initializer()) 
	
	# Step 8: train the model
	for i in range(100): # run 100 epochs
		for x, y in data:
			# Session runs train_op to minimize loss
			sess.run(optimizer, feed_dict={X: x, Y:y}) 
	
	# Step 9: output the values of w and b
	w_out, b_out = sess.run([w, b])
```

## tensorflow 控制流
观察上面线性回归所使用的loss function，是个简单的二次函数  
分析离群点，假设有一个离样本较远的离群点，那么这个离群点造成的loss fuction上的损失较大，会大大影响整个模型的建模。

**使用[huber loss](https://en.wikipedia.org/wiki/Huber_loss)代替原来简单的loss fuction**  
其定义如下所示    

$$
L_\delta(y,f(x))=\left\{
\begin{array}{ll}
\frac12(y-f(x))^2,&\textrm{for }|y-f(x)|\leq\delta\\
\delta\cdot(|y-f(x)|-\delta/2),& \textrm{otherwise.}
\end{array}
\right.
$$

Huber loss给离群点设置了相对更小的权重,因而提升了拟合的效果。

一个显然的事实是由于tensorflow 定义和执行的分离，我们不能用python的条件分支语句来控制optimizer使用哪一种loss function,tensor flow提供了分支控制的方法

Ops | Methods
:-|:-
Control Flow Ops | tf.count_up_to, tf.cond, tf.case, tf.while_loop, tf.group ...
Comparison Ops | tf.equal, tf.not_equal, tf.less, tf.greater, tf.where, ...
Logical Ops | tf.logical_and, tf.logical_not, tf.logical_or, tf.logical_xor
Debugging Ops | tf.is_finite, tf.is_inf, tf.is_nan, tf.Assert, tf.Print, ...

huber_loss：
```python
def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)
```
## tensorflow 输入
### placeholder & feed_dict
note1跳过了对tensorflow基本输入方式的叙述。实际上由于graph在定义的时候不需要考虑实际输入数据的特性。一般创建输入变量的时候实际上是为要输入的变量预留位置，使用tf.placeholder定义,如下是一个使用的例子
```python
a = tf.placeholder(dtype, shape=None, name=None)
...
with tf.Session() as sess:
   sess.run(something, feed_dict = {a:[1,2,3]})
```
shape参数制定了传入的tensor的结构，shape为None意味着任意结构的tensor都能被接收（可能潜在地会引入bug）

### tf.data
placeholder让数据的处理和带入图中运算分开，在tensorflow框架之外完成（完全可以用numpy等工具处理），不过这样带来的不好的地方之一在于，数据处理被放在了python的单一线程当中，会让数据处理较慢。（大量数据要从外部一个个装载到place_holder处）  

如上述代码当中看起来就不优雅的一段：
```python
	for i in range(100): # run 100 epochs
		for x, y in data:
			# Session runs train_op to minimize loss
			sess.run(optimizer, feed_dict={X: x, Y:y}) 
```
将数据分100次载入place_holder当中实际上较大的拖慢了程序的速度。还需要考虑的是在并行计算的时候载入feed_dict可能阻碍了其它操作的执行。

tensorflow提供的解决方案是将数据存储在tf.data.Dataset object当中
```python
tf.data.Dataset.from_tensor_slices((features, labels))
# can use numpy arrays as features and labels 
```

之后使用迭代器来访问dataset当中的每一个数据
```python
# we use make_initializable_iterator for multiple epochs
iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next() 
···
for i in range(100): 
        # reset where iterator point to
        sess.run(iterator.initializer)
        total_loss = 0
        try:
            while True:
                sess.run([optimizer]) 
        except tf.errors.OutOfRangeError:
            pass
```

dataset 也支持许多原生的对数据集的操作来改变数据集或是生成新的数据集
```python
dataset = dataset.shuffle(1000)
dataset = dataset.repeat(100)
dataset = dataset.batch(128)
dataset = dataset.map(lambda x: tf.one_hot(x, 10)) 
# convert each element of dataset to one_hot vector
```
## Optimizers
默认情况下optimizer在每一轮迭代的过程中自动更新loss function所依赖的所有变量，若有不想更新的变量，在定义的时候加上参数trainable=False

(to do: add contont about more detailed control of model trains using tf.gradient)

## Refs
[03_Lecture note_Linear and Logistic Regression](https://docs.google.com/document/d/1kMGs68rIHWHifBiqlU3j_2ZkrNj9RquGTe8tJ7eR1sE/edit#)  
[Huber Loss](https://en.wikipedia.org/wiki/Huber_loss)
