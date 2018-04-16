---
title: TensorFlow 学习笔记3 Manage EXperiments #3
date: 2018-04-11 20:59:00
tags: [tensorflow]
---
# Tensorflow 学习笔记 #3

**keywords** :  model base, variable sharing, model sharing

## 构建tensorflow模型的一般步骤
Phase1: **assenmble graph** 
1. Import Data
2. Define the weigths
3. Define the inferece model
4. Define loss function
5. Define optimizer


Phase2: **execute the computation**
1. initialize all model variables for the first time
2. Initialize iterator / feed training data
3. Excecute the inference model on the training data
4. compute cost
5. Adjust model parameters to minimize cost 

利用python面向对象的性质为自己的模型简历一个类：
```python
class Model:
    def __init__(self, params):
        pass

    def _import_data(self):
        """ Step 1: import data """
        pass

    def _create_embedding(self):
        """ Step 2: in word2vec, it's actually the weights that we care about """
        pass

    def _create_loss(self):
        """ Step 3 + 4: define the inference + the loss function """
        pass

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        pass
```


## Variable Sharing
### Name Scope
    为了能够在tensor board上较为清晰的辨识出节点之间的关系，引入name_scope可将其分组
```python
with tf.name_scope(name_of_that_scope):
    # declare op1
    # declare op2
    # declare op3
```

### Variable Scope
    使用Varibale scope来做到变量共享，在variable_scope中使用get_variable方法来获取之前创建的变量而不是新的一个变量
```python
with tf.variable_scope("xxx") as scope:
    # a = tf.get_variable("x",.)..
```

## tensorflow 实验管理
### 使用checkpoint保存训练中间结果
对于一个需要较长时间训练的模型来说，断点恢复能力是十分必要的。  
tensorflow也设置了相应的机制，即为checkpoint，可以用来周期性的保存当前模型的参数等数据。  
实现这一点的是tf.train.Sacer() 类，它会将图的变量保存在二进制文件当中。 
```python
tf.train.Saver.save(
    sess,
    save_path,
    global_step=None,
    latest_filename=None,
    meta_graph_suffix='meta',
    write_meta_graph=True,
    write_state=True
)
```
常用的保存checkpoint的方法如下所示：
```python
# define model

# create a saver object
saver = tf.train.Saver()

# launch a session to execute the computation
with tf.Session() as sess:
    # actual training loop
    for step in range(training_steps): 
	sess.run([optimizer])
	if (step + 1) % 1000 == 0:
	   saver.save(sess, 'checkpoint_directory/model_name', global_step=global_step)
``` 
这里的global_step是一个用来记录图训练了多少步的变量，创建其的时候需要设置其不能被训练。
（optimizer默认训练所有变量）
```python
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
```
optimizer一般也接收一个global_step变量的输入，每一次优化更新之后会将global_step的值自增
```python
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)
```

tensorflow还支持在一个文件夹里面找checkpoint,如果有合法的，恢复checkpoint,否则继续执行
```python
ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
if ckpt and ckpt.model_checkpoint_path:
     saver.restore(sess, ckpt.model_checkpoint_path)
```

### 使用tf.summary可视化训练数据

```python
def _create_summaries(self):
     with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)            
            tf.summary.histogram("histogram loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()
```
summary_op和其它operation一样，需要在sess中运行得到结果。  
得到结果之后使用add_summary把结果写入writer当中，就可以在tensorbord中看到add_summary的图的各种曲线啦  
```python
writer.add_summary(summary, global_step=step)
```
summary的用法参考[这里](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)

### control randomization

### Auto diff
