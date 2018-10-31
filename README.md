VGG网络解读
==========


# 一. VGG简介

> VGGNet提出目的即为了探究在大规模图像识别任务中，卷积网络深度对模型精确度有何影响。这个网络的结构用的都是特别小的3x3的卷积模版（stride：1，padding:1），以及5个2x2的池化层（stride：2），将卷积层的深度提升到了16-19层，并在当年ImageNet挑战中再定位和分类问题上取得地第一第二的好成绩。

# 二. VGG结构详解

> VGGNet的所有卷积层有相同的配置，即卷积核大小为3x3，步长为1，填充为1；共有5个最大池化层，大小都为2x2，步长为2；共有三个全连接层，前两层都有4096通道，第三层共1000类及代表1000个标签类别；最后一层为softmax层；所有隐藏层后都带有ReLU非线性激活函数；经过实验证明，AlexNet中提出的局部响应归一化（LRN）对性能提升并没有什么帮助，而且还浪费了内存的计算的损耗。VGGNet的网络结构如图所示

![image](https://github.com/ShaoQiBNU/VGG/blob/master/images/1.jpg)


# 三. 代码

> 参考https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py. ，利用VGG16实现MNIST识别，代码如下：

```python
########## load packages ##########
import tensorflow as tf

##################### load data ##########################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_sets", one_hot=True)

########## set net hyperparameters ##########
learning_rate = 0.0001

epochs = 20
batch_size_train = 128
batch_size_test = 100

display_step = 20

########## set net parameters ##########
#### img shape:28*28 ####
n_input = 784

#### 0-9 digits ####
n_classes = 10

########## placeholder ##########
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


##################### build net model ##########################

######### VGG 16 layer ##########
def VGG16(x, n_classes):
    ####### reshape input picture ########
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    ####### first conv ########
    #### conv 1_1 ####
    conv1 = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='SAME')

    #### relu ####
    conv1 = tf.nn.relu(conv1)

    #### conv 1_2 ####
    conv1 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, strides=1, padding='SAME')

    #### relu ####
    conv1 = tf.nn.relu(conv1)

    ####### max pool ########
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### second conv ########
    #### conv 2_1 ####
    conv2 = tf.layers.conv2d(pool1, filters=128, kernel_size=3, strides=1, padding='SAME')

    #### relu ####
    conv2 = tf.nn.relu(conv2)

    #### conv 2_2 ####
    conv2 = tf.layers.conv2d(conv2, filters=128, kernel_size=3, strides=1, padding='SAME')

    #### relu ####
    conv2 = tf.nn.relu(conv2)

    ####### max pool ########
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### third conv ########
    #### conv 3_1 ####
    conv3 = tf.layers.conv2d(pool2, filters=256, kernel_size=3, strides=1, padding='SAME')

    #### relu ####
    conv3 = tf.nn.relu(conv3)

    #### conv 3_2 ####
    conv3 = tf.layers.conv2d(conv3, filters=256, kernel_size=3, strides=1, padding='SAME')

    #### relu ####
    conv3 = tf.nn.relu(conv3)

    #### conv 3_3 ####
    conv3 = tf.layers.conv2d(conv3, filters=256, kernel_size=1, strides=1, padding='SAME')

    #### relu ####
    conv3 = tf.nn.relu(conv3)

    ####### max pool ########
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### fourth conv ########
    #### conv 4_1 ####
    conv4 = tf.layers.conv2d(pool3, filters=512, kernel_size=3, strides=1, padding='SAME')

    #### relu ####
    conv4 = tf.nn.relu(conv4)

    #### conv 4_2 ####
    conv4 = tf.layers.conv2d(conv4, filters=512, kernel_size=3, strides=1, padding='SAME')

    #### relu ####
    conv4 = tf.nn.relu(conv4)

    #### conv 4_3 ####
    conv4 = tf.layers.conv2d(conv4, filters=512, kernel_size=1, strides=1, padding='SAME')

    #### relu ####
    conv4 = tf.nn.relu(conv4)

    ####### max pool ########
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### fifth conv ########
    #### conv 5_1 ####
    conv5 = tf.layers.conv2d(pool4, filters=512, kernel_size=3, strides=1, padding='SAME')

    #### relu ####
    conv5 = tf.nn.relu(conv5)

    #### conv 5_2 ####
    conv5 = tf.layers.conv2d(conv5, filters=512, kernel_size=3, strides=1, padding='SAME')

    #### relu ####
    conv5 = tf.nn.relu(conv5)

    #### conv 5_3 ####
    conv5 = tf.layers.conv2d(conv5, filters=512, kernel_size=1, strides=1, padding='SAME')

    #### relu ####
    conv5 = tf.nn.relu(conv5)

    ####### max pool ########
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### flatten 影像展平 ########
    flatten = tf.reshape(pool5, (-1, 1 * 1 * 512))

    ####### fc1 ########
    fc1 = tf.layers.dense(flatten, 4096)

    #### relu ####
    fc1 = tf.nn.relu(fc1)

    ####### fc2 ########
    fc2 = tf.layers.dense(fc1, 4096)

    #### relu ####
    fc2 = tf.nn.relu(fc2)

    ####### out 10类 可根据数据集进行调整########
    out = tf.layers.dense(fc2, n_classes)

    return out


########## define model, loss and optimizer ##########

#### model pred 影像判断结果 ####
pred = VGG16(x, n_classes)

#### loss 损失计算 ####
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

##################### train and evaluate model ##########################

########## initialize variables ##########
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    #### epoch 世代循环 ####
    for epoch in range(epochs + 1):

        #### iteration ####
        for _ in range(mnist.train.num_examples // batch_size_train):

            step += 1

            ##### get x,y #####
            batch_x, batch_y = mnist.train.next_batch(batch_size_train)

            ##### optimizer ####
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            ##### show loss and acc #####
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples // batch_size_test):
        batch_x, batch_y = mnist.test.next_batch(batch_size_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))

```
