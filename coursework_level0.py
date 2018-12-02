#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas 
import numpy as np
import matplotlib.pylab as pl
import tensorflow as tf

from PIL import Image
import random

import math

data = pandas.read_csv('./annotations.csv',header = None, 
                       names = ['FileName','StartPoint_X','StartPoint_Y','EndPoint_X','EndPoint_Y'])


# In[2]:


level_0 = data[data['FileName'].str.contains('0_level')]

level_0


# In[3]:


#get FileName array

array_FileName = level_0.FileName.values.reshape(-1,1)

#get StartPoint_X, StartPoint_Y,EndPoint_X,EndPoint_Y array

array_Point = level_0.drop(columns=['FileName'], inplace=False).values


array_Point.shape



# In[4]:


# image processing

holdout = random.sample(range(0,8471),8000)

array_imageName = array_FileName[holdout]


size = 28,28


# grey image
img = Image.open('./level0/'+array_imageName[0][0]+'.png').convert('L')

img.thumbnail(size)

image_array = np.array(img).reshape(-1,1)



for i in range(1,len(array_imageName)):
    img = Image.open('./level0/'+array_imageName[i][0]+'.png').convert('L')
    img.thumbnail(size)
    image_array = np.column_stack((image_array,np.array(img).reshape(-1,1)))
    
test_train_input = image_array.T
test_train_output = array_Point[holdout]




#print(test_train_input)


#image_array = image_array.reshape(-1,1)

print(test_train_input.shape)
print(test_train_output.shape)

#np.savetxt('new.csv', image_array, delimiter = ' ')


# In[5]:


test_train_input.astype(np.float32)

#test_train_output.dtype


# In[6]:


test_train_output.astype(np.float32)


# In[7]:


#design network

tf.reset_default_graph() # 这个可以不用细究，是为了防止重复定义报错

# 给X、Y定义placeholder，要指定数据类型、形状：
X = tf.placeholder(dtype=tf.float32,shape=[None,784],name='Input')
Y = tf.placeholder(dtype=tf.float32,shape=[None,4],name='Output')

# 定义各个参数：
W1 = tf.get_variable('W1',[784,128],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable('b1',[128],initializer=tf.zeros_initializer())
W2 = tf.get_variable('W2',[128,64],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable('b2',[64],initializer=tf.zeros_initializer())
W3 = tf.get_variable('W3',[64,4],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable('b3',[4],initializer=tf.zeros_initializer())



# In[29]:




A1 = tf.nn.relu(tf.matmul(X,W1)+b1,name='A1')
A2 = tf.nn.relu(tf.matmul(A1,W2)+b2,name='A2')
Z3 = tf.matmul(A2,W3)+b3







# In[30]:


#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
cost = tf.reduce_mean(tf.reduce_sum(tf.square(Y - Z3),reduction_indices=[1]))



# In[32]:


trainer = tf.train.AdamOptimizer().minimize(cost)
#learning_rate = tf.placeholder(tf.float32, shape=[])

#trainer = tf.train.GradientDescentOptimizer (0.5).minimize (cost)
#init = tf.global_variables_initializer ()
#correct_prediction = tf.equal (tf.argmax (Y, 1), tf.argmax (Z3, 1))
#accuracy = tf.reduce_mean (tf.cast (correct_prediction, 'float'))


# In[33]:


with tf.Session() as sess:
    # 首先给所有的变量都初始化（不用管什么意思，反正是一句必须的话）：
    sess.run(tf.global_variables_initializer())

    # 定义一个costs列表，来装迭代过程中的cost，从而好画图分析模型训练进展
    costs = []

    # 指定迭代次数：
    for it in range(8000):
        # 这里我们可以使用mnist自带的一个函数train.next_batch，可以方便地取出一个个地小数据集，从而可以加快我们的训练：
        #X_batch,Y_batch = mnist.train.next_batch(batch_size=64)

        # 我们最终需要的是trainer跑起来，并获得cost，所以我们run trainer和cost，同时要把X、Y给feed进去：
        _,batch_cost = sess.run([trainer,cost],feed_dict={X:test_train_input,Y:test_train_output})
        costs.append(batch_cost)

        # 每100个迭代就打印一次cost：
        if it%100 == 0:
            print('iteration%d ,batch_cost: '%it,batch_cost)

    # 训练完成，我们来分别看看来训练集和测试集上的准确率：
    predictions = tf.equal(tf.argmax(tf.transpose(Z3)),tf.argmax(tf.transpose(Y)))
    accuracy = tf.reduce_mean(tf.cast(predictions,'float'))
    print("Training set accuracy: ",sess.run(accuracy,feed_dict={X:test_train_input,Y:test_train_output}))


# In[34]:


sess.close()


# In[35]:


# In[ ]:





# In[ ]:




