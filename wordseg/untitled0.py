# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 13:10:39 2018

@author: zhangbq
"""
import tensorflow as tf
import numpy as np
tf.reset_default_graph()
# 创建输入数据
X = np.random.randn(2, 4, 5)# 批次 、序列长度、样本维度
print(X.shape)
# 第二个样本长度为3
X[1,2:] = 0
seq_lengths = [4, 2]

Gstacked_rnn = []
Gstacked_bw_rnn = []
for i in range(10):
    Gstacked_rnn.append(tf.contrib.rnn.LSTMCell(120))
    Gstacked_bw_rnn.append(tf.contrib.rnn.LSTMCell(120))

#建立前向和后向的三层RNN
Gmcell = tf.contrib.rnn.MultiRNNCell(Gstacked_rnn)
Gmcell_bw = tf.contrib.rnn.MultiRNNCell(Gstacked_bw_rnn)

sGbioutputs, sGoutput_state_fw, sGoutput_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([Gmcell],[Gmcell_bw], X,sequence_length=seq_lengths,                                           dtype=tf.float64)


Gbioutputs, Goutput_state_fw = tf.nn.bidirectional_dynamic_rnn(Gmcell,Gmcell_bw, X,sequence_length=seq_lengths,dtype=tf.float64)
#建立一个会话
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

sgbresult,sgstate_fw,sgstate_bw=sess.run([sGbioutputs,sGoutput_state_fw,sGoutput_state_bw])
print("全序列：\n", sgbresult[0])
print("短序列：\n", sgbresult[1])
print('Gru的状态：',len(sgstate_fw[0]),'\n',sgstate_fw[0][0],'\n',sgstate_fw[0][1],'\n',sgstate_fw[0][2])
print('Gru的状态：',len(sgstate_bw[0]),'\n',sgstate_bw[0][0],'\n',sgstate_bw[0][1],'\n',sgstate_bw[0][2])