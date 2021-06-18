
import numpy as np
import os
import time
import keras
import pandas
import sys
#import tensorflow.compat.v1 as tf
import tensorflow as tf


#if you use tensorflow 1 you need to enable eager execution

#tf.enable_eager_execution()
#tf.disable_v2_behavior()


#print(tf.__version__)

def Conv1D(inputs, oc, k=3, s=1, p='SAME', d=1, with_bn=False, act='relu', use_bias=False, name='Conv2D'):

    #outputs = inputs
    
    #outputs = tf.compat.v1.layers.conv1d(
    #    inputs=outputs, filters=oc, kernel_size=k, strides=s, dilation_rate=d, padding=p, use_bias=use_bias, name=name)
    
   
    outputs2 = tf.keras.layers.Conv1D(filters=oc, kernel_size=k,activation=act,strides=s, dilation_rate=d,padding=p,use_bias=use_bias)(inputs)
   


    
    if with_bn:
        outputs2 = tf.layers.batch_normalization(outputs2, name=name + '_bn')
    if act:
        if act == 'relu':
            outputs2 = tf.nn.relu(outputs2, name=name + '_relu')
    return outputs2

class UGP(object):

    def __init__(self,data):
        self.key_cnn_layers = 5
        self.query_cnn_layers = 5
        self.graph_layers = 'every'
        self.graph_bias =  0.1
        self.graph_scope = 'Graph_Predictor'
        self.sentence_length=5
        self.data=data


    def Calculate_Graph(self, key_feature, query_feature, idx):

            W_k = tf.Variable(tf.ones([self.sentence_length, 16]), name='W_key_{}'.format(idx))
            W_q = tf.Variable(tf.ones([self.sentence_length, 16]), name='W_query_{}'.format(idx))
            key_feature = tf.einsum('ijk,jk -> ijk', key_feature, W_k, name = 'key_feature_matmul_{}'.format(idx))
            query_feature = tf.einsum('ijk,jk -> ijk', query_feature, W_q, name = 'query_feature_matmul_{}'.format(idx))
            query_feature = tf.transpose(query_feature, [0,2,1])

            graph = tf.add(
                tf.matmul(key_feature, query_feature),
                self.graph_bias, name='add_bias{}'.format(idx))

            graph = tf.nn.relu(graph)
            graph = tf.square(graph)
            sum_graph = tf.reduce_sum(graph, axis=2, keepdims=True)
            graph = tf.divide(graph, sum_graph)
            
            return graph


    def Graph_Predictor(self):

        with tf.compat.v1.variable_scope(self.graph_scope):

            assert self.graph_layers == 'every' if isinstance(self.graph_layers, str) else \
                len(self.graph_layers) > 0 if isinstance(self.graph_layers, list) else False, \
                'Graph layers must be \'every\' or a list of int numbers.'

            key_cnn_groups = []
            query_cnn_groups = []

            inputss = tf.convert_to_tensor(self.data,dtype=tf.float32)
            key_temp = inputss
            query_temp = inputss


            for i in range(self.key_cnn_layers):
                key_temp = Conv1D(
                    key_temp, 16, 3,
                    1, 'SAME', 1, False, 'relu', False, 'KeyCNN_1_{}'.format(i))

                key_temp = Conv1D(
                    key_temp, 16, 1,
                    1, 'SAME', 1, False, 'relu', False, 'KeyCNN_2_{}'.format(i))
                
                key_cnn_groups.append(key_temp)
                
                query_temp = Conv1D(
                    query_temp, 16, 3,
                    1, 'SAME', 1, False, 'relu', False, 'QueryCNN_1_{}'.format(i))
                
                query_temp = Conv1D(
                    query_temp, 16, 1,
                    1, 'SAME', 1, False, 'relu', False, 'QueryCNN_2_{}'.format(i))
                query_cnn_groups.append(query_temp)
            
            graphs = []
            if self.graph_layers == 'every':

                for i in range(len(key_cnn_groups)):
                    graphs.append(self.Calculate_Graph(key_cnn_groups[i], query_cnn_groups[i], i))

            else:
                for i in self.graph_layers:
                    if i >= 1:
                        graphs.append(self.Calculate_Graph(key_cnn_groups[i - 1], query_cnn_groups[i - 1], i))


            self.graphs = graphs
            return graphs
