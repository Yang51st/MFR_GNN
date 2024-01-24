import tensorflow as tf
import math
from keras.layers import Layer
from keras.layers import Layer
from CustomLayers import GraphCNNGlobal, GCNLayer


class ES_GNN(tf.keras.Model):
    def __init__(self):
        super(ES_GNN, self).__init__()
        #Here there be no input variables, just declare number of layers to __init__ them as well.
        self.GCN1=GCNLayer(filters=10,name="GCN1")
        self.bn_1 = tf.keras.layers.BatchNormalization(name="BN1")
        self.dp_1 = tf.keras.layers.Dropout(rate=0.3, name="DP1")
        self.GCN2=GCNLayer(filters=10,name="GCN2")
        self.bn_2 = tf.keras.layers.BatchNormalization(name="BN2")
        self.dp_2 = tf.keras.layers.Dropout(rate=0.3, name="DP2")
        self.GCN3=GCNLayer(filters=10,name="GCN3")
        self.bn_3 = tf.keras.layers.BatchNormalization(name="BN3")
        self.dp_3 = tf.keras.layers.Dropout(rate=0.3, name="DP3")
        self.GCN4=GCNLayer(filters=10,name="GCN4")
        self.bn_4 = tf.keras.layers.BatchNormalization(name="BN4")
        self.dp_4 = tf.keras.layers.Dropout(rate=0.3, name="DP4")
        self.bn_5 = tf.keras.layers.BatchNormalization(name="BN5")
        self.mlp1 = tf.keras.layers.Dense(units=10,name="MLP1")
        self.mlp2 = tf.keras.layers.Dense(units=10,name="MLP2")
    def call(self, inputs, training=False):
        N, ADJ = inputs
        x1=self.GCN1([N,ADJ])
        x1=self.bn_1(x1,training=training)
        x1=self.dp_1(x1,training=training)


        x2=self.GCN2([x1,ADJ])
        x2=self.bn_2(x2,training=training)
        x2=self.dp_2(x2,training=training)


        x3=self.GCN3([x2,ADJ])
        x3=self.bn_3(x3,training=training)
        x3=self.dp_3(x3,training=training)


        x4=self.GCN4([x3,ADJ])
        x4=self.bn_4(x4,training=training)
        x4=self.dp_4(x4,training=training)


        x1=x1+x2+x3+x4
        xf=self.bn_5(x1,training=training)


        h1=self.mlp1(xf)
        h2=self.mlp2(xf)
        otpt=tf.matmul(h1,h2,transpose_b=True)
        return otpt