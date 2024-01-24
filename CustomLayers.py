import tensorflow as tf
import math
from keras.layers import Layer


class GraphCNNGlobal(object):
    BN_DECAY = 0.999
    GRAPHCNN_INIT_FACTOR = 1.
    GRAPHCNN_I_FACTOR = 1.0


class GCNLayer(Layer):
    def __init__(self,filters,name="GCN",**kwargs):
        super(GCNLayer,self).__init__(name=name,**kwargs)
        self.num_filters=filters
        self.weight_decay=0.0005
        self.W=None
        self.W_I=None
        self.b=None
    def build(self,input_shape):
        # num_features = (c)
        # num_nodes = (n)
        # num_filters = (j)
        # W_dim = (c x j)
        # W_I_dim = (c x j)
        # b_dim = (n x j)
        V_shape, _ = input_shape
        num_features = V_shape[1]
        W_dim = [num_features, self.num_filters]
        W_I_dim = [num_features, self.num_filters]
        b_dim = [self.num_filters]
        W_stddev = math.sqrt(1.0 / num_features * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)
        W_I_stddev = math.sqrt(
            GraphCNNGlobal.GRAPHCNN_I_FACTOR / num_features * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)
        self.W = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="W")
        self.W_I = self.add_weight(
            shape=W_I_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_I_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="W_I")
        self.b = self.add_weight(
            shape=b_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name="bias")
    def call(self, input, training=None):
        V, A = input #yields (n x c) and (n x n)
        n = tf.matmul(A, V) #yields (n x c)
        output = tf.matmul(n, self.W) + tf.matmul(V, self.W_I) + self.b
        #above performs (n x c)(c x j) + (n x c)(c x j) + (n x j)
        #yields (n x j)
        return output