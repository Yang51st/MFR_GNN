import json
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import tensorflow as tf
import math
from keras.layers import Layer
from CustomLayers import GraphCNNGlobal, GCNLayer
from CustomGNN import ES_GNN


with open("data1/aag/graphs.json") as filename:
    da = json.load(filename)
    #data is a list and will have length corresponding to number of step files generated
    #data[step_number][0] is name of step file, data[step_number][1] contains the below in dictionary form:
    """
            'graph': graph,
            'graph_face_attr': graph_face_attr,
            'graph_face_grid': graph_face_grid,
            'graph_edge_attr': graph_edge_attr,
            'graph_edge_grid': graph_edge_grid,
    """
#graph at edges is stored in a strange fashion, where first list is node numbers, and the correspondingly indexed
#value in the second list is a node that the first-list node is connected to
shape_data_list=[]
adjacency_matrix_list=[]
instance_matrix_list=[]


model=ES_GNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)


def AAGLoss(y_true,y_pred):
    comp=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM,from_logits=True)(y_true=y_true,y_pred=y_pred)
    return comp/(y_true.shape[1])**2


for i in range(22):
    shape_data=da[i]
    opened_file_name=shape_data[0]
    adjacency_data=shape_data[1]["graph"]["edges"]
    number_of_nodes=shape_data[1]["graph"]["num_nodes"]
    graph=nx.DiGraph()
    adjacency_matrix=[[0 for j in range(number_of_nodes)] for k in range(number_of_nodes)]
    for a_index in range(len(adjacency_data[0])):
        graph.add_edge(adjacency_data[0][a_index],adjacency_data[1][a_index])
        adjacency_matrix[adjacency_data[0][a_index]][adjacency_data[1][a_index]]=1
    with open("data1/labels/"+opened_file_name+".json") as fn:
        dt=json.load(fn)
    instance_matrix=dt[0][1]["inst"]
    shape_data_list.append(shape_data[1]["graph_face_attr"])
    adjacency_matrix_list.append(adjacency_matrix)
    instance_matrix_list.append(instance_matrix)
    N=tf.Variable(np.array(shape_data_list[-1]),dtype="float32")
    A=tf.Variable(np.array(adjacency_matrix_list[-1]),dtype="float32")
    L=tf.Variable(np.array(instance_matrix_list[-1]),dtype="float32")


    if i<21:
        with tf.GradientTape() as tape:
            logits=model([N,A],training=True)
            loss_value=AAGLoss(L,logits)
            grads=tape.gradient(loss_value,model.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    else:
        print(model([N,A],training=True))
        print(model.summary())