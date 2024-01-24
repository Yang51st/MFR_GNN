from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer
import networkx as nx
from occwl.solid import Solid
from occwl.face import Face
from occwl.graph import face_adjacency
import matplotlib.pyplot as plt
import numpy as np
from random import randint


step_reader = STEPControl_Reader()
step_reader.ReadFile("data1/steps/20240119_144510_11.step")
step_reader.TransferRoot()
my_shape = step_reader.Shape()
shape_topology=TopologyExplorer(my_shape)
faces=list(shape_topology.faces())
edges=list(shape_topology.edges())
wires=list(shape_topology.wires())
num_faces=len(faces)
face_adjacency_graph=nx.DiGraph()


graph=face_adjacency(Solid(my_shape))
"""
face_adj = np.zeros((num_faces, num_faces))
for face_idx in graph.nodes:
    for neighbor in graph.neighbors(face_idx):
        face_adj[face_idx][neighbor] = 1
"""


# show FAG
plt.figure(figsize=(8, 8))
plt.axis('off')
#fag = nx.from_numpy_array(face_adj)
cols=[(0.5,0.5,0.5) for c in range(len(graph.nodes()))]
print(graph.nodes.data())


def face_node_index(face,graph):
    count=0
    for node in list(graph.nodes.data()):
        node_face=node[1]
        target=node_face['face']
        if Face(face)==target:
            return count
        count+=1
    else:
        return None


face_node_index(faces[5],graph)


# use same position layout to align fag with instance adjacency
nx.draw_networkx(graph, with_labels=True, node_size=350, node_color=cols)
plt.show()