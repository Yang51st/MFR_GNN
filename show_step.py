from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.WebGl import x3dom_renderer, threejs_renderer
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRepTools import breptools_Read
from OCC.Extend.TopologyUtils import TopologyExplorer


step_reader = STEPControl_Reader()
step_reader.ReadFile("data1/steps/20240119_144510_3.step")
step_reader.TransferRoot()
myshape = step_reader.Shape()


my_renderer=threejs_renderer.ThreejsRenderer()
faces=TopologyExplorer(myshape).faces()
for face in faces:
    my_renderer.DisplayShape(face,color=(0.65,0.65,0.7))
my_renderer.render()


"""
my_renderer = x3dom_renderer.X3DomRenderer()
my_renderer.DisplayShape(myshape,shininess=0.2,transparency=0.)
my_renderer.render()
"""