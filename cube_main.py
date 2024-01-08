# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:07:43 2024

@author: Tom
"""
from cube_class import *
from cube_gui import *

if __name__ == '__main__':
    Cube = RubiksCube()
    Cube.scramble_cube(10)
    gui = CubeGui()
    gui.draw3dtk(Cube)
    gui.setup_gui()    