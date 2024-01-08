# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:12:53 2024

@author: Tom
"""

import tkinter as tk
from matplotlib.figure import Figure 
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class CubeGui():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Rubik's Cube.jpg")
        self.height = 900
        self.width = 1600
        self.window.geometry(f"{self.width}x{self.height}")
        self.font_size = 13 
        
        self.plt = plt
        self.fig = figure(figsize=(8, 8), dpi=80)
        self.ax = self.plt.subplot(projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.window)
    
    def draw_window(self):
        self.window.mainloop() 
    def draw3dtk(self,cube):
        '''
        Plots 3 planes of 1x1 in 3d using the information in the self.cube attribute
        '''
        self.ax.clear()
          
        self.canvas.draw()
        
        top_cols = cube.clock_rot([row[3:6:] for row in cube.cube[:3]])
        front_cols = [row[::-1] for row in cube.clock_rot(cube.clock_rot([row[3:6] for row in cube.cube[3:6]]))]
        right_cols = [row[::-1] for row in cube.clock_rot(cube.clock_rot([row[6:] for row in cube.cube[3:6]]))]
        left_cols = cube.clock_rot(cube.clock_rot([row[:3] for row in cube.cube[3:6]]))
        bot_cols = [row[::-1] for row in cube.clock_rot([row[3:6] for row in cube.cube[6:9]])]#[row[3:6] for row in cube.cube[6:9]]
        back_cols = [row[3:6] for row in cube.cube[9:]]
        for i in range(3):
            for j in range(3):
                '''
                Top Face
                '''
                x,y = np.meshgrid(range(i,i+2),range(j,j+2))
                z = np.ones(shape = (2,2))*3
                self.ax.plot_surface(x,y,z ,color = np.array(top_cols[i][j])/255)                
                '''
                Front Face
                '''
                z2,x2 = np.meshgrid(range(i,i+2),range(j,j+2))
                y2 = z2*0
                self.ax.plot_surface(x2,y2,z2, color = np.array(front_cols[i][j])/255)                
                '''
                Right Face
                '''
                z3,y3 = np.meshgrid(range(i,i+2),range(j,j+2))
                x3 = np.ones(shape = (2,2))*3
                self.ax.plot_surface(x3,y3,z3, color = np.array(right_cols[i][j])/255)
                '''
                Left Face
                '''
                z4,y4 = np.meshgrid(range(i,i+2),range(j,j+2))
                x4 = np.ones(shape = (2,2))*0
                self.ax.plot_surface(x4,y4,z4, color = np.array(left_cols[i][j])/255)
                '''
                Bot Face
                '''
                x5,y5 = np.meshgrid(range(i,i+2),range(j,j+2))
                z5 = np.ones(shape = (2,2))*0
                self.ax.plot_surface(x5,y5,z5 ,color = np.array(bot_cols[i][j])/255)
                '''
                Back Face
                '''
                z6,x6 = np.meshgrid(range(i,i+2),range(j,j+2))
                y6 = np.ones(shape = (2,2))*3
                self.ax.plot_surface(x6,y6,z6, color = np.array(back_cols[i][j])/255)  
        self.plt.axis('off')
        
        self.canvas.get_tk_widget().pack() 
  
        # creating the Matplotlib toolbar 
        toolbar = NavigationToolbar2Tk(self.canvas, self.window) 
        toolbar.update() 
      
        # placing the toolbar on the Tkinter window 
        self.canvas.get_tk_widget().pack() 
        #self.plt.show()
    def update_canvas(self):
        self.canvas.figure.canvas = None
    def setup_gui(self):
        self.draw_window()
        
if __name__ == '__main__':
    from cube_class import *
    Cube = RubiksCube()
    Cube.scramble_cube(10)
   
    gui = CubeGui()
    
    gui.draw3dtk(cube = Cube)
    Cube.draw2d()
    
    gui.setup_gui()