# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:48:04 2024

@author: Tom
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import figure

class RubiksCube():
    '''
    ToDo:
        Add bottom face rotations
        Add back face rotations
        
        
    '''
    def __init__(self):
        self.colour_dict = {
            'r':[255,0,0],
            'g':[0,255,0],
            'b':[0,0,255],
            'w':[255,255,255],
            'o':[255,87,51],
            'y':[255,234,0],
            'bl':[0,0,0]
            }
        self.moves = []
        self.generate_cube()
    
    def col_f(self,col_):
        return [[self.colour_dict[col_] for i in range(3)] for j in range(3)]
    def h_comb(self,l_a,l_b):
        if len(l_a) != len(l_b):
            print('make sure both matrices have the same i dimension ')
            return
        return [l_a[i]+l_b[i] for i in range(len(l_a))]

    def v_comb(self,l_a,l_b):
        if len(l_a[0]) != len(l_b[0]):
            print('make sure both matrices have the same j dimension ')
            return
        return l_a+l_b
    
    def generate_cube(self):
        r_f = self.col_f('r')
        g_f = self.col_f('g')        
        b_f = self.col_f('b')
        w_f = self.col_f('w')
        o_f = self.col_f('o')
        y_f = self.col_f('y')
        bl_f = self.col_f('bl')        
        r_1 = self.h_comb(self.h_comb(bl_f,r_f), bl_f)
        r_2 = self.h_comb(self.h_comb(b_f,w_f), g_f)
        r_3 = self.h_comb(self.h_comb(bl_f,o_f), bl_f)
        r_4 = self.h_comb(self.h_comb(bl_f,y_f), bl_f)        
        self.cube =  self.v_comb(self.v_comb(self.v_comb(r_1,r_2),r_3),r_4)
        
    def draw2d(self):
        figure(figsize=(8, 6), dpi=80)
        plt.imshow(self.cube)
        plt.show() 
    
    def draw3d(self):
        '''
        Plots 3 planes of 1x1 in 3d using the information in the self.cube attribute
        '''
        figure(figsize=(8, 6), dpi=80)
        ax = plt.subplot(projection='3d')
        top_cols = self.clock_rot([row[3:6:] for row in self.cube[:3]])
        front_cols = [row[::-1] for row in self.clock_rot(self.clock_rot([row[3:6] for row in self.cube[3:6]]))]
        right_cols = [row[::-1] for row in self.clock_rot(self.clock_rot([row[6:] for row in self.cube[3:6]]))]
        for i in range(3):
            for j in range(3):
                '''
                Top Face
                '''
                x,y = np.meshgrid(range(i,i+2),range(j,j+2))
                z = np.ones(shape = (2,2))*3
                ax.plot_surface(x,y,z ,color = np.array(top_cols[i][j])/255)                
                '''
                Front Face
                '''
                z2,x2 = np.meshgrid(range(i,i+2),range(j,j+2))
                y2 = z2*0
                ax.plot_surface(x2,y2,z2, color = np.array(front_cols[i][j])/255)                
                '''
                Right Face
                '''
                z3,y3 = np.meshgrid(range(i,i+2),range(j,j+2))
                x3 = np.ones(shape = (2,2))*3
                ax.plot_surface(x3,y3,z3, color = np.array(right_cols[i][j])/255)
        plt.axis('off')
        plt.show()
        
    def clock_rot(self,face_lst):
        ''' 
        Function for rotation a 3x3 matrix clockwise
        '''
        return list(zip(*face_lst[::-1]))
    
    def anticlock_rot(self, face_lst):
        ''' 
        Function for rotation a 3x3 matrix counter-clockwise
        Rotates it clockwise 3 times
        '''
        return self.clock_rot(self.clock_rot(self.clock_rot(face_lst)))
    
    def r_u(self, draw = False):
        '''
        Turns the right column of the cube upwards, from facing the middle face which connects the 'T' in the plots
        '''
        current_vals = [self.cube[i][5] for i in range(len(self.cube))]
        for i in range(12):
            self.cube[i][5] = current_vals[(i+3)%12]
        g_face = self.clock_rot([row[-3::] for row in self.cube[3:6]])
        for i in range(3):
            for j in range(3):
                self.cube[i+3][j+6] = g_face[i][j]
        self.moves.append('RU')
        if draw:
            self.draw2d()
            
    def r_d(self, draw = False):
        '''
        Turns the right column of the cube downwards, from facing the middle face which connects the 'T' in the plots
        '''
        current_vals = [self.cube[i][5] for i in range(len(self.cube))]
        for i in range(12):
            self.cube[i][5] = current_vals[(i-3)%12]
        r_face = self.anticlock_rot([row[-3::] for row in self.cube[3:6]])
        for i in range(3):
            for j in range(3):
                self.cube[i+3][j+6] = r_face[i][j] 
        self.moves.append('RD')
        if draw:
            self.draw2d()
    def l_u(self, draw = False):
        '''
        Turns the left column of the cube upwards, from facing the middle face which connects the 'T' in the plots
        '''
        current_vals = [self.cube[i][3] for i in range(len(self.cube))]
        for i in range(12):
            self.cube[i][3] = current_vals[(i+3)%12]
        l_face = self.anticlock_rot([row[:3:] for row in self.cube[3:6]])
        for i in range(3):
            for j in range(3):
                self.cube[i+3][j] = l_face[i][j]
        self.moves.append('LU')
        if draw:
            self.draw2d()
            
    def l_d(self, draw = False):
        '''
        Turns the left column of the cube downwards, from facing the middle face which connects the 'T' in the plots
        '''
        current_vals = [self.cube[i][3] for i in range(len(self.cube))]
        for i in range(12):
            self.cube[i][3] = current_vals[(i-3)%12]
        l_face = self.clock_rot([row[:3:] for row in self.cube[3:6]])
        for i in range(3):
            for j in range(3):
                self.cube[i+3][j] = l_face[i][j]
        self.moves.append('LD')
        if draw:
            self.draw2d()
            
    def t_l(self, draw = False):
        '''
        Turns the top face of the cube clockwise
        '''
        current_vals = self.cube[3] + self.cube[-1][3:6][::-1]
        for i in range(3):
            self.cube[3][i] = current_vals[i+3] 
            self.cube[3][i+3] = current_vals[i+6]
            self.cube[3][6+i] = current_vals[i+9]
            self.cube[-1][i+3] = current_vals[2-i]
        t_face = self.clock_rot([row[3:6] for row in self.cube[:3]])
        for i in range(3):
            for j in range(3):
                self.cube[i][j+3] = t_face[i][j]
        self.moves.append('TL')
        if draw:
            self.draw2d()
                
    def t_r(self, draw = False):
        '''
        Turns the top face of the cube counter-clockwise
        '''        
        current_vals = self.cube[3] + self.cube[-1][3:6][::-1]
        for i in range(3):
            self.cube[3][i] = current_vals[i+9]#current_vals[i+3] 
            self.cube[3][i+3] = current_vals[i]
            self.cube[3][6+i] = current_vals[i+3]
            self.cube[-1][i+3] = current_vals[8-i]
        t_face = self.anticlock_rot([row[3:6] for row in self.cube[:3]])
        for i in range(3):
            for j in range(3):
                self.cube[i][j+3] = t_face[i][j]
        self.moves.append('TR')
        if draw:
            self.draw2d()
                
    def f_c(self, draw = False):
        '''
        Function to rotate the front face clockwise
        Reverse of lists in new vals done prior to assigning to new locations,
        to save any indexing of a[i-3] etc
        '''
        new_right_vals = self.cube[2][3:6]                                     #from top face connected to front face
        new_bot_vals = [self.cube[i][6] for i in range(3,6)][::-1]             #from right vals connected to front face
        new_left_vals = self.cube[6][3:6]                                      #from bot face connected to front face
        new_top_vals = [self.cube[i][2] for i in range(3,6)][::-1]             #from left face connected to front face        
        new_mid = self.clock_rot( [row[3:6] for row in self.cube[3:6]])        
        for i in range(3):
            self.cube[i+3][6] = new_right_vals[i]
            self.cube[6][i+3] = new_bot_vals[i]            
            self.cube[i+3][2] = new_left_vals[i]
            self.cube[2][i+3] = new_top_vals[i]            
            for j in range(3):
                self.cube[i+3][j+3] = new_mid[i][j]
        self.moves.append('FR')
        if draw:
            self.draw2d()
        
    def f_ac(self, draw = False):
        '''
        Function to rotate the front face clockwise
        Reverse of lists in new vals done prior to assigning to new locations,
        to save any indexing of a[i-3] etc
        '''
        new_left_vals = self.cube[2][3:6][::-1]                                #from top face connected to front face
        new_top_vals = [self.cube[i][6] for i in range(3,6)]                   #from right vals connected to front face
        new_right_vals = self.cube[6][3:6][::-1]                               #from bot face connected to front face
        new_bot_vals = [self.cube[i][2] for i in range(3,6)]                   #from left face connected to front face        
        new_mid = self.anticlock_rot( [row[3:6] for row in self.cube[3:6]])        
        for i in range(3):
            self.cube[i+3][6] = new_right_vals[i]
            self.cube[6][i+3] = new_bot_vals[i]            
            self.cube[i+3][2] = new_left_vals[i]
            self.cube[2][i+3] = new_top_vals[i]            
            for j in range(3):
                self.cube[i+3][j+3] = new_mid[i][j]
        self.moves.append('FL')
        if draw:
            self.draw2d()
        
    def turn_cube_right(self, draw = False):
        '''
        new mappings for right rotation, mappings done in variables below based on commented below
        right -> back with a 180 degree rotation
        mid -> right
        left -> mid
        back -> left with a 180 degree rotation
        bottom -> bottom with a clockwise rotation
        top -> top with an anticlock rotation
        '''
        new_top = self.anticlock_rot([row[3:6] for row in self.cube[:3]])              #old top, i 0-3, j 3-6
        new_mid = [row[:3] for row in self.cube[3:6]]                                  #old left i 3-6, j 0-3
        new_right = [row[3:6] for row in self.cube[3:6]]                               #old mid i 3-6, j 3-6
        new_back = self.clock_rot(self.clock_rot([row[6:] for row in self.cube[3:6]])) #old right i 3-6, j 6-9
        new_bot =  self.clock_rot([row[3:6] for row in self.cube[6:9]])                #old bot i 6-9, j 3-6
        new_left =  self.clock_rot(self.clock_rot([row[3:6] for row in self.cube[9:]]))#old back i 9-12, j 3-6
        for i in range(3):
            for j in range(3):
                self.cube[i][j+3] = new_top[i][j]
                self.cube[i+3][j] = new_left[i][j]
                self.cube[i+3][j+3] = new_mid[i][j]
                self.cube[i+3][j+6] = new_right[i][j]
                self.cube[i+6][j+3] = new_bot[i][j]
                self.cube[i+9][j+3] = new_back[i][j]
        self.moves.append('ROT RIGHT')   
        if draw:
            self.draw2d()
            
    def turn_cube_left(self, draw = False):
        '''
        new mappings for left rotation, mappings done in variables below based on commented below
        right -> mid
        mid -> left
        left -> back 180 rot
        back -> right 180 rot
        bottom -> bottom anticlockwise rot
        top -> top clockwise rot
        '''
        new_top = self.clock_rot([row[3:6] for row in self.cube[:3]])                   #old top, i 0-3, j 3-6
        new_mid = [row[6:] for row in self.cube[3:6]]                              #old right i 3-6, j 6-9
        new_back = self.clock_rot(self.clock_rot([row[:3] for row in self.cube[3:6]]  ))     #old left i 3-6, j 0-3
        new_left = [row[3:6] for row in self.cube[3:6]]                            #old mid i 3-6, j 3-6
        new_bot =  self.anticlock_rot([row[3:6] for row in self.cube[6:9]])             #old bot i 6-9, j 3-6
        new_right =  self.clock_rot(self.clock_rot([row[3:6] for row in self.cube[9:]]))     #old back i 9-12, j 3-6        
        for i in range(3):
            for j in range(3):
                self.cube[i][j+3] = new_top[i][j]
                self.cube[i+3][j] = new_left[i][j]
                self.cube[i+3][j+3] = new_mid[i][j]
                self.cube[i+3][j+6] = new_right[i][j]
                
                self.cube[i+6][j+3] = new_bot[i][j]
                self.cube[i+9][j+3] = new_back[i][j]
        self.moves.append('ROT LEFT')  
        if draw:
            self.draw2d()
            
    def rotate_cube_anticlock(self, draw = False):
        '''
        new mappings for left rotation, mappings done in variables below based on commented below
        right -> mid
        mid -> left
        left -> back 180 rot
        back -> right 180 rot
        bottom -> bottom anticlockwise rot
        top -> top clockwise rot
        '''
        new_left = self.anticlock_rot([row[3:6] for row in self.cube[:3]])               #old top, i 0-3, j 3-6
        new_top= self.anticlock_rot([row[6:] for row in self.cube[3:6]])                 #old right i 3-6, j 6-9
        new_bot = self.anticlock_rot([row[:3] for row in self.cube[3:6]]  )    #old left i 3-6, j 0-3
        new_mid = self.anticlock_rot([row[3:6] for row in self.cube[3:6]])                            #old mid i 3-6, j 3-6
        new_right =  self.anticlock_rot([row[3:6] for row in self.cube[6:9]])             #old bot i 6-9, j 3-6
        new_back =  self.clock_rot([row[3:6] for row in self.cube[9:]])    #old back i 9-12, j 3-6        
        for i in range(3):
            for j in range(3):
                self.cube[i][j+3] = new_top[i][j]
                self.cube[i+3][j] = new_left[i][j]
                self.cube[i+3][j+3] = new_mid[i][j]
                self.cube[i+3][j+6] = new_right[i][j]
                
                self.cube[i+6][j+3] = new_bot[i][j]
                self.cube[i+9][j+3] = new_back[i][j]
        self.moves.append('ROT LEFT')  
        if draw:
            self.draw2d()
    def rotate_cube_clock(self, draw = False):
        '''
        new mappings for left rotation, mappings done in variables below based on commented below
        right -> mid
        mid -> left
        left -> back 180 rot
        back -> right 180 rot
        bottom -> bottom anticlockwise rot
        top -> top clockwise rot
        '''
        new_right = self.clock_rot([row[3:6] for row in self.cube[:3]])               #old top, i 0-3, j 3-6
        
        new_bot = self.clock_rot([row[6:] for row in self.cube[3:6]])                 #old right i 3-6, j 6-9
        new_top = self.clock_rot([row[:3] for row in self.cube[3:6]]  )    #old left i 3-6, j 0-3
        new_mid = self.clock_rot([row[3:6] for row in self.cube[3:6]])                            #old mid i 3-6, j 3-6
        new_left =  self.clock_rot([row[3:6] for row in self.cube[6:9]])             #old bot i 6-9, j 3-6
        new_back =  self.anticlock_rot([row[3:6] for row in self.cube[9:]])    #old back i 9-12, j 3-6        
        for i in range(3):
            for j in range(3):
                self.cube[i][j+3] = new_top[i][j]
                self.cube[i+3][j] = new_left[i][j]
                self.cube[i+3][j+3] = new_mid[i][j]
                self.cube[i+3][j+6] = new_right[i][j]
                
                self.cube[i+6][j+3] = new_bot[i][j]
                self.cube[i+9][j+3] = new_back[i][j]
        self.moves.append('ROT LEFT')  
        if draw:
            self.draw2d()
            
    def scramble_cube(self,num_moves, show_steps = False):
        '''
        Scrambles the cube by picking a random function 
        '''
        move_fn_dict = {
            0: self.r_u,
            1: self.r_d,
            2: self.l_u,
            3: self.l_d,
            4: self.t_l,
            5: self.t_r,
            6: self.f_ac,
            7: self.f_c
            }        
        for i in range(num_moves):
            my_num = random.randint(0,7)
            move_fn_dict[my_num]()
            if show_steps:
                plt.imshow(self.cube)
                plt.show()
                print(f'{move_fn_dict[my_num].__name__}\n\n')
                
    def right_corners(self,show_steps = False):
        '''
        Function to perform the RU, TL, RD, TR algorithm,
        Funcitonally swaps the top right and bottom right corners of the cube,
        looking at it from the front face
        '''
        self.r_u()
        if show_steps:
            self.draw2d()
        self.t_l()
        if show_steps:
            self.draw2d()
        self.r_d()
        if show_steps:
            self.draw2d()
        self.t_r()
        if show_steps:
            self.draw2d()
        
    def left_corners(self,show_steps = False):
        '''
        Function to perform the LU, TR, LD, TL algorithm,
        Funcitonally swaps the top left and bottom left corners of the cube,
        looking at it from the front face
        '''
        self.l_u()
        if show_steps:
            self.draw2d()
        self.t_r()
        if show_steps:
            self.draw2d()
        self.l_d()
        if show_steps:
            self.draw2d()
        self.t_l()
        if show_steps:
            self.draw2d()

    def topmid_2_midleft(self, show_steps = False):
        '''
        Places the top middle block to the left middle position on the cube
        Looking at it from the front face
        Functionally does a TL followed by left_corners
        Then rotates the cube right and does right_corners then rotates back left
        '''
        self.t_r()
        if show_steps:
            self.draw2d()
        self.left_corners(show_steps=show_steps)
        if show_steps:
            self.draw2d()
        self.turn_cube_right()
        if show_steps:
            self.draw2d()
        self.right_corners(show_steps = show_steps)
        self.turn_cube_left()

    def topmid_2_midright(self, show_steps = False):
        '''
        Places the top middle block to the right middle position on the cube
        Looking at it from the front face
        Functionally does a TR followed by right_corners
        Then rotates the cube left and does left_corners then rotates back right
        '''
        self.t_l()
        if show_steps:
            self.draw2d()
        self.right_corners(show_steps=show_steps)
        if show_steps:
            self.draw2d()
        self.turn_cube_left()
        if show_steps:
            self.draw2d()
        self.left_corners(show_steps = show_steps)
        self.turn_cube_right()
        
if __name__ == '__main__':
    c1 = RubiksCube()
    c1.draw2d()
    c1.scramble_cube((10))
    c1.draw2d()
    c1.rotate_cube_anticlock()        
    c1.draw2d()
    c1.rotate_cube_clock()        
    c1.draw2d()
    print(c1.moves)
    
    