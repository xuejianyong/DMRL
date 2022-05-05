# -*- coding: UTF-8 -*-
"""

"""

import numpy as np
import time
import sys
from  PIL import Image, ImageTk # ImageGrab
# import pyscreenshot as ImageGrab
import tkinter as tk
import itertools as it
import random
import os
import ast
import math
import keyboard
# import win32gui

UNIT = 70   # pixels
MAZE_H = 7  # grid height
MAZE_W = 7  # grid width
INTERVAL = 30
DOT_SIZE = 20
window_width = MAZE_W*UNIT
window_height = MAZE_H*UNIT

window_width_plus = 850
window_height_plus = 600
output_grid = 5
import matplotlib.pyplot as plt

x_range = range(0, MAZE_H)
y_range = range(0, MAZE_W)
all_locations = np.array(list(it.product(x_range,y_range)))*UNIT
all_locations = all_locations.tolist()
prey_move_directions = ['ne', 'nw', 'se', 'sw']


# all_locations = []
# for j in range(MAZE_H):
#     for i in range(MAZE_W):
#         all_locations.append([i,j])
# all_locations = (np.array(all_locations)*UNIT).tolist()

interaction_images = {
    'agent':'images/agent.png',
    'orange':'images/orange.png',
    'wall':'images/wall.png'
}

bg_color = {
    '5':'#191970',
    '4':'#191970',
    '3':'#0000FF',
    '2':'#4169E1',
    '1':'#6495ED',
    '0':'#B0C4DE'
}
currentPath = os.getcwd()


class TextMaze:
    def __init__(self, agent_original, prey_original, prey_direction):
        self.mark = 'text'

        # settings
        self.action_space = ['north', 'south', 'east', 'west', 'stay']
        self.n_actions = len(self.action_space)
        self.prey_move_directions = ['ne', 'nw', 'se', 'sw']

        # environment
        self.matrix = np.zeros((MAZE_H, MAZE_W), dtype=np.int8)

        # agent position
        x_agent, y_agent = agent_original
        self.agent_original_position = [x_agent, y_agent]
        self.agent_current_position = self.agent_original_position
        index_agentX = int(x_agent / UNIT)
        index_agentY = int(y_agent / UNIT)
        self.matrix[index_agentY][index_agentX] = 1
        # self.matrix[0, 0] = 1

        # prey position
        x_prey, y_prey = prey_original
        self.prey_original_position = [x_prey, y_prey]
        self.prey_current_position = self.prey_original_position
        index_preyX = int(x_prey / UNIT)
        index_preyY = int(y_prey / UNIT)
        self.matrix[index_preyY][index_preyX] = 2

        # prey movement direction
        self.prey_direction = prey_direction

    def initial_prey(self,scenario_1):
        # if scenario_1 > 0:
        #     self.canvas.delete(self.fruit)
        x_prey, y_prey = all_locations[random.randint(0, len(all_locations) - 1)]
        self.prey_original_position = [x_prey, y_prey]
        self.prey_current_position = self.prey_original_position
        self.prey_direction = self.prey_move_directions[random.randint(0, 3)]
        index_preyX = int(x_prey / UNIT)
        index_preyY = int(y_prey / UNIT)
        self.matrix[index_preyY][index_preyX] = 2  # 2 means the prey

    def reset(self):
        """
        self.matrix = np.zeros((MAZE_H, MAZE_W), dtype=np.int8)
        self.agent_current_position = self.agent_original_position
        self.matrix[0, 0] = 1
        self.prey_current_position = self.prey_original_position
        index_preyX = int(self.prey_original_position[0] / UNIT)
        index_preyY = int(self.prey_original_position[1] / UNIT)
        self.matrix[index_preyY][index_preyX] = 2
        """
        # return (np.array(self.prey_current_position) - np.array(self.agent_current_position)).tolist()
        return self.map_coords((np.array(self.prey_current_position) - np.array(self.agent_current_position)).tolist())

    def map_coords(self, observation):
        coordinate_x, coordinate_y = observation
        if coordinate_x > 210:
            coordinate_x = coordinate_x - 490
        elif coordinate_x < -210:
            coordinate_x = 490 - abs(coordinate_x)

        if coordinate_y > 210:
            coordinate_y = coordinate_y - 490
        elif coordinate_y < -210:
            coordinate_y = 490-abs(coordinate_y)


        # if coordinate_y > 210:
        #     coordinate_y = 490 - coordinate_y
        # elif 0 <= coordinate_y <= 210:
        #     coordinate_y = -coordinate_y
        # elif -210 <= coordinate_y < 0:
        #     coordinate_y = abs(coordinate_y)
        # elif coordinate_y < 210:
        #     coordinate_y = -(490-abs(coordinate_y))
        coords = [coordinate_x, coordinate_y]
        return coords

    def step_new(self, action):
        """
        # ['north', 'south', 'east', 'west', 'stay']
        s = self.agent_current_position
        base_action = np.array([0, 0])
        if action == 0:   # north
            if s[1] >= UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # south
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # east
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # west
            if s[0] >= UNIT:
                base_action[0] -= UNIT
        elif action == 4:   # stay
            pass

        # move prey
        fruit_coords = self.prey_current_position
        base_fruit = np.array([0, 0])
        if  self.prey_direction == 'ne':
            if (fruit_coords[0] == 0 and fruit_coords[1] == 0) or (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == (MAZE_W-1)*UNIT or fruit_coords[1] == 0:
                base_fruit[0] = fruit_coords[1]-fruit_coords[0]
                base_fruit[1] = fruit_coords[0]-fruit_coords[1]
            else:
                base_fruit[0] += UNIT
                base_fruit[1] -= UNIT
        elif self.prey_direction == 'nw':
            if (fruit_coords[0]==0 and fruit_coords[1]==(MAZE_H-1)*UNIT) or (fruit_coords[0]==(MAZE_W-1)*UNIT and fruit_coords[1]==0):
                pass
            elif fruit_coords[0] == 0 or fruit_coords[1] == 0:
                base_fruit[0] = (MAZE_H-1)*UNIT-fruit_coords[1]-fruit_coords[0]
                base_fruit[1] = (MAZE_W-1)*UNIT-fruit_coords[0]-fruit_coords[1]
            else:
                base_fruit[0] -= UNIT
                base_fruit[1] -= UNIT
        elif self.prey_direction == 'se':
            if (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == 0) or (fruit_coords[0] == 0 and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == (MAZE_W-1)*UNIT or fruit_coords[1] == (MAZE_H-1)*UNIT:
                base_fruit[0] = (MAZE_H-1)*UNIT - fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = (MAZE_W-1)*UNIT - fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] += UNIT
                base_fruit[1] += UNIT
        elif self.prey_direction == 'sw':  # testici
            if (fruit_coords[0] == 0 and fruit_coords[1] == 0) or (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == 0 or fruit_coords[1] == (MAZE_H-1)*UNIT:
                base_fruit[0] = fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] -= UNIT
                base_fruit[1] += UNIT
        """
        # agent : ['north', 'south', 'east', 'west', 'stay']
        [agent_x, agent_y] = self.agent_current_position
        index_agentX = int(agent_x / UNIT)
        index_agentY = int(agent_y / UNIT)
        self.matrix[index_agentY][index_agentX] = 0
        if action == 0:  # north
            if agent_y >= UNIT:
                agent_y -= UNIT
            elif agent_y == 0:
                agent_y = (MAZE_H - 1) * UNIT
        elif action == 1:  # south
            if agent_y < (MAZE_H - 1) * UNIT:
                agent_y += UNIT
            elif agent_y == (MAZE_H - 1) * UNIT:
                agent_y = 0
        elif action == 2:  # east
            if agent_x < (MAZE_W - 1) * UNIT:
                agent_x += UNIT
            elif agent_x == (MAZE_W - 1) * UNIT:
                agent_x = 0
        elif action == 3:  # west
            if agent_x >= UNIT:
                agent_x -= UNIT
            elif agent_x == 0:
                agent_x = (MAZE_W - 1) * UNIT
        elif action == 4:  # stay
            pass
        self.agent_current_position = [agent_x, agent_y]

        index_agentX = int(agent_x / UNIT)
        index_agentY = int(agent_y / UNIT)
        self.matrix[index_agentY][index_agentX] = 1

        # move prey
        [fruit_x, fruit_y] = self.prey_current_position
        index_preyX = int(fruit_x / UNIT)
        index_preyY = int(fruit_y / UNIT)
        self.matrix[index_preyY][index_preyX] = 0
        if self.prey_direction == 'ne':  # north east
            if (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y == 0):
                fruit_x = 0
                fruit_y = (MAZE_H - 1) * UNIT
            elif (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y != 0):
                fruit_x = 0
                fruit_y -= UNIT
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y == 0):
                fruit_x += UNIT
                fruit_y = (MAZE_H - 1) * UNIT
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y != 0):
                fruit_x += UNIT
                fruit_y -= UNIT
        elif self.prey_direction == 'nw':
            if (fruit_x == 0) and (fruit_y == 0):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y = (MAZE_H-1)*UNIT
            elif (fruit_x == 0) and (fruit_y != 0):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y -= UNIT
            elif (fruit_x != 0) and (fruit_y == 0):
                fruit_x -= UNIT
                fruit_y = (MAZE_H-1)*UNIT
            elif (fruit_x != 0) and (fruit_y != 0):
                fruit_x -= UNIT
                fruit_y -= UNIT
        elif self.prey_direction == 'se':
            if (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x = 0
                fruit_y = 0
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x += UNIT
                fruit_y = 0
            elif (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x = 0
                fruit_y += UNIT
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x += UNIT
                fruit_y += UNIT
        elif self.prey_direction == 'sw':  # testici
            if (fruit_x == 0) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y = 0
            elif (fruit_x != 0) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x -= UNIT
                fruit_y = 0
            elif (fruit_x == 0) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y += UNIT
            elif (fruit_x != 0) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x -= UNIT
                fruit_y += UNIT
        self.prey_current_position = [fruit_x, fruit_y]

        index_preyX = int(fruit_x / UNIT)
        index_preyY = int(fruit_y / UNIT)
        self.matrix[index_preyY][index_preyX] = 2

        # get next observation of state, with its corresponding reward
        # s_ = (np.array(self.prey_current_position) - np.array(self.agent_current_position)).tolist()
        s_ = self.map_coords((np.array(self.prey_current_position) - np.array(self.agent_current_position)).tolist())
        if self.prey_direction == 'sw' and s_ == [0, 0]:  # hunter catches the prey
            reward = 10
            done = True
        elif self.prey_direction != 'sw' and s_ == [0, 0]:
            reward = -1
            done = True
        else:
            reward = -1
            done = False
        return s_, reward, done

    def step(self, action):
        """
        # ['north', 'south', 'east', 'west', 'stay']
        s = self.agent_current_position
        base_action = np.array([0, 0])
        if action == 0:   # north
            if s[1] >= UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # south
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # east
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # west
            if s[0] >= UNIT:
                base_action[0] -= UNIT
        elif action == 4:   # stay
            pass

        # move prey
        fruit_coords = self.prey_current_position
        base_fruit = np.array([0, 0])
        if  self.prey_direction == 'ne':
            if (fruit_coords[0] == 0 and fruit_coords[1] == 0) or (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == (MAZE_W-1)*UNIT or fruit_coords[1] == 0:
                base_fruit[0] = fruit_coords[1]-fruit_coords[0]
                base_fruit[1] = fruit_coords[0]-fruit_coords[1]
            else:
                base_fruit[0] += UNIT
                base_fruit[1] -= UNIT
        elif self.prey_direction == 'nw':
            if (fruit_coords[0]==0 and fruit_coords[1]==(MAZE_H-1)*UNIT) or (fruit_coords[0]==(MAZE_W-1)*UNIT and fruit_coords[1]==0):
                pass
            elif fruit_coords[0] == 0 or fruit_coords[1] == 0:
                base_fruit[0] = (MAZE_H-1)*UNIT-fruit_coords[1]-fruit_coords[0]
                base_fruit[1] = (MAZE_W-1)*UNIT-fruit_coords[0]-fruit_coords[1]
            else:
                base_fruit[0] -= UNIT
                base_fruit[1] -= UNIT
        elif self.prey_direction == 'se':
            if (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == 0) or (fruit_coords[0] == 0 and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == (MAZE_W-1)*UNIT or fruit_coords[1] == (MAZE_H-1)*UNIT:
                base_fruit[0] = (MAZE_H-1)*UNIT - fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = (MAZE_W-1)*UNIT - fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] += UNIT
                base_fruit[1] += UNIT
        elif self.prey_direction == 'sw':  # testici
            if (fruit_coords[0] == 0 and fruit_coords[1] == 0) or (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == 0 or fruit_coords[1] == (MAZE_H-1)*UNIT:
                base_fruit[0] = fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] -= UNIT
                base_fruit[1] += UNIT
        """

        # agent : ['north', 'south', 'east', 'west', 'stay']
        [agent_x, agent_y] = self.agent_current_position
        index_agentX = int(agent_x / UNIT)
        index_agentY = int(agent_y / UNIT)
        self.matrix[index_agentY][index_agentX] = 0
        if action == 0:  # north
            if agent_y >= UNIT:
                agent_y -= UNIT
            elif agent_y == 0:
                agent_y = (MAZE_H - 1) * UNIT
        elif action == 1:  # south
            if agent_y < (MAZE_H - 1) * UNIT:
                agent_y += UNIT
            elif agent_y == (MAZE_H - 1) * UNIT:
                agent_y = 0
        elif action == 2:  # east
            if agent_x < (MAZE_W - 1) * UNIT:
                agent_x += UNIT
            elif agent_x == (MAZE_W - 1) * UNIT:
                agent_x = 0
        elif action == 3:  # west
            if agent_x >= UNIT:
                agent_x -= UNIT
            elif agent_x == 0:
                agent_x = (MAZE_W - 1) * UNIT
        elif action == 4:  # stay
            pass
        self.agent_current_position = [agent_x, agent_y]

        index_agentX = int(agent_x / UNIT)
        index_agentY = int(agent_y / UNIT)
        self.matrix[index_agentY][index_agentX] = 1

        # move prey
        [fruit_x, fruit_y] = self.prey_current_position
        index_preyX = int(fruit_x / UNIT)
        index_preyY = int(fruit_y / UNIT)
        self.matrix[index_preyY][index_preyX] = 0
        if self.prey_direction == 'ne':  # north east
            if (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y == 0):
                fruit_x = 0
                fruit_y = (MAZE_H - 1) * UNIT
            elif (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y != 0):
                fruit_x = 0
                fruit_y -= UNIT
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y == 0):
                fruit_x += UNIT
                fruit_y = (MAZE_H - 1) * UNIT
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y != 0):
                fruit_x += UNIT
                fruit_y -= UNIT
        elif self.prey_direction == 'nw':
            if (fruit_x == 0) and (fruit_y == 0):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y = (MAZE_H-1)*UNIT
            elif (fruit_x == 0) and (fruit_y != 0):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y -= UNIT
            elif (fruit_x != 0) and (fruit_y == 0):
                fruit_x -= UNIT
                fruit_y = (MAZE_H-1)*UNIT
            elif (fruit_x != 0) and (fruit_y != 0):
                fruit_x -= UNIT
                fruit_y -= UNIT
        elif self.prey_direction == 'se':
            if (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x = 0
                fruit_y = 0
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x += UNIT
                fruit_y = 0
            elif (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x = 0
                fruit_y += UNIT
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x += UNIT
                fruit_y += UNIT
        elif self.prey_direction == 'sw':  # testici
            if (fruit_x == 0) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y = 0
            elif (fruit_x != 0) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x -= UNIT
                fruit_y = 0
            elif (fruit_x == 0) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y += UNIT
            elif (fruit_x != 0) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x -= UNIT
                fruit_y += UNIT
        self.prey_current_position = [fruit_x, fruit_y]

        index_preyX = int(fruit_x / UNIT)
        index_preyY = int(fruit_y / UNIT)
        self.matrix[index_preyY][index_preyX] = 2

        # get next observation of state, with its corresponding reward
        s_ = (np.array(self.prey_current_position) - np.array(self.agent_current_position)).tolist()
        if s_ == [0, 0]:  # hunter catches the prey
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        return s_, reward, done

class GraphicMaze(tk.Tk, object):
    def __init__(self, agent_text, prey_text, prey_direction_text, require):
        super(GraphicMaze, self).__init__()
        self.mark = 'graphic'
        self.require = require
        self.action_space = ['north', 'south', 'east', 'west', 'stay']
        self.n_actions = len(self.action_space)
        self.prey_move_directions = ['ne', 'nw', 'se', 'sw']

        # self.title('MMRL-Maze')
        self.title('Torus grid world')
        self.geometry("{}x{}+{}+{}".format(window_width, window_height, 100, 100))
        self._build_maze(agent_text, prey_text, prey_direction_text)



    def _build_maze(self, agent_text, prey_text, prey_direction_text):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):  # create |
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):  # create --
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create the agent and the fruit(or the prey)
        agent_image = Image.open(os.path.join(currentPath, interaction_images['agent']))
        agent_image = agent_image.resize((UNIT, UNIT), Image.ANTIALIAS)
        self.agent_img = ImageTk.PhotoImage(agent_image)
        x_agent, y_agent = agent_text[0], agent_text[1]
        self.agent_original_position = agent_text
        self.agent_current_position = agent_text
        self.agent = self.canvas.create_image(x_agent, y_agent, anchor="nw", image=self.agent_img)

        # fruit_image = Image.open(interaction_images['orange'])
        fruit_image = Image.open(os.path.join(currentPath, interaction_images['orange']))
        fruit_image = fruit_image.resize((UNIT, UNIT), Image.ANTIALIAS)
        self.fruit_img = ImageTk.PhotoImage(fruit_image)
        x_prey, y_prey = prey_text[0], prey_text[1]
        self.prey_original_position = prey_text
        self.prey_current_position = prey_text
        self.fruit = self.canvas.create_image(x_prey, y_prey, anchor="nw", image=self.fruit_img)
        self.prey_direction = prey_direction_text

        # bind the function
        self.canvas.bind("<Button-1>", lambda event: self.drawRect(event))

        # pack all
        self.canvas.pack()

    def drawRect(self, event):
        click_x = (event.x // UNIT) * UNIT
        click_y = (event.y // UNIT) * UNIT
        click_x_color = click_x + UNIT // 2
        click_y_color = click_y + UNIT // 2
        color = ImageGrab.grab().getpixel(
            (event.x_root - event.x + click_x_color, event.y_root - event.y + click_y_color))
        if color[0] == 51:  # 这块的处理方式需要考虑一下，看能不能尝试使用 del 的方法来处理
            # print('get the point')
            self.canvas.create_rectangle(click_x + 1, click_y + 1, click_x + UNIT - 1, click_y + UNIT - 1, fill='white',
                                         outline='white')
        else:
            self.canvas.create_image(click_x + 1, click_y + 1, anchor="nw", image=self.wall_img)

    def reset(self):
        # self.update()
        # time.sleep(1)
        self.canvas.delete(self.agent)  # reset the position of the agent and update the canvas
        self.agent = self.canvas.create_image(0, 0, anchor="nw", image=self.agent_img)

        self.canvas.delete(self.fruit)
        self.fruit = self.canvas.create_image(self.prey_original_position[0], self.prey_original_position[1],anchor="nw", image=self.fruit_img)

        self.update()
        time.sleep(1)

        agent_coords = self.canvas.coords(self.agent)
        x_agent = agent_coords[0]
        y_agent = agent_coords[1]
        observation = [int(x_agent), int(y_agent)]
        # return self.canvas.coords(self.agent)  # return observation
        return observation

    def initial_prey(self, scenario_1):
        if scenario_1 > 0:
            self.canvas.delete(self.fruit)
        x_prey, y_prey = all_locations[random.randint(0, len(all_locations) - 1)]
        self.prey_original_position = [x_prey, y_prey]
        self.fruit = self.canvas.create_image(x_prey, y_prey, anchor="nw", image=self.fruit_img)
        self.prey_direction = self.prey_move_directions[random.randint(0, 3)]
        # self.prey_direction = 'sw'

    def step_new(self, action):
        """
        # ['north', 'south', 'east', 'west', 'stay']
        s = self.agent_current_position
        base_action = np.array([0, 0])
        if action == 0:   # north
            if s[1] >= UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # south
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # east
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # west
            if s[0] >= UNIT:
                base_action[0] -= UNIT
        elif action == 4:   # stay
            pass

        # move prey
        fruit_coords = self.prey_current_position
        base_fruit = np.array([0, 0])
        if  self.prey_direction == 'ne':
            if (fruit_coords[0] == 0 and fruit_coords[1] == 0) or (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == (MAZE_W-1)*UNIT or fruit_coords[1] == 0:
                base_fruit[0] = fruit_coords[1]-fruit_coords[0]
                base_fruit[1] = fruit_coords[0]-fruit_coords[1]
            else:
                base_fruit[0] += UNIT
                base_fruit[1] -= UNIT
        elif self.prey_direction == 'nw':
            if (fruit_coords[0]==0 and fruit_coords[1]==(MAZE_H-1)*UNIT) or (fruit_coords[0]==(MAZE_W-1)*UNIT and fruit_coords[1]==0):
                pass
            elif fruit_coords[0] == 0 or fruit_coords[1] == 0:
                base_fruit[0] = (MAZE_H-1)*UNIT-fruit_coords[1]-fruit_coords[0]
                base_fruit[1] = (MAZE_W-1)*UNIT-fruit_coords[0]-fruit_coords[1]
            else:
                base_fruit[0] -= UNIT
                base_fruit[1] -= UNIT
        elif self.prey_direction == 'se':
            if (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == 0) or (fruit_coords[0] == 0 and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == (MAZE_W-1)*UNIT or fruit_coords[1] == (MAZE_H-1)*UNIT:
                base_fruit[0] = (MAZE_H-1)*UNIT - fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = (MAZE_W-1)*UNIT - fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] += UNIT
                base_fruit[1] += UNIT
        elif self.prey_direction == 'sw':  # testici
            if (fruit_coords[0] == 0 and fruit_coords[1] == 0) or (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == 0 or fruit_coords[1] == (MAZE_H-1)*UNIT:
                base_fruit[0] = fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] -= UNIT
                base_fruit[1] += UNIT
        """
        # agent : ['north', 'south', 'east', 'west', 'stay']
        [agent_x, agent_y] = self.agent_current_position
        index_agentX = int(agent_x / UNIT)
        index_agentY = int(agent_y / UNIT)
        # self.matrix[index_agentY][index_agentX] = 0
        if action == 0:  # north
            if agent_y >= UNIT:
                agent_y -= UNIT
            elif agent_y == 0:
                agent_y = (MAZE_H - 1) * UNIT
        elif action == 1:  # south
            if agent_y < (MAZE_H - 1) * UNIT:
                agent_y += UNIT
            elif agent_y == (MAZE_H - 1) * UNIT:
                agent_y = 0
        elif action == 2:  # east
            if agent_x < (MAZE_W - 1) * UNIT:
                agent_x += UNIT
            elif agent_x == (MAZE_W - 1) * UNIT:
                agent_x = 0
        elif action == 3:  # west
            if agent_x >= UNIT:
                agent_x -= UNIT
            elif agent_x == 0:
                agent_x = (MAZE_W - 1) * UNIT
        elif action == 4:  # stay
            pass
        self.agent_current_position = [agent_x, agent_y]

        # index_agentX = int(agent_x / UNIT)
        # index_agentY = int(agent_y / UNIT)
        # self.matrix[index_agentY][index_agentX] = 1

        self.canvas.delete(self.agent)  # reset the position of the agent and update the canvas
        self.agent = self.canvas.create_image(agent_x, agent_y, anchor="nw", image=self.agent_img)

        # move prey
        [fruit_x, fruit_y] = self.prey_current_position
        index_preyX = int(fruit_x / UNIT)
        index_preyY = int(fruit_y / UNIT)
        # self.matrix[index_preyY][index_preyX] = 0
        if self.prey_direction == 'ne':  # north east
            if (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y == 0):
                fruit_x = 0
                fruit_y = (MAZE_H - 1) * UNIT
            elif (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y != 0):
                fruit_x = 0
                fruit_y -= UNIT
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y == 0):
                fruit_x += UNIT
                fruit_y = (MAZE_H - 1) * UNIT
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y != 0):
                fruit_x += UNIT
                fruit_y -= UNIT
        elif self.prey_direction == 'nw':
            if (fruit_x == 0) and (fruit_y == 0):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y = (MAZE_H-1)*UNIT
            elif (fruit_x == 0) and (fruit_y != 0):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y -= UNIT
            elif (fruit_x != 0) and (fruit_y == 0):
                fruit_x -= UNIT
                fruit_y = (MAZE_H-1)*UNIT
            elif (fruit_x != 0) and (fruit_y != 0):
                fruit_x -= UNIT
                fruit_y -= UNIT
        elif self.prey_direction == 'se':
            if (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x = 0
                fruit_y = 0
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x += UNIT
                fruit_y = 0
            elif (fruit_x == (MAZE_W-1)*UNIT) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x = 0
                fruit_y += UNIT
            elif (fruit_x != (MAZE_W-1)*UNIT) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x += UNIT
                fruit_y += UNIT
        elif self.prey_direction == 'sw':  # testici
            if (fruit_x == 0) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y = 0
            elif (fruit_x != 0) and (fruit_y == (MAZE_H-1)*UNIT):
                fruit_x -= UNIT
                fruit_y = 0
            elif (fruit_x == 0) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x = (MAZE_W-1)*UNIT
                fruit_y += UNIT
            elif (fruit_x != 0) and (fruit_y != (MAZE_H-1)*UNIT):
                fruit_x -= UNIT
                fruit_y += UNIT
        self.prey_current_position = [fruit_x, fruit_y]

        self.canvas.delete(self.fruit)
        self.fruit = self.canvas.create_image(fruit_x, fruit_y,anchor="nw", image=self.fruit_img)

        # index_preyX = int(fruit_x / UNIT)
        # index_preyY = int(fruit_y / UNIT)
        # self.matrix[index_preyY][index_preyX] = 2

        # get next observation of state, with its corresponding reward
        # s_ = (np.array(self.prey_current_position) - np.array(self.agent_current_position)).tolist()

        """
        s_ = self.map_coords((np.array(self.prey_current_position) - np.array(self.agent_current_position)).tolist())
        if s_ == [0, 0]:  # hunter catches the prey
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        return s_, reward, done
        """

    def step(self, action):
        # print("action: "+self.action_space[action])
        # move agent
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        if action == 0:  # north
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # south
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # east
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # west
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 4:  # stay
            pass
        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

        # move prey
        fruit_coords = list(map(int, self.canvas.coords(self.fruit)))
        base_fruit = np.array([0, 0])
        if self.prey_direction == 'ne':
            if (fruit_coords[0] == 0 and fruit_coords[1] == 0) or (
                    fruit_coords[0] == (MAZE_W - 1) * UNIT and fruit_coords[1] == (MAZE_H - 1) * UNIT):
                pass
            elif fruit_coords[0] == (MAZE_W - 1) * UNIT or fruit_coords[1] == 0:
                base_fruit[0] = fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] += UNIT
                base_fruit[1] -= UNIT
        elif self.prey_direction == 'nw':
            if (fruit_coords[0] == 0 and fruit_coords[1] == (MAZE_H - 1) * UNIT) or (
                    fruit_coords[0] == (MAZE_W - 1) * UNIT and fruit_coords[1] == 0):
                pass
            elif fruit_coords[0] == 0 or fruit_coords[1] == 0:
                base_fruit[0] = (MAZE_H - 1) * UNIT - fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = (MAZE_W - 1) * UNIT - fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] -= UNIT
                base_fruit[1] -= UNIT
        elif self.prey_direction == 'se':
            if (fruit_coords[0] == (MAZE_W - 1) * UNIT and fruit_coords[1] == 0) or (
                    fruit_coords[0] == 0 and fruit_coords[1] == (MAZE_H - 1) * UNIT):
                pass
            elif fruit_coords[0] == (MAZE_W - 1) * UNIT or fruit_coords[1] == (MAZE_H - 1) * UNIT:
                base_fruit[0] = (MAZE_H - 1) * UNIT - fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = (MAZE_W - 1) * UNIT - fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] += UNIT
                base_fruit[1] += UNIT
        elif self.prey_direction == 'sw':  # testici
            if (fruit_coords[0] == 0 and fruit_coords[1] == 0) or (
                    fruit_coords[0] == (MAZE_W - 1) * UNIT and fruit_coords[1] == (MAZE_H - 1) * UNIT):
                pass
            elif fruit_coords[0] == 0 or fruit_coords[1] == (MAZE_H - 1) * UNIT:
                base_fruit[0] = fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] -= UNIT
                base_fruit[1] += UNIT
        self.canvas.move(self.fruit, base_fruit[0], base_fruit[1])  # move fruit

        # get next state and the corresponding reward
        s_ = self.canvas.coords(self.agent)
        if s_ == self.canvas.coords(self.fruit):
            reward = 10
            done = True
            # s_ = 'terminal'  # comment this state
            # with the reason that the ternimal state is dynamic and it could be anywhere
            # thus we just record the final state with the flag of "done"
        else:
            reward = -1
            done = False

        agent_coords = self.canvas.coords(self.agent)
        x_agent = agent_coords[0]
        y_agent = agent_coords[1]
        s_ = [int(x_agent), int(y_agent)]

        return s_, reward, done

    def render(self):
        self.update()
        time.sleep(0.2)

    def show_modules(self, modules):
        popWindow = tk.Tk()
        x_cordinate = int((self.winfo_screenwidth() / 2) - (window_width_plus / 2))
        y_cordinate = int((self.winfo_screenheight() / 2) - (window_height_plus / 2))
        popWindow.geometry("{}x{}+{}+{}".format(window_width_plus, window_height_plus, x_cordinate, y_cordinate))
        popWindow.title("Results")
        cvs = tk.Canvas(popWindow, bg='white', height=window_height_plus, width=window_width_plus)
        # cvs.place(x=INTERVAL, y=INTERVAL)
        cvs.pack()

        for module_i in modules:
            # print(module_i.serial)
            q_table = module_i.q_table
            module_id = int(module_i.serial)

            x_anchor = DOT_SIZE + (module_id % output_grid) * (MAZE_W * DOT_SIZE) + (module_id % output_grid) * DOT_SIZE
            y_anchor = DOT_SIZE + (math.floor(module_id / output_grid)) * (MAZE_H * DOT_SIZE) + (
                math.floor(module_id / output_grid)) * DOT_SIZE

            cvs.create_line(x_anchor, y_anchor, x_anchor + MAZE_W * DOT_SIZE, y_anchor)
            cvs.create_line(x_anchor, y_anchor, x_anchor, y_anchor + MAZE_H * DOT_SIZE)
            cvs.create_line(x_anchor + MAZE_W * DOT_SIZE, y_anchor, x_anchor + MAZE_W * DOT_SIZE,
                            y_anchor + MAZE_H * DOT_SIZE)
            cvs.create_line(x_anchor, y_anchor + MAZE_W * DOT_SIZE, x_anchor + MAZE_W * DOT_SIZE,
                            y_anchor + MAZE_H * DOT_SIZE)

            n_max = q_table.max().max()
            n_min = q_table.min().min()
            piece = (n_max - n_min) / 5
            # print(q_table)
            indexList = q_table.index
            for index_i in indexList:
                if index_i != 'terminal':
                    index_coords = ast.literal_eval(index_i)
                    index_value = np.max(q_table.loc[index_i, :])
                    n_piece = math.floor((index_value - n_min) / piece)
                    x_coord = index_coords[0] / UNIT * DOT_SIZE
                    y_coord = index_coords[1] / UNIT * DOT_SIZE
                    cvs.create_rectangle(x_anchor + x_coord + 1, y_anchor + y_coord + 1,
                                         x_anchor + x_coord + DOT_SIZE - 1, y_anchor + y_coord + DOT_SIZE - 1,
                                         fill=bg_color[str(n_piece)], outline='white')
        # popWindow.mainloop()

        # cvs.update()
        # HWND = win32gui.GetFocus()
        # rect = win32gui.GetWindowRect(HWND)
        # x = rect[0]
        # x1 = x + cvs.winfo_width()
        # y = rect[1]
        # y1 = y + cvs.winfo_height()
        # im = ImageGrab.grab((x, y, x1, y1))
        # im.save("second.jpeg", 'jpeg')

    def show_table_data(self, q_table):
        popWindow = tk.Tk()
        x_cordinate = int((self.winfo_screenwidth() / 2) - (window_width_plus / 2))
        y_cordinate = int((self.winfo_screenheight() / 2) - (window_height_plus / 2))
        popWindow.geometry("{}x{}+{}+{}".format(window_width_plus, window_height_plus, x_cordinate, y_cordinate))
        popWindow.title("Results")
        cvs = tk.Canvas(popWindow, bg='white', height=window_height_plus, width=window_width_plus)
        # cvs.place(x=INTERVAL, y=INTERVAL)
        cvs.pack()

        module_id = 1

        x_anchor = DOT_SIZE + (module_id % output_grid) * (MAZE_W * DOT_SIZE) + (module_id % output_grid) * DOT_SIZE
        y_anchor = DOT_SIZE + (math.floor(module_id / output_grid)) * (MAZE_H * DOT_SIZE) + (
            math.floor(module_id / output_grid)) * DOT_SIZE

        cvs.create_line(x_anchor, y_anchor, x_anchor + MAZE_W * DOT_SIZE, y_anchor)
        cvs.create_line(x_anchor, y_anchor, x_anchor, y_anchor + MAZE_H * DOT_SIZE)
        cvs.create_line(x_anchor + MAZE_W * DOT_SIZE, y_anchor, x_anchor + MAZE_W * DOT_SIZE,
                        y_anchor + MAZE_H * DOT_SIZE)
        cvs.create_line(x_anchor, y_anchor + MAZE_W * DOT_SIZE, x_anchor + MAZE_W * DOT_SIZE,
                        y_anchor + MAZE_H * DOT_SIZE)

        n_max = q_table.max().max()
        n_min = q_table.min().min()
        piece = (n_max - n_min) / 5
        # print(q_table)
        indexList = q_table.index
        for index_i in indexList:
            if index_i != 'terminal':
                index_coords = ast.literal_eval(index_i)
                index_value = np.max(q_table.loc[index_i, :])
                n_piece = math.floor((index_value - n_min) / piece)
                x_coord = index_coords[0] / UNIT * DOT_SIZE
                y_coord = index_coords[1] / UNIT * DOT_SIZE
                cvs.create_rectangle(x_anchor + x_coord + 1, y_anchor + y_coord + 1, x_anchor + x_coord + DOT_SIZE - 1,
                                     y_anchor + y_coord + DOT_SIZE - 1, fill=bg_color[str(n_piece)], outline='white')
    # popWindow.mainloop()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 4
            env.step_new(a)
            # if done:
            #     break



if __name__ == '__main__':
    # env = Maze()
    # env.after(100, update)
    # env.mainloop()

    agent_original = [0, 0]
    if agent_original in all_locations:
        all_locations.remove(agent_original)
    # prey_direction = prey_move_directions[random.randint(0, 3)]
    # prey_original = all_locations[random.randint(0, len(all_locations) - 1)]
    prey_direction = 'se'
    prey_original = [210, 420]
    # print(agent_original, prey_original, prey_direction)
    # env = TextMaze(agent_original, prey_original, prey_direction, require='nn')
    env = GraphicMaze(agent_original, prey_original, prey_direction, require='nn')
    env.after(100, update)
    env.mainloop()