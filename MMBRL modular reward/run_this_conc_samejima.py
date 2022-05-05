# -*- coding: UTF-8 -*-
"""
Project description:
The implementation of Multiple model-based reinforcement learning
Specificaly, this project is trying to follow the environmetn settings like trials and
the predict the prey in the prediction model.

Model designed by  Doya, 2002
Reimplemented by: Jianyong

"""

from environment_multitask_samejima import TextMaze, GraphicMaze
import matplotlib.pyplot as plt
import numpy as np
# from module import MultipleModel
import tkinter as tk
import ast
import itertools as it
import pandas as pd
import math
# from model_MMBRL import MMBRL, Module
from model_conc_samejima import MMBRL, Module
import time
import os

import configparser
# from interaction import Interaction

config = configparser.ConfigParser()
config.read('../../app.ini')

N_MODULE = config.getint('env','n_modules')
# SIMULATIONS = config.getint('env','simulation_test')
SIMULATIONS = config.getint('env','simulation')
TRIAL = config.getint('env','trial')
MULTITRIAL = config.getint('env','multitrial')
PERIODTRIAL = config.getint('env','periodtrial')
TRIAL_SAMEJIMA = config.getint('env','trial_samejima')

EPISODES = 1000
MAX_STEP = 100

UNIT = 70   # pixels
DOT_SIZE = 20
window_width_plus = 850
window_height_plus = 600
output_grid = 5

bg_color = {
    '5':'#191970',
    '4':'#191970',
    '3':'#0000FF',
    '2':'#4169E1',
    '1':'#6495ED',
    '0':'#B0C4DE'
}

MAZE_H = 7  # grid height
MAZE_W = 7  # grid width
x_range = range(0, MAZE_H)
y_range = range(0, MAZE_W)
# all_locations = np.array(list(it.product(x_range, y_range)))*UNIT
# all_locations = all_locations.tolist()


prey_move_directions = ['ne', 'nw', 'se', 'sw']
action_space = ['north', 'south', 'east', 'west', 'stay']

step_list = {}
simulation_list = []
direction_dict = {}  # count the number of times the direction appears
tasks = {0:['ne'],
         1:['ne','nw'],
         2:['ne','nw', 'se'],
         3:['ne','nw', 'se', 'sw']
         }

np.random.seed(1)

def get_all_locations():
    all_locations = np.array(list(it.product(x_range, y_range))) * UNIT
    all_locations = all_locations.tolist()
    return all_locations

def update():
    print("--- Interaction starts ---")
    start = time.time()
    filename = "samejima_modular_reward-m"+str(N_MODULE)+'s'+str(SIMULATIONS) +'mts'+ str(TRIAL_SAMEJIMA)
    for simulation_i in range(SIMULATIONS):  # SIMULATIONS
        step_list_i = []
        model = MMBRL(n_modules=N_MODULE, trial=TRIAL_SAMEJIMA, method='softmax')  #   softmax
        resp_method = 'samejima_modular_reward'
        for trial in range(TRIAL_SAMEJIMA):  # MULTITRIAL_SAMEJIMA
            agent_original = [0, 0]
            step = 0
            reward_total = 0
            value_total = 0.0
            for prey_direction in prey_move_directions:
                flag = 0
                # ------------ environment initialization ------------ #
                all_locations_i = get_all_locations()
                if agent_original in all_locations_i:
                    all_locations_i.remove(agent_original)
                prey_original = all_locations_i[np.random.randint(0, len(all_locations_i))]
                env = TextMaze(agent_original, prey_original, prey_direction)
                # ------------ interaction starts ------------ #
                observation = env.reset()
                while True:
                    action = model.choose_action(str(observation))

                    for module_i in model.modules:
                        value_total += module_i.responsibility * module_i.state_values[str(observation)]

                    observation_, reward, done = env.step_new(action)
                    model.learn(str(observation), action, reward, str(observation_), resp_method, flag)
                    observation = observation_
                    step += 1
                    flag += 1
                    reward_total += reward
                    if done:
                        agent_original = env.agent_current_position
                        break
            print('simulation: %d/%d, [trial: %d/%d],  with %d steps.' % (simulation_i, SIMULATIONS, trial, TRIAL_SAMEJIMA, step))
            # step_list_i.append(step)

            list_values = []
            list_values.append(step)
            list_values.append(reward_total / step)
            # for module_i in model.modules:
            #     value_total += max(module_i.state_values.values())
            list_values.append(value_total / step)
            step_list_i.append(list_values)

        model.mark += model.method+'-'+str(simulation_i)
        step_list[model.mark] = step_list_i

    # show figures
    print('over')  # end of game
    # plot_step()

    period = time.time() - start
    if period > 3600:
        hour = period // 3600
        minute = (period - 3600 * hour) // 60
        second = period - 3600 * hour - minute * 60
        print('Time in this training:', str(hour) + ' h ' + str(minute) + ' min ' + str(second) + ' seds. ')
    else:
        minute = (period) // 60
        second = period - minute * 60
        print('Time in this training:', str(minute) + ' min ' + str(second) + ' seds. ')

    # plot_step_mean(filename)
    plot_step_multi_mean(filename)

    # for module_i in model.modules:
    #     show_pcolor_values(module_i.state_values)


    # show_modules()
    # model.show_pcolor_sv()

def plot_step_multi_mean(filename):
    currentpath = os.getcwd()
    filepath = os.path.join(currentpath, 'data')
    if not os.path.isdir(filepath):
        os.mkdir('data')
    means = []
    for model_mark in step_list.keys():
        step_list_i = step_list[model_mark]
        means.append(step_list_i)
    step_list_toal = []
    reward_list_toal = []
    value_list_toal = []
    for s_i in means:
        step_list_record = []
        reward_list_record = []
        value_list_record = []
        for ele in s_i:
            step_data = ele[0]
            reward_data = ele[1]
            value_data = ele[2]
            step_list_record.append(step_data)
            reward_list_record.append(reward_data)
            value_list_record.append(value_data)
        step_list_toal.append(step_list_record)
        reward_list_toal.append(reward_list_record)
        value_list_toal.append(value_list_record)

    step_mean_value = np.mean(step_list_toal, axis=0)
    reward_mean_value = np.mean(reward_list_toal, axis=0)
    value_mean_value = np.mean(value_list_toal, axis=0)

    # print(mean_value)
    filename += '-' + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    f = open("./data/" + filename + ".txt", 'w')
    for mean_step, mean_reward, mean_value in zip(step_mean_value, reward_mean_value, value_mean_value):
        f.write(str(mean_step) + ','+ str(mean_reward)+ ','+ str(mean_value) + '\n')
    f.close()

    labelname = os.path.basename(__file__)
    plt.plot(step_mean_value, label=labelname)
    plt.plot(reward_mean_value, label=labelname)
    plt.plot(value_mean_value, label=labelname)
    plt.xlabel('Number of trial')
    plt.ylabel('Number of steps in a trial')
    plt.legend()
    plt.show()


def get_max_key(sas_dict):
    key_max = ''
    value_max = 0.0
    # print(sas_dict)
    for key_i in sas_dict.keys():
        if sas_dict[key_i] > value_max:
            key_max = key_i
            value_max = sas_dict[key_i]
    return key_max

def show_graphic(agent_original, prey_original, prey_direction, model):
    env_graphic = GraphicMaze(agent_original, prey_original, prey_direction)
    print(model.mark)
    for i in range(5):
        # print('showing the %d episode' % i)
        step = 0
        observation = env_graphic.reset()
        while True:
            action = model.choose_action(str(observation))
            observation_, reward, done = env_graphic.step(action)
            env_graphic.render()
            observation = observation_
            step += 1
            if done or (step >= MAX_STEP):
                break
        print('The episode: %d,  with %d steps.' % (i, step))

def show_pcolor(q_table):
    # print(q_table)
    indexlist = q_table.index
    x = np.arange(0, 7, 1)
    y = np.arange(0, 7, 1)
    values = np.zeros([7, 7])
    for index_i in indexlist:
        if index_i != 'terminal':
            index_coords = ast.literal_eval(index_i)
            index_value = np.max(q_table.loc[index_i, :])
            x_coord = int(index_coords[0] / UNIT)
            y_coord = 6 - int(index_coords[1] / UNIT)
            values[y_coord][x_coord] = index_value
    plt.title('Max action value in each state')
    im = plt.pcolormesh(x, y, values, vmin=np.min(values), vmax=np.max(values), shading='auto')
    plt.colorbar(im)
    plt.show()

def show_pcolor_values(state_value_dict):
    # print(len(state_value_dict.keys()))
    indexlist = state_value_dict.keys()
    x = np.arange(0, 7, 1)
    y = np.arange(0, 7, 1)
    values = np.zeros([7, 7])
    for index_i in indexlist:
        if index_i != 'terminal':
            index_coords = ast.literal_eval(index_i)
            index_value = state_value_dict[index_i]
            x_coord = int((index_coords[0]+210) / UNIT)
            y_coord = int((index_coords[1]+210) / UNIT)
            values[x_coord][y_coord] = index_value
    plt.title('State values')
    im = plt.pcolormesh(x, y, values, vmin=np.min(values), vmax=np.max(values), shading='auto')
    plt.colorbar(im)
    plt.show()

def show_table_data(q_table):
    popWindow = tk.Tk()
    # x_cordinate = int((self.winfo_screenwidth() / 2) - (window_width_plus / 2))
    # y_cordinate = int((self.winfo_screenheight() / 2) - (window_height_plus / 2))
    popWindow.geometry("{}x{}+{}+{}".format(window_width_plus, window_height_plus, 100, 100))
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

"""
def show_table_data(q_table):
    popWindow = tk.Tk()
    popWindow.geometry("{}x{}+{}+{}".format(window_width_plus, window_height_plus, 100, 100))
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
"""

def plot_step():
    for model_mark in step_list.keys():
        step_list_i = step_list[model_mark]
        # plt.plot(list(np.arange(EPISODES) + 1), step_list)
        plt.plot(step_list_i, label=model_mark)
        plt.xlabel('Episodes')
        plt.ylabel('Step')
    plt.legend()
    plt.show()

def plot_step_mean(filename):
    currentpath = os.getcwd()
    filepath = os.path.join(currentpath, 'data')
    if not os.path.isdir(filepath):
        os.mkdir('data')

    means = []
    for model_mark in step_list.keys():
        step_list_i = step_list[model_mark]
        means.append(step_list_i)
    mean_value = np.mean(means, axis=0)
    filename += '-'+time.strftime("%Y%m%d-%H%M%S", time.localtime())
    f = open("./data/"+filename+".txt", 'w')
    for ele in mean_value:
        f.write(str(ele) + '\n')
    f.close()
    labelname = os.path.basename(__file__)
    plt.plot(mean_value, label=labelname)
    plt.xlabel('Number of trial')
    plt.ylabel('Number of steps in a trial')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    update()
