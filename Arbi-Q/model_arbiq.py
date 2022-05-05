# -*- coding: UTF-8 -*-
"""
This is an implementation of  MMBRL from Doya, 2002.
Specifially, much of information is unclear, thus I'm considering to implement it
based on the structure of MMBRL and thoughts that I have.

Information:
author:
links:
documents:

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from collections import deque
import itertools as it
import math
import tkinter as tk


UNIT = 70   # pixels
DOT_SIZE = 20

window_width_plus = 850
window_height_plus = 600
output_grid = 5

MAZE_H = 7  # grid height
MAZE_W = 7  # grid width
x_range = range(0, MAZE_H)
y_range = range(0, MAZE_W)
all_locations_int = np.array(list(it.product(x_range, y_range)))*UNIT
all_locations_int = all_locations_int.tolist()
all_locations = []
for location in all_locations_int:
    all_locations.append(str(location))
# all_locations = all_locations.append('terminal')

bg_color = {
    '5':'#191970',
    '4':'#191970',
    '3':'#0000FF',
    '2':'#4169E1',
    '1':'#6495ED',
    '0':'#B0C4DE'
}

np.random.seed(1)


class MMBRL:
    def __init__(self, n_modules, trial, method='epsilon-greedy'):  # epsilon-greedy
        self.mark = 'MMBRL Doya 2002 '
        self.action_space = ['north', 'south', 'east', 'west', 'stay']
        self.n_actions = len(self.action_space)
        self.actions = list(range(self.n_actions))
        self.n_modules = n_modules
        self.modules = []
        self.alpha = 0.85  #  or0.8, 0.87, 0.99 plotted the results in Figure 2, controls the  the memory effects
        self.beta = trial/500
        self.ksai = 0.001 # 0.1, 0.001
        self.lr = 0.2
        self.gamma = 0.9
        self.epsilon = 0.9
        self.sigma = 10
        self.eta = 0.5
        self.method = method
        self.module_method = 'greedy'

        self.watch_list = []
        self.watch_limit = 10

        self.ac_state_action_count = {}
        self.ac_state_action_pro = {}
        self.reward_ac = {}

        self.ac_state_action_state_pro = {}
        self.ac_state_action_reward_value = {}
        self.ac_state_value = {}

        self.memory = []
        self.modules_count = {}
        self.modules_pro = {}
        self.elements = []
        self.module_serials = []

        for i in range(n_modules):
            module_i = Module(i, self.actions, self.n_modules)
            self.module_serials.append(module_i.serial)
            self.modules.append(module_i)
            self.elements.append(i)
        self.selected_module = self.modules[0]

        self.ca_table = pd.DataFrame(columns=self.elements, dtype=np.float64)

    def module_selection(self, observation):
        self.check_state_exist_in_ca(observation)
        state_module = self.ca_table.loc[observation, :]
        if self.module_method == 'epsilon-greedy':
            if np.random.uniform() < self.epsilon:
                module_index = np.random.choice(state_module[state_module == np.max(state_module)].index)
            else:
                module_index = np.random.choice(self.actions)
        elif self.module_method == 'greedy':
            module_index = np.random.choice(state_module[state_module == np.min(state_module)].index)
        elif self.module_method == 'softmax':
            pi = np.exp(self.beta * np.array(state_module)) / sum(np.exp(self.beta * np.array(state_module)))
            module_index = np.random.choice(self.actions, p=pi)
        return self.modules[module_index]

    def check_state_exist_in_ca(self, state):
        if state not in self.ca_table.index:
            self.ca_table = self.ca_table.append(
                pd.Series(
                    [0.]*len(self.elements),
                    index=self.ca_table.columns,
                    name=state,
                )
            )


    def ac_update_state_actions(self, s, a):  # the action_list under the state of s
        # { state : {action1: count1, action2:count2} }
        if s not in self.ac_state_action_count.keys():
            self.ac_state_action_count[s] = {}
            self.ac_state_action_count[s][a] = 1
        else:
            if a not in self.ac_state_action_count[s].keys():
                self.ac_state_action_count[s][a] = 1
            else:
                self.ac_state_action_count[s][a] += 1

        # compute the probabilities
        # { state : {action1: pro1, action2:pro2} }
        # initialize the state_action probability
        if s not in self.ac_state_action_pro.keys():
            self.ac_state_action_pro[s] = {}
            self.ac_state_action_pro[s][a] = 1.0
        else:
            if a not in self.ac_state_action_pro[s].keys():
                self.ac_state_action_pro[s][a] = 1.0
        # start to update state_action probability
        action_count_dict = self.ac_state_action_count[s]
        count_sum = sum(action_count_dict.values())
        for action_i in action_count_dict.keys():
            if action_i not in self.ac_state_action_pro[s].keys():
                print('warning!!!, action_i of ac_state_action_count not exists in ac_state_action_pro')
            action_i_count = action_count_dict[action_i]
            self.ac_state_action_pro[s][action_i] = action_i_count / count_sum

    def choose_action(self, observation):
        self.selected_module = self.module_selection(observation)


        # state_action = pd.Series([0.0] * len(self.actions), index=self.actions, name='composite_state_action_value')
        if observation not in self.ac_state_value.keys():
            self.ac_state_value[observation] = 0.0
        for module_i in self.modules:
            module_i.check_state_exist(observation)
            if observation not in module_i.state_values.keys():
                module_i.state_values[observation] = 0.0
            if observation not in module_i.eligibility.keys():
                module_i.eligibility[observation] = 0.0

        state_action = self.selected_module.q_table.loc[observation, :]
        # action selection strategies
        if self.method == 'epsilon-greedy':
            if np.random.uniform() < self.epsilon:
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                action = np.random.choice(self.actions)
        elif self.method == 'greedy':
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        elif self.method == 'stochastic-greedy':
            pi = np.exp(self.beta * np.array(state_action)) / sum(np.exp(self.beta * np.array(state_action)))
            action = np.random.choice(self.actions, p=pi)
        elif self.method == 'softmax':
            pi = np.exp(np.array(state_action)) / sum(np.exp(np.array(state_action)))
            action = np.random.choice(self.actions, p=pi)
        return action

    def learn(self, s, a, r, s_, done, resp_method):
        key = s + str(a)
        # ------------ parameters initialization ------------ #
        # initializing the AC
        if s_ not in self.ac_state_value.keys():
            self.ac_state_value[s_] = 0.0
        self.check_state_exist_in_ca(s_)
        if key not in self.ac_state_action_reward_value.keys():
            self.ac_state_action_reward_value[key] = 0.0
        if key not in self.ac_state_action_state_pro.keys():  # dynamic function in module i
            self.ac_state_action_state_pro[key] = {}
            self.ac_state_action_state_pro[key][s_] = np.random.uniform()
        else:
            if s_ not in self.ac_state_action_state_pro[key].keys():
                self.ac_state_action_state_pro[key][s_] = np.random.uniform()

        # initializing the modules
        for module_i in self.modules:
            module_i.check_state_exist(s_)
            if s_ not in module_i.state_values.keys():  # state value function in module i
                module_i.state_values[s_] = 0.0
            if s_ not in module_i.eligibility.keys():   # eligibility traces
                module_i.eligibility[s_] = 0.0
            if key not in module_i.state_action_reward_value.keys():  # reward function in module i
                module_i.state_action_reward_value[key] = 0.0
            if key not in module_i.state_action_state_pro.keys():  # dynamic function in module i
                module_i.state_action_state_pro[key] = {}
                module_i.state_action_state_pro[key][s_] = np.random.uniform()
            else:
                if s_ not in module_i.state_action_state_pro[key].keys():
                    module_i.state_action_state_pro[key][s_] = np.random.uniform()

        # ------------ responsibility calculation ------------ #
        """
        if resp_method == 'original':
            dynamic_values = []
            for module_i in self.modules:
                predict_value_i = module_i.state_action_state_pro[key][s_] * module_i.responsibility_predictor
                dynamic_values.append(predict_value_i)
            sum_predict_values = sum(dynamic_values)
            if sum_predict_values == 0.0:
                sum_predict_values = 1.0
            dynamic_values = np.array(dynamic_values)/sum_predict_values
            module_index = np.argmax(dynamic_values)
            module_from_error = self.modules[module_index]
            self.selected_module = module_from_error
            for module_j, resp in zip(self.modules, dynamic_values):
                module_j.responsibility = resp
                module_j.responsibility_predictor = pow(resp, self.alpha)
        elif resp_method == 'error_softmax':
            dynamic_values = []
            for module_i in self.modules:
                predict_error_i = (module_i.state_action_state_pro[key][s_] - 1) #* module_i.responsibility_predictor
                dynamic_values.append(predict_error_i)
            dynamic_values = np.exp(dynamic_values)/sum(np.exp(dynamic_values))
            module_from_error = np.random.choice(self.modules, p=dynamic_values)
            self.selected_module = module_from_error
            self.selected_module.selected_times += 1
            for module_j, resp in zip(self.modules, dynamic_values):
                module_j.responsibility = resp
                module_j.responsibility_predictor = pow(resp, self.alpha)
        elif resp_method == 'samejima_modular_reward':
            dynamic_values = []
            for module_i in self.modules:
                sum_probability_i = sum(module_i.state_action_state_pro[key].values())
                predict_probability = module_i.state_action_state_pro[key][s_]/sum_probability_i  # p(s_|s,a)
                upper_predict_probability = predict_probability * module_i.responsibility_predictor
                dynamic_values.append(upper_predict_probability)
            dynamic_values = np.array(dynamic_values) / sum(dynamic_values)  # get responsibility value
            for module_j, resp in zip(self.modules, dynamic_values):
                module_j.responsibility = resp
                module_j.responsibility_predictor = pow(resp, self.alpha)
        elif resp_method == 'jj':
            dynamic_values = []
            for module_i in self.modules:
                sum_probability_i = sum(module_i.state_action_state_pro[key].values())
                predict_probability = module_i.state_action_state_pro[key][s_] / sum_probability_i  # p(s_|s,a)
                predict_error_i = 1 - predict_probability
                predict_error_i = np.exp(-pow(predict_error_i, 2)/2*pow(self.sigma, 2))
                upper_i = module_i.responsibility_predictor * predict_error_i
                dynamic_values.append(upper_i)
            dynamic_values = np.array(dynamic_values) / sum(dynamic_values)
            for module_j, resp in zip(self.modules, dynamic_values):
                module_j.responsibility = resp
                module_j.responsibility_predictor = pow(resp, self.alpha)
        """
        # ------------ module selection ------------ #
        """
        if len(self.watch_list) % self.watch_limit == 0:
            module_responsibility = pd.Series([0.0] * self.n_modules, index=self.module_serials, name='module_responsibilities')
            watch_error_i = []
            if len(self.watch_list) == 0:  # the first time
                for module_i in self.modules:
                    sum_probability_i = sum(module_i.state_action_state_pro[key].values())
                    predict_probability = module_i.state_action_state_pro[key][s_] / sum_probability_i  # p(s_|s,a)
                    module_responsibility[module_i.serial] = predict_probability
                    watch_error_i.append(predict_probability)
                    # watch_error_i.append(module_i.responsibility)
                self.watch_list.append(watch_error_i)
                max_module_serial = np.random.choice(module_responsibility[module_responsibility == np.max(module_responsibility)].index)
                for module_i in self.modules:
                    if module_i.serial == max_module_serial:
                        self.selected_module = module_i
            else:  # average the watch records
                mean_errors = np.sum(self.watch_list, axis=0)
                for i in range(self.n_modules):
                    module_responsibility[str(i)] = mean_errors[i]
                max_module_serial = np.random.choice(module_responsibility[module_responsibility == np.max(module_responsibility)].index)
                for module_i in self.modules:
                    if module_i.serial == max_module_serial:
                        self.selected_module = module_i
                self.watch_list = []
        else:  # append the records
            watch_error_i = []
            for module_i in self.modules:
                sum_probability_i = sum(module_i.state_action_state_pro[key].values())
                predict_probability = module_i.state_action_state_pro[key][s_] / sum_probability_i  # p(s_|s,a)
                watch_error_i.append(predict_probability)
                # watch_error_i.append(module_i.responsibility)
            self.watch_list.append(watch_error_i)
        """
        # ------------ module update ------------ #

        # update state-reward function
        state_error = r + self.gamma * self.ac_state_value[s_] - self.ac_state_value[s]  #?index
        self.ac_state_value[s] += state_error * self.lr

        # update state predictor
        self.ac_state_action_state_pro[key][s_] += self.lr * (1 - self.ksai * self.ac_state_action_state_pro[key][s_])

        # update state_action reward model
        ac_reward_error = r - self.ac_state_action_reward_value[key]
        self.ac_state_action_reward_value[key] += ac_reward_error * self.lr

        # update state-module value in the ac_table: Q_ca(s_ca, m) from m and R_ca(s_ca)
        self.ac_update_state_actions(s, a)
        ac_action_count_dict = self.ac_state_action_count[s]  # get the action list in this state
        for action_i in ac_action_count_dict.keys():
            ac_state_action_str = s + str(action_i)
            ac_latter_value = 0.0
            ac_state_action_state_dict = self.ac_state_action_state_pro[ac_state_action_str]
            ac_sum_probability = sum((ac_state_action_state_dict.values()))
            for next_state_i in ac_state_action_state_dict.keys():
                ac_latter_value += ac_state_action_state_dict[next_state_i]/ac_sum_probability * self.ac_state_value[next_state_i]
            self.ca_table.loc[s, self.selected_module.serial] = self.ac_state_action_reward_value[ac_state_action_str] + self.gamma * ac_latter_value

        # module_i = self.selected_module
        for module_i in self.modules:

            # update state prediction model
            sum_probability_i = sum(module_i.state_action_state_pro[key].values())
            predict_error_i = sum_probability_i - module_i.state_action_state_pro[key][s_]
            # module_i.state_action_state_pro[key][s_] += module_i.responsibility * (1 - self.ksai * module_i.state_action_state_pro[key][s_])
            for next_state_i in module_i.state_action_state_pro[key].keys():
                if next_state_i == s_:
                    module_i.state_action_state_pro[key][next_state_i] += module_i.responsibility * predict_error_i * self.lr
                else:
                    error_i = module_i.state_action_state_pro[key][next_state_i]
                    module_i.state_action_state_pro[key][next_state_i] -= module_i.responsibility * error_i * self.lr

            # update state_action reward model
            reward_error = r - module_i.state_action_reward_value[key]
            module_i.state_action_reward_value[key] += module_i.responsibility * reward_error * self.lr

            # update state_value function
            module_i.eligibility[s] *= self.gamma * self.eta
            module_i.eligibility[s] += 1
            td_error = r + self.gamma * module_i.state_values[s_] - module_i.state_values[s]
            # td_error = state_error
            module_i.state_values[s] += module_i.responsibility * td_error * self.lr # * module_i.eligibility[s]

            # update state-action value in the q_table
            module_i.update_state_actions(s, a)
            action_count_dict = module_i.state_action_count[s]  # get the action list in this state
            for action_i in action_count_dict.keys():
                state_action_str = s + str(action_i)
                latter_value = 0.0
                state_action_state_dict = module_i.state_action_state_pro[state_action_str]
                sum_probability = sum((state_action_state_dict.values()))
                for next_state_i in state_action_state_dict.keys():
                    latter_value += state_action_state_dict[next_state_i]/sum_probability * module_i.state_values[next_state_i]
                module_i.q_table.loc[s, action_i] = module_i.state_action_reward_value[state_action_str] + self.gamma * latter_value

    def show_modules(self):
        popWindow = tk.Tk()
        popWindow.geometry("{}x{}+{}+{}".format(window_width_plus, window_height_plus, 100, 100))
        popWindow.title("Results")
        cvs = tk.Canvas(popWindow, bg='white', height=window_height_plus, width=window_width_plus)
        # cvs.place(x=INTERVAL, y=INTERVAL)
        cvs.pack()

        for module_i in self.modules:
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

class Module:
    def __init__(self, serial, actions, n_modules):
        self.serial = serial
        self.actions = actions
        self.method = 'softmax'  # epsilon-greedy
        self.epsilon = 0.9
        self.lr = 0.1
        self.gamma = 0.9
        self.beta = 1
        self.task = ''
        self.mark = 'single MBRL '

        self.responsibility = 1/n_modules
        self.responsibility_predictor = 1/n_modules
        self.responsibility_previous = 1/n_modules
        self.selected_times = 0  # modules are selected times

        self.state_action_count = {}
        self.state_action_pro = {}
        self.state_action_state_count = {}
        self.state_action_state_pro = {}
        self.state_action_reward_count = {}
        self.state_action_reward_value = {}
        self.state_values = {}
        self.eligibility = {}

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # self.initialize_module()

    def initialize_module(self):
        # the initialization of state_value, state_action_value, and state_action_n_state_probability
        for state_i in all_locations:
            self.state_values[state_i] = 0.0
            for action_i in self.actions:
                state_action_str = state_i+str(action_i)
                self.state_action_reward[state_action_str] = 0.0
                self.state_action_state_pro[state_action_str] = {}
                for next_state_i in all_locations:
                    self.state_action_state_pro[state_action_str][next_state_i] = 1/len(all_locations)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0.]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if observation not in self.state_values.keys():
            self.state_values[observation] = 0.0

        state_action = self.q_table.loc[observation, :]
        if self.method == 'epsilon-greedy':
            if np.random.uniform() < self.epsilon:
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                action = np.random.choice(self.actions)
        elif self.method == 'greedy':
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        elif self.method == 'stochastic-greedy':
            pi = np.exp(self.beta * np.array(state_action)) / sum(np.exp(self.beta * np.array(state_action)))
            action = np.random.choice(self.actions, p=pi)
        elif self.method == 'softmax':
            # ?testici, a problem in softmax is that the exp value will be very
            # large if the state-action value is big, then this function will not work
            pi = np.exp(np.array(state_action)) / sum(np.exp(np.array(state_action)))
            action = np.random.choice(self.actions, p=pi)
        return action

    def update_dicts(self, s, a, r, s_):
        self.update_state_actions(s, a)
        self.update_state_action_state(s, a, s_)  # P{s_|s,a}
        self.update_state_action_reward(s, a, r)  # R{r|s,a}

        # compute state_actin_value
        self.update_state_action_value(s,a)
        # compute state_value
        self.update_state_value(s)

        self.check_update_functions()

    def update_state_actions(self, s, a):  # the action_list under the state of s
        # { state : {action1: count1, action2:count2} }
        if s not in self.state_action_count.keys():
            self.state_action_count[s] = {}
            self.state_action_count[s][a] = 1
        else:
            if a not in self.state_action_count[s].keys():
                self.state_action_count[s][a] = 1
            else:
                self.state_action_count[s][a] += 1

        # compute the probabilities
        # { state : {action1: pro1, action2:pro2} }
        if s not in self.state_action_pro.keys():
            self.state_action_pro[s] = {}
            self.state_action_pro[s][a] = 1.0
        else:
            if a not in self.state_action_pro[s].keys():
                self.state_action_pro[s][a] = 1.0

        # start to update state_action_pro
        action_count_dict = self.state_action_count[s]
        count_sum = sum(action_count_dict.values())
        for action_i in action_count_dict.keys():
            if action_i not in self.state_action_pro[s].keys():
                print('warning!!!, action_i of state_action_count not exists in state_action_pro')
            action_i_count = action_count_dict[action_i]
            self.state_action_pro[s][action_i] = action_i_count / count_sum

    def update_state_action_state(self, s, a, s_):
        # { state_action: {state1: count1, state2:count2} }
        key = s + str(a)
        if key not in self.state_action_state_count.keys():
            self.state_action_state_count[key] = {}
            self.state_action_state_count[key][s_] = 1
        else:
            if s_ not in self.state_action_state_count[key].keys():
                self.state_action_state_count[key][s_] = 1
            else:
                self.state_action_state_count[key][s_] += 1

        # { state_action: {state1: pro1, state2:pro2} }
        if key not in self.state_action_state_pro.keys():
            self.state_action_state_pro[key] = {}
            self.state_action_state_pro[key][s_] = 1.0
        else:
            if s_ not in self.state_action_state_pro[key].keys():
                self.state_action_state_pro[key][s_] = 1.0
        state_count_dict = self.state_action_state_count[key]
        count_sum = sum(state_count_dict.values())
        for next_state_i in state_count_dict.keys():
            if next_state_i not in self.state_action_state_pro[key].keys():
                print('warning!!!, next_state_i of state_action_state_count not exists in state_action_state_pro')
            next_state_i_count = state_count_dict[next_state_i]
            self.state_action_state_pro[key][next_state_i] = next_state_i_count / count_sum

    def update_state_action_reward(self, s, a, r):
        # state_action: {reward1: count1, reward2:count2}
        key = s + str(a)
        if key not in self.state_action_reward_count.keys():
            self.state_action_reward_count[key] = {}
            self.state_action_reward_count[key][r] = 1
        else:
            if r not in self.state_action_reward_count[key].keys():
                self.state_action_reward_count[key][r] = 1
            else:
                self.state_action_reward_count[key][r] += 1
        reward_count = self.state_action_reward_count[key]
        count_sum = sum(reward_count.values())
        reward = sum(np.array(list(reward_count.keys())) * np.array(list(reward_count.values())) / count_sum)
        self.state_action_reward_value[key] = reward

    def update_state_action_value(self, s, a):
        key = s + str(a)
        later_value = 0.0
        for next_state_i in self.state_action_state_pro[key].keys():
            later_value_i = self.state_action_state_pro[key][next_state_i] * self.state_values[next_state_i]
            later_value += later_value_i
        self.q_table.loc[s,a] = self.state_action_reward_value[key] + self.gamma*later_value

    def update_state_value(self, s):
        state_value = 0.0
        state_action_list = self.state_action_pro[s]
        for action_i in state_action_list.keys():
            state_value_i = 0.0
            if action_i in state_action_list.keys():
                state_value_i = self.state_action_pro[s][action_i] * self.q_table.loc[s, action_i]
            state_value += state_value_i
        self.state_values[s] = state_value

    def check_update_functions(self):
        if self.state_action_state_pro.keys() != self.state_action_state_count.keys():
            print('warning!!! \nfunction of state_action_state')
        if self.state_action_reward_count.keys() != self.state_action_reward_value.keys():
            print('warning!!! \nfunction of state_action_reward')

    def learn_from_error(self, s, a, r, s_):
        key = s + str(a)
        self.check_state_exist(s_)
        if s_ not in self.state_values.keys():
            self.state_values[s_] = 0
        if key not in self.state_action_reward_value.keys():
            self.state_action_reward_value[key] = 0.0
        if key not in self.state_action_state_pro.keys():
            self.state_action_state_pro[key] = {}
            self.state_action_state_pro[key][s_] = np.random.uniform()
        else:
            if s_ not in self.state_action_state_pro[key].keys():
                self.state_action_state_pro[key][s_] = np.random.uniform()

        sum_probability_i = sum(self.state_action_state_pro[key].values())
        for next_state_i in self.state_action_state_pro[key].keys():
            # get the p(x'|x,a)
            predict_probability_i = self.state_action_state_pro[key][next_state_i] / sum_probability_i
            c_j_x = 0
            if next_state_i == s_:
                c_j_x = 1
            predict_error_i = self.state_action_state_pro[key][next_state_i] - c_j_x
            if predict_error_i > 0:
                self.state_action_state_pro[key][next_state_i] -= self.lr * predict_error_i
            else:
                self.state_action_state_pro[key][next_state_i] += self.lr * abs(predict_error_i)

        reward_error = r - self.state_action_reward_value[key]
        self.state_action_reward_value[key] += self.lr * reward_error

        td_error = r + self.gamma * self.state_values[s_] - self.state_values[s]
        self.state_values[s] += self.lr * td_error

        # update state-action values in the q_table
        self.update_state_actions(s, a)
        action_count_dict = self.state_action_count[s]
        for action_i in action_count_dict.keys():
            state_action_str = s+str(action_i)
            latter_value = 0.0
            state_action_state_dict = self.state_action_state_pro[state_action_str]
            for next_state_i in state_action_state_dict.keys():
                latter_value += state_action_state_dict[next_state_i] * self.state_values[next_state_i]  # p*v(s')
            self.q_table.loc[s, action_i] = self.state_action_reward_value[state_action_str] + self.gamma * latter_value


"""
class MMBRL_single_Doya_error_update:
    def __init__(self, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, method='epsilon-greedy', beta=1):
        self.mark = 'MMBRL single'
        self.beta = beta

        self.action_space = ['north', 'south', 'east', 'west', 'stay']
        self.n_actions = len(self.action_space)
        self.actions = list(range(self.n_actions))

        self.n_states = 49
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.method = method

        self.state_action_count = {}
        self.state_action_pro = {}
        self.all_locaitons = all_locations

        self.state_action_reward_count = {}
        self.state_action_reward = {}

        self.state_action_state_count = {}
        self.state_action_state_pro = {}
        self.state_values = {}

        # self.initialization_values()  # the state_value, state_action_reward, state_action_n_state_pro

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # state-all_actions_values: 0,1,2,3,4

        self.n_module = N_MODULE
        self.responsibility_prediction = 1/self.n_module
        self.responsibility_signal = 1/self.n_module

    def initialization_values(self):
        # dynamic model
        for state_i in self.all_locaitons:
            # state value function
            self.state_values[state_i] = 0.0
            for action_i in self.actions:
                state_action_str = state_i+str(action_i)
                # reward model
                self.state_action_reward[state_action_str] = 0.0
                # dynamic model
                self.state_action_state_pro[state_action_str] = {}
                for next_state_i in self.all_locaitons:
                    self.state_action_state_pro[state_action_str][next_state_i] = 1/len(all_locations)

    def choose_action(self, observation):  # str
        self.check_state_exist(observation)
        if observation not in self.state_values.keys():
            self.state_values[observation] = 0.0

        state_action = self.q_table.loc[observation, :]
        if self.method == 'epsilon-greedy':
            if np.random.uniform() < self.epsilon:
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                action = np.random.choice(self.actions)
        elif self.method == 'greedy':
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        elif self.method == 'softmax':
            pi = np.exp(self.beta * np.array(state_action)) / sum(np.exp(self.beta * np.array(state_action)))
            action = np.random.choice(self.actions, p=pi)
        return action

    def update_state_action_value(self, s, a):
        key = s + str(a)
        later_value = 0.0
        for next_state_i in self.state_action_state_pro[key].keys():
            later_value_i = self.state_action_state_pro[key][next_state_i] * self.state_values[next_state_i]
            later_value += later_value_i
        self.q_table.loc[s, a] = self.state_action_reward[key] + self.gamma*later_value

    def update_state_value(self, s):
        state_value = 0.0
        state_action_list = self.state_action_pro[s]
        for action_i in state_action_list.keys():
            state_value_i = 0.0
            if action_i in state_action_list.keys():
                state_value_i = self.state_action_pro[s][action_i] * self.q_table.loc[s, action_i]
            state_value += state_value_i
        self.state_values[s] = state_value

    def update_state_actions(self, s, a):
        # update state action counts
        # { state : {action1: count1, action2:count2} }
        if s not in self.state_action_count.keys():
            self.state_action_count[s] = {}
            self.state_action_count[s][a] = 1
        else:
            if a not in self.state_action_count[s].keys():
                self.state_action_count[s][a] = 1
            else:
                self.state_action_count[s][a] += 1

        # compute the probabilities
        # { state : {action1: pro1, action2:pro2} }
        if s not in self.state_action_pro.keys():
            self.state_action_pro[s] = {}
            self.state_action_pro[s][a] = 1.0
        else:
            if a not in self.state_action_pro[s].keys():
                self.state_action_pro[s][a] = 1.0

        # start to update state_action_pro
        action_count_dict = self.state_action_count[s]
        count_sum = sum(action_count_dict.values())
        for action_i in action_count_dict.keys():
            if action_i not in self.state_action_pro[s].keys():
                print('warning!!!, action_i not exists in state_action_pro')
            action_i_count = action_count_dict[action_i]
            self.state_action_pro[s][action_i] = action_i_count / count_sum

    def update_state_action_reward(self, s, a, r):
        # state_action: {reward1: count1, reward2:count2}
        key = s + str(a)
        if key not in self.state_action_reward_count.keys():
            self.state_action_reward_count[key] = {}
            self.state_action_reward_count[key][r] = 1
        else:
            if r not in self.state_action_reward_count[key].keys():
                self.state_action_reward_count[key][r] = 1
            else:
                self.state_action_reward_count[key][r] += 1
        reward_count = self.state_action_reward_count[key]
        count_sum = sum(reward_count.values())
        reward = sum(np.array(list(reward_count.keys())) * np.array(list(reward_count.values())) / count_sum)
        self.state_action_reward[key] = reward

    def update_state_action_state(self, s, a, s_):
        # { state_action: {state1: count1, state2:count2} }
        key = s + str(a)
        if key not in self.state_action_state_count.keys():
            self.state_action_state_count[key] = {}
            self.state_action_state_count[key][s_] = 1
        else:
            if s_ not in self.state_action_state_count[key].keys():
                self.state_action_state_count[key][s_] = 1
            else:
                self.state_action_state_count[key][s_] += 1

        # { state_action: {state1: pro1, state2:pro2} }
        if key not in self.state_action_state_pro.keys():
            self.state_action_state_pro[key] = {}
            self.state_action_state_pro[key][s_] = 1.0
        else:
            if s_ not in self.state_action_state_pro[key].keys():
                self.state_action_state_pro[key][s_] = 1.0

        state_count_dict = self.state_action_state_count[key]
        count_sum = sum(state_count_dict.values())
        for next_state_i in state_count_dict.keys():
            if next_state_i not in self.state_action_state_pro[key].keys():
                print('warning!!!, next_state_i not exists in state_action_state_pro')
            next_state_i_count = state_count_dict[next_state_i]
            self.state_action_state_pro[key][next_state_i] = next_state_i_count / count_sum

        if self.state_action_state_count.keys() != self.state_action_state_pro.keys():
            print(len(self.state_action_state_count.keys()))
            print(len(self.state_action_state_pro.keys()))
            print('there is a problem in the fucntion of update_state_action_state')

    def update_dicts(self, s, a, r, s_):
        if s_ not in self.state_values.keys():
            self.state_values[s_] = 0

        self.update_state_actions(s, a)
        self.update_state_action_state(s, a, s_)  # P{s_|s,a}
        self.update_state_action_reward(s, a, r)  # R{r|s,a}

        self.update_state_action_value(s,a)  # Q(s,a)
        self.update_state_value(s)           # V(s)

        self.check_update_functions()

    def check_update_functions(self):
        if self.state_action_state_pro.keys() != self.state_action_state_count.keys():
            print('warning!!!  function of state_action_state')
        if self.state_action_reward_count.keys() != self.state_action_reward.keys():
            print('warning!!! function of state_action_reward')

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        key = s + str(a)

        # update dynamic model
        for state_i in self.all_locaitons:
            for action_i in self.actions:
                state_action_str = state_i+str(action_i)
                for next_state_i in self.all_locaitons:
                    c_j_x = 0
                    if next_state_i == s_:
                        c_j_x = 1
                    predict_error_i = self.state_action_state_pro[state_action_str][next_state_i] - c_j_x
                    if predict_error_i > 0:
                        self.state_action_state_pro[state_action_str][next_state_i] -= self.lr * predict_error_i
                    else:
                        self.state_action_state_pro[state_action_str][next_state_i] += self.lr * abs(predict_error_i)

        reward_error = r - self.state_action_reward[key]
        self.state_action_reward[key] += self.lr * reward_error

        td_error = r + self.gamma*self.state_values[s_] - self.state_values[s]
        self.state_values[s] += self.lr * td_error

        # update state-action values in the q_table
        latter_value = 0.0
        state_action_state_dict = self.state_action_state_pro[key]
        for next_state_i in state_action_state_dict.keys():
            latter_value += state_action_state_dict[next_state_i] * self.state_values[next_state_i]  # p*v(s')
        self.q_table.loc[s, a] = self.state_action_reward[key] + self.gamma * latter_value



    # (str(observation), action, reward, str(observation_))
    def learn_from_error(self, s, a, r, s_):
        self.check_state_exist(s_)

        key = s + str(a)
        if key not in self.state_action_state_pro.keys():
            self.state_action_state_pro[key] = {}
            self.state_action_state_pro[key][s_] = 1.0
        else:
            if s_ not in self.state_action_state_pro[key].keys():
                self.state_action_state_pro[key][s_] = 0.0

        for next_state_i in self.state_action_state_pro[key].keys():
            c_j_x = 0
            if next_state_i == s_:
                c_j_x = 1
            predict_error_i = self.state_action_state_pro[key][next_state_i] - c_j_x
            if predict_error_i > 0:
                self.state_action_state_pro[key][next_state_i] -= self.lr * predict_error_i
            else:
                self.state_action_state_pro[key][next_state_i] += self.lr * abs(predict_error_i)

        if key not in self.state_action_reward.keys():
            self.state_action_reward[key] = 0.0
        reward_error = r - self.state_action_reward[key]
        self.state_action_reward[key] += self.lr * reward_error

        if s_ not in self.state_values.keys():
            self.state_values[s_] = 0
        td_error = r + self.gamma * self.state_values[s_] - self.state_values[s]
        self.state_values[s] += self.lr * td_error

        # update state-action values in the q_table
        self.update_state_actions(s, a)
        action_count_dict = self.state_action_count[s]
        for action_i in action_count_dict.keys():
            state_action_str = s+str(action_i)
            latter_value = 0.0
            state_action_state_dict = self.state_action_state_pro[state_action_str]
            for next_state_i in state_action_state_dict.keys():
                latter_value += state_action_state_dict[next_state_i] * self.state_values[next_state_i]  # p*v(s')
            self.q_table.loc[s, action_i] = self.state_action_reward[state_action_str] + self.gamma * latter_value








    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def plot_cost(self):
        plt.plot(np.arange(20), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def show_pcolor_sv(self):
        x = np.arange(0, 7, 1)
        y = np.arange(0, 7, 1)
        values = np.zeros([7, 7])

        # {state: {aciton1:count, action2:count2}}
        state_list = self.state_action_counts.keys()
        for state_i in state_list:
            if state_i != 'terminal':
                action_count_dict = self.state_action_counts[state_i]
                print('action count dict', action_count_dict)
                count_sum = sum(action_count_dict.values())
                state_value = 0.0
                for action in action_count_dict.keys():
                    print('state, aciton', state_i, action)
                    action_weight = action_count_dict[action] / count_sum
                    action_value = self.q_table.loc[state_i, action] * action_weight
                    state_value += action_value
                index_coords = ast.literal_eval(state_i)
                index_value = state_value
                x_coord = int(index_coords[0] / UNIT)
                y_coord = 6 - int(index_coords[1] / UNIT)
                values[y_coord][x_coord] = index_value
        plt.title('State values')
        im = plt.pcolormesh(x, y, values, vmin=np.min(values), vmax=np.max(values), shading='auto')
        plt.colorbar(im)
        plt.show()
"""


