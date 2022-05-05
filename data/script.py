import os

import matplotlib.pyplot as plt



filelist = {}
currentpath = os.getcwd()
for file in os.listdir(os.getcwd()):
    file_strings = file.split('.')
    if len(file_strings) > 1:
        if file_strings[1] == 'txt':
            print(file)
            filelist[file_strings[0]] = file

width = 1000
index_i = 0

linestyle_list = ['-', '--', '-.', ':', 'solid']
print(filelist.keys())
style_index = 0
for key in filelist.keys():
    filename = filelist[key]
    f = open(filename)
    line = f.readline()
    x_axis_data = []
    datalist = []
    data_index = 0
    while line:
        line_values = line.split(',')
        step_values = line_values[0]
        step = float(step_values)
        if data_index % 20 == 0:
            x_axis_data.append(data_index)
            datalist.append(step)
        data_index += 1
        line = f.readline()
    f.close()
    # print(linestyle_list[style_index])
    plt.plot(x_axis_data, datalist, label=key, linestyle=linestyle_list[style_index])
    style_index += 1
    plt.xlabel('Number of trial')
    #plt.ylabel('Number of steps in a trial')
    #plt.ylabel('Averaged reward')
    plt.ylabel('Total value of states')




plt.legend()
plt.show()


