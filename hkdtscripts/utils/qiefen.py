import os
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import glob
from tsmoothie.smoother import *
import time
import sys

# directoryPath = "C:/Users/chais/Matlab_files/Elevator EMSD/Analysis elevator/tsEMSDAB001/"
# directoryPath = "C:/Users/chais/Matlab_files/Elevator EMSD/Analysis elevator/New task sequence comparison/ts20200605/"
# directoryPath = "M:/Dropbox/Projects/EMSD Elevator project/Phase1 - Subsequent works/ts_dataset/"
# directoryPath = "/Users/chaisongjian/Dropbox/Projects/EMSD Elevator project/Phase1 - Subsequent works/ts_dataset"
# directoryPath = "E:/EMSDPOWER_202003/"
#directoryPath = "D:/EMSDPOWER_202003/"
directoryPath = "/home/gyj/Desktop/qiefen/EMSDAB013/EMSDAB013_202105/"
#directoryPath = "C:/test/"

os.chdir(directoryPath)
xx = 0
yy = 0
# %%

###################################################
############## Data preprocessing ################
###################################################
CarSeg_list = []
CarDur_list = []
DoorSeg_list = []
DoorDur_list = []
Seg_list = []
Duration_list = []

progress = 0  # progress flag
nfiles = len(glob.glob(directoryPath + '*.csv'))


for file_name in glob.glob(directoryPath + '*.csv'):
    print(file_name)

    dataset_raw = pd.read_csv(file_name, engine='python')
    dataset = dataset_raw.copy()

    dataset['time'] = pd.to_datetime(dataset['time'])  # convert to datetime
    dataset['time'] = dataset['time'] + timedelta(hours=8)  # UTC+8
    dataset = dataset.set_index('time')  # set datetime-based index
    #dataset = dataset.rename(columns={'AI1': 'Door', 'AI2': 'Motor', 'AI3': 'Brake', 'AI4': 'Safety'})
    dataset = dataset.rename(columns={'AI1': 'Safety', 'AI2': 'Motor', 'AI3': 'Brake A', 'AI4': 'Door', 'AI5': 'Brake B'})
    dataset = dataset.drop(columns='name')
    dataset = dataset.drop(columns='RPS')
    dataset = dataset.drop(columns='SPEED')

    dataset.loc[dataset['DIST'] > 4500, 'DIST'] = np.nan
    dataset['DIST'] = dataset['DIST'].interpolate()


    dist = 0.01 * dataset['DIST']  # convert cm to m
    dist = pd.Series(dist).rolling(window=20).mean()  # moving average distance signal
    dataset['Dist_MA'] = dist
    velocity = dist.diff() / 0.05  # calculate the velocity
    dataset['Velocity'] = velocity.abs()

    smoother = KalmanSmoother(component='level_trend',
                              component_noise={'level': 0.1, 'trend': 0.1})




    # dataset['Brake'] = dataset['Brake'].replace(0.75,0)
    # dataset['Motor'] = dataset['Motor'].replace(0.9,0)
    ###################################################
    ############## Time series segment ################
    ###################################################

    seq_slice = dataset.loc[str(dataset.index.date[0]) + ' 07:00:00': str(dataset.index.date[0]) + ' 20:00:00']

    print('KF start')
    smoother.smooth(seq_slice[['Motor', 'Brake A', 'Door', 'Brake B']].T)

    new_seq = smoother.smooth_data.T
    seq_slice['Motor_KF'] = new_seq[:, 0]
    seq_slice['Brake A_KF'] = new_seq[:, 1]
    seq_slice['Door_KF'] = new_seq[:, 2]
    seq_slice['Brake B_KF'] = new_seq[:, 3]
    print('KF over')

    seq_slice.loc[seq_slice['Brake A_KF'] < 0.05, 'Brake A_KF'] = 0
    seq_slice.loc[seq_slice['Brake B_KF'] < 0.05, 'Brake B_KF'] = 0

    seq_slice = seq_slice.reset_index()
    seq_slice = seq_slice.rename(columns={'index': 'time'})

    ind_start = 0
    ind_end = 0
    ind_prestart = 0

    i = 0
    while i < (len(seq_slice) - 120):

        # print(seq_slice.loc[i, 'time'])

        if (seq_slice.loc[i, 'Brake A_KF'] == 0) & (seq_slice.loc[(i + 1):(i + 120), 'Brake A_KF'] > 0).all():
            #        seg_ind[i-10,0] = i # cycle starting index
            seq_slice.loc[i, 'seg_flag'] = 1  # use 1 to mark cycle starting point
            ind_start = i

            DoorSeg_list.append(seq_slice.iloc[ind_end:ind_start])  # Door motion cycle list
            DoorDur_list.append(
                seq_slice.loc[ind_start, 'time'] - seq_slice.loc[ind_end, 'time'])  # Door motion duration list

            Seg_list.append(seq_slice.iloc[ind_prestart:ind_start])  # The whole operational cycle
            Duration_list.append(seq_slice.loc[ind_start, 'time'] - seq_slice.loc[
                ind_prestart, 'time'])  # The whole operational cycle duration list

            if abs(Duration_list[-1].total_seconds() - Seg_list[-1].shape[0] / 20) > 1:  # remove the segments with loss data
                del (Seg_list[-1])
                del (Duration_list[-1])
                del (DoorSeg_list[-1])
                del (DoorDur_list[-1])
                del (CarSeg_list[-1])
                del (CarDur_list[-1])
                xx = xx+1

        elif (seq_slice.loc[(i - 120):(i - 1), 'Brake A_KF'] > 0).all() & (seq_slice.loc[i, 'Brake A_KF'] == 0):

            seq_slice.loc[i + 1, 'seg_flag'] = 2  # use 2 to mark cycle ending point
            ind_end = i + 1

            CarSeg_list.append(seq_slice.iloc[ind_start:ind_end])  # lift car traveling cycle list
            CarDur_list.append(
                seq_slice.loc[ind_end, 'time'] - seq_slice.loc[ind_start, 'time'])  # car traveling duration list
            ind_prestart = ind_start

        i += 1

    if ind_start == ind_prestart:
        DoorSeg_list.append(seq_slice.iloc[ind_end:i])  # Door motion cycle list
        DoorDur_list.append(seq_slice.loc[i, 'time'] - seq_slice.loc[ind_end, 'time'])  # Door motion duration list
        Seg_list.append(seq_slice.iloc[ind_prestart:i])  # The whole operational cycle
        Duration_list.append(
            seq_slice.loc[i, 'time'] - seq_slice.loc[ind_prestart, 'time'])  # The whole operational cycle duration list


# %%
###################################################################
################ Remove the segments with loss ####################
###################################################################


# i = 0
# # iterate over the list
# while i < len(seg_list):
#     # check if element begins with 'd'
#     if abs(duration_list[i].total_seconds() - seg_list[i].shape[0] / 20 ) > 1:
#         # remove it
#         del(duration_list[i])
#         del(seg_list[i])
#     else:
#         i += 1

# %%

###################################################
########### Time series visulization ##############
###################################################
def sequence_plot(sequence_slice, fig_L, fig_W, fontsize, linewidth):
    sns.set(rc={'figure.figsize': (fig_L, fig_W)})
    sns.set(font_scale=fontsize)
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    lns1 = ax1.plot(sequence_slice['Door'], linestyle='-',
                    linewidth=linewidth, color='#1F77B4', label='Door')
    lns2 = ax1.plot(sequence_slice['Brake A'], linestyle='-',
                    linewidth=linewidth, color='#2C9F2C', label='Brake A')
    lns3 = ax1.plot(sequence_slice['Safety'], linestyle='-',
                    linewidth=linewidth, color='#9467BD', label='Safety')
    lns5 = ax1.plot(sequence_slice['Brake B'], linestyle='-',
                    linewidth=linewidth, color='#b45c1f', label='Brake B')
    # ax1.grid(False)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Door/Brake/Safety Current (A)', color=color)
    # ax1.set_ylim([0,2.0])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    lns4 = ax2.plot(sequence_slice['Motor'], linestyle='-',
                    linewidth=linewidth, color=color, label='Motor')
    ax2.grid(False)
    ax2.set_ylabel('Motor Current (A)', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_ylim([0,35])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Set x-axis major ticks to weekly interval, on Mondays
    # ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    # Format x-tick labels as %H:%m:%s
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns1 + lns2 + lns3 + lns5 + lns4
    labs = [l.get_label() for l in lns]
    #    ax1.legend(lns, labs, bbox_to_anchor=(1.05, 1.0), loc='upper left',
    #               fancybox=True, framealpha=1,shadow=True, borderpad=1)
    ax1.legend(lns, labs, ncol=5, loc='upper center')

    time_sequence = sequence_slice.index[-1] - sequence_slice.index[0]

    plt.title(str(sequence_slice.index[0]) +
              '---' + str(sequence_slice.index[-1]) + '  [  ' + str(time_sequence) + '  ]')


# %%

fig_L = 12
fig_W = 5
fontsize = 1.5
linewidth = 2
#sequence_slice = seq_slice
sequence_slice = Seg_list[1]
# sequence_slice = Seg_list[10901]
# sequence_slice = dataset.loc['2020-03-03 11:37:15':'2020-03-19 11:37:50']

sequence_slice['time'] = pd.to_datetime(sequence_slice['time'])  # convert to datetime
sequence_slice = sequence_slice.set_index('time')
sequence_plot(sequence_slice, fig_L, fig_W, fontsize, linewidth)

# del(DoorDur_list[7706])

# %%
###################################################
################ Save the list ####################
###################################################


################ Remove the CarSegment with one record ####################

i = 0
# iterate over the list
while i < len(CarSeg_list):
    # check if element begins with 'd'
    if len(CarSeg_list[i]) == 1:
        # remove it
        del (Seg_list[i])
        del (Duration_list[i])
        del (DoorSeg_list[i])
        del (DoorDur_list[i])
        del (CarSeg_list[i])
        del (CarDur_list[i])
        yy = yy+1

    else:
        i += 1

import pickle

with open('/home/gyj/Desktop/qiefen/EMSDAB013/EMSDAB013_202105/KF_AB013SegList_202105', 'wb') as f:
    pickle.dump(Seg_list, f)

with open('/home/gyj/Desktop/qiefen/EMSDAB013/EMSDAB013_202105/KF_AB013DurationList_202105', 'wb') as f:
    pickle.dump(Duration_list, f)

with open('/home/gyj/Desktop/qiefen/EMSDAB013/EMSDAB013_202105/KF_AB013CarSegList_202105', 'wb') as f:
    pickle.dump(CarSeg_list, f)

with open('/home/gyj/Desktop/qiefen/EMSDAB013/EMSDAB013_202105/KF_AB013CarDurList_202105', 'wb') as f:
    pickle.dump(CarDur_list, f)

with open('/home/gyj/Desktop/qiefen/EMSDAB013/EMSDAB013_202105/KF_AB013DoorSegList_202105', 'wb') as f:
    pickle.dump(DoorSeg_list, f)

with open('/home/gyj/Desktop/qiefen/EMSDAB013/EMSDAB013_202105/KF_AB013DoorDurList_202105', 'wb') as f:
    pickle.dump(DoorDur_list, f)

print('finish')

# with open('D:/Backup/桌面/data/NewEMSDAB013/EMSDAB013_202105/KF_AB013SegList_202105', 'wb') as f:
#     pickle.dump(Seg_list, f)
#
# with open('D:/Backup/桌面/data/NewEMSDAB013/EMSDAB013_202105/KF_AB013DurationList_202105', 'wb') as f:
#     pickle.dump(Duration_list, f)
#
# with open('D:/Backup/桌面/data/NewEMSDAB013/EMSDAB013_202105/KF_AB013CarSegList_202105', 'wb') as f:
#     pickle.dump(CarSeg_list, f)
#
# with open('D:/Backup/桌面/data/NewEMSDAB013/EMSDAB013_202105/KF_AB013CarDurList_202105', 'wb') as f:
#     pickle.dump(CarDur_list, f)
#
# with open('D:/Backup/桌面/data/NewEMSDAB013/EMSDAB013_202105/KF_AB013DoorSegList_202105', 'wb') as f:
#     pickle.dump(DoorSeg_list, f)
#
# with open('D:/Backup/桌面/data/NewEMSDAB013/EMSDAB013_202105/KF_AB013DoorDurList_202105', 'wb') as f:
#     pickle.dump(DoorDur_list, f)

# os.chdir("D:/DT")
# os.chdir("C:/Users/chais/Spyder_files")
# aa = pd.read_pickle('DurationList_202003')
# aaa = pd.read_pickle('SegList_202003')

# sequence_slice = Seg_list[37]
# sequence_slice['time'] = pd.to_datetime(sequence_slice['time'])  # convert to datetime
# sequence_slice = sequence_slice.set_index('time')
# sequence_plot(sequence_slice, fig_L, fig_W, fontsize, linewidth)