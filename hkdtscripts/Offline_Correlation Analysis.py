# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:29:36 2023

@author: chais
"""


#%% Offline test main code

import os


import pandas as pd

# os.chdir("C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts")
# from utils_lift import *

os.chdir("C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/Utils/")
from utils_ErgatianLifts import *

os.chdir("C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts/")
from paras_lift import paras_QEBL8
# from utils_lift_QEBL8 import *
from utils_lift import *
# from devpy import run_14_QML3

import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf
#%% Load the dataset for single file (单个csv文件读取)



dataset = pd.read_csv('C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/DataFiles/QM L3_2023-01-06.csv',index_col = False)


dataset['Time'] = pd.to_datetime(dataset['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
dataset = dataset.astype({'Motor':'float','Brake':'float','Safety':'float','Door':'float','Resv-1':'float','Resv-2':'float','Resv-3':'float','Distance':'float'})

#%% select a period for analysis

seq = dataset.loc[(dataset['Time']>='2023-01-02 16:00:00') & (dataset['Time']<='2023-01-02 17:00:00')]

seq_var = seq[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3']]
#%%
############################################################# 
#################### Correlation paras ####################
#############################################################
# pg.corr(x=dataset['Safety'], y=dataset['Door'])
# pg.corr(x=seq['Motor'], y=seq['Brake'])
# pg.corr(x=dataset['Resv-1'], y=dataset['Brake'])
# pg.corr(x=dataset['Resv-1'], y=dataset['Motor'])


# calculate the half-hourly correlation coefficients between two vars 
corr_list = []

df_corr = pd.DataFrame({'Time':pd.date_range(start='2023-01-06 00:00', end='2023-01-06 23:30',freq='30min'), 
                   'Corr':0})
i = 0
for hour in range(24):
    df_hour = dataset[dataset['Time'].dt.hour == hour]

    for half in range(2):
        df_halfhour = df_hour[(df_hour['Time'].dt.minute >= half*30) & (df_hour['Time'].dt.minute < (half+1)*30)]
        # df_hour.to_csv(f"df_{hour}.csv")
        pearson_corr = df_halfhour['Safety'].corr(df_halfhour['Door'])
        # corr_list.append(pearson_corr)
        # print('hour ' + str(hour) + ' Pearson Corr is ' + str(pearson_corr))
        
        df_corr.loc[i, 'Corr'] = pearson_corr
        i = i+1
corr_list.append(df_corr)
# plt.plot(corr_list)
df_corr = pd.concat(corr_list)

df_corr['Time'] = pd.to_datetime(df_corr['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
df_corr = df_corr.set_index('Time')


#%%  画画脚本
sns.set(rc={'figure.figsize':(24, 5)})
sns.set(font_scale=1)  
fig, ax1 = plt.subplots()
lns0 = ax1.plot(df_corr['Corr'], linestyle='-', 
                linewidth=1.5, color='#0c0c0d')


#ax1.grid(False)
ax1.tick_params(axis='y')
ax1.set_ylabel('Pearson Correlation Coefficient')
# ax1.set_ylim([-0.3,0.4])
ax1.set_xlabel('Time')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))


# df_corr.plot()
# # plt.title('Half-hourly Pearson Corr Dynamics between [Safety] and [Door] over four successive days')
# plt.xlabel('Time')
# plt.ylabel('Pearson Correlation Coefficient')
# plt.ylim((-0.3,0.4))
# plt.show()
#%%
############################################################# 
#################### Scatter correlation ####################
#############################################################
# sns.set(style='white', font_scale=1.2)

# g = sns.JointGrid(data=df, x='POA(W/m^2)', y='GrossACPower', height=5)
# g = g.plot_joint(sns.regplot, color="xkcd:muted blue")
# g = g.plot_marginals(sns.distplot, kde=False, bins=12, color="xkcd:bluey grey")
# g.ax_joint.text(145, 1800, 'r = 0.979, p < .001', fontstyle='italic')
# plt.tight_layout()

# sns.set(rc={'figure.figsize':(10, 8)})
# sns.set(font_scale=1.5) 
# sns.lmplot(x="POA(W/m^2)", y="GrossACPower", data=df);
# plt.title("Scatter Plot with Linear fit")


sns.jointplot(x=seq["Safety"], y=seq["Door"],color='orange', kind='reg',
              joint_kws={'line_kws':{'color':'navy'}})


#%%
############################################################# 
####################### Pair plot ##########################
#############################################################
seq_pair = seq_var[['Brake','Resv-1']]

sns.set(rc={'figure.figsize':(20, 16)})
sns.set(font_scale=1.5) 
sns.pairplot(seq_pair)
#%%
############################################################# 
#################### Correlation matrix ####################
#############################################################
sns.set(rc={'figure.figsize':(10, 8)})
sns.set(font_scale=1.5)
sns.heatmap(seq.corr(), annot=True,fmt='.2f', linewidths=1,cmap="YlGnBu",)
