# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:10:00 2020

@author: Jens Ringsholm
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functions import fitexp, fit2exp, fitLin, fit2dpol, fit3dpol, plot_multi

from scipy import stats
from datetime import timedelta

plt.close('all')


#import matplotlib.style as style
#style.use('fivethirtyeight')

#style.use('seaborn-bright')




############ import data

val = pd.read_csv('data/valve_data.csv', index_col = 'rt', parse_dates=True)


########## design columns

val = val.dropna()

val =val[(np.abs(stats.zscore(val)) < 30).all(axis=1)]

############# test if and remove duplicates

val = val[~val.index.duplicated()]


############# find peaks


from scipy.signal import argrelextrema
n = 15

val['min'] = val.iloc[argrelextrema(val.Pressure.values, np.less_equal, order=n)[0]]['Pressure']
val['max'] = val.iloc[argrelextrema(val.Pressure.values, np.greater_equal, order=n)[0]]['Pressure']




fig1, ax1 = plt.subplots(figsize=(12, 6))         

val['Pressure'].plot()
plt.scatter(val.index, val['max'], c='r')
plt.scatter(val.index, val['min'], c='g')

############## plot overview

fig2, axes = plt.subplots(nrows=1, ncols=len(val.columns), figsize = (10, 8))
val.plot(subplots=True, ax=axes)
plt.gcf().autofmt_xdate()
plt.tight_layout()



fig1, ax1 = plt.subplots(figsize=(12, 6))         
plt.scatter(val.index, val['min'], c='g')
plt.scatter(val.index, val['max'], c='r')

ax1.set_xlim(val.index[0],val.index[-1] )


fig1, ax2 = plt.subplots(figsize=(12, 6))         

non = val['min'].dropna()/val['min'].dropna()


non.rolling(2).sum().plot()



ax2.set_xlim(val.index[0],val.index[-1] )




'''


################# multiaxis plot

bir_multi = bir.drop(columns= ['pres','epoch'] )


fig1, axes = plt.subplots(figsize = (10, 8))
ax = plot_multi(bir_multi, title= 'BIR Plot')
plt.gcf().autofmt_xdate()
plt.tight_layout()


############### Head pla

head_pla_multi = head_pla.drop(columns= ['epoch', 'co2', 'temp_ext'] )

fig2, axes = plt.subplots(figsize = (10, 8))
ax = plot_multi(head_pla_multi, title= 'nr5 pla Plot')
plt.gcf().autofmt_xdate()
plt.tight_layout()


############### BN04

############### BN05






################## Import and clean SG_tilt ######################

##################################################################

tilt = pd.read_csv('data/tilt.csv', index_col = 'Timepoint', parse_dates=True)

# remove empty 
tilt = tilt.dropna()
#tilt = tilt.drop(columns='Timepoint')


# align and cut time
tilt.index = tilt.index + timedelta(hours=-2)

# cut to relevant data
tilt = tilt['2020-05-09 15:00:00':]

######### plot for inspection ################################

if inspect == True:
    fig2, axes = plt.subplots(nrows=2, ncols=1, figsize = (10, 8))
    tilt.plot(subplots=True, ax=axes)
    plt.tight_layout()
    fig2.savefig('figs\Tilt_view.png')

################## Import and clean SG_Platoo ######################

pla = pd.read_csv('data/Plaato_SG.csv',index_col = 'epoch', parse_dates=True)

pla.index=(pd.to_datetime(pla.index,unit='ms'))

BPM = pd.read_csv('data/Plaato_BPM.csv',index_col = 'epoch', parse_dates=True)

BPM.index=(pd.to_datetime(BPM.index,unit='ms'))

Ptemp = pd.read_csv('data/Plaato_Temp.csv',index_col = 'epoch', parse_dates=True)

Ptemp.index=(pd.to_datetime(Ptemp.index,unit='ms'))


# refine and create one df

pla  = pla.drop(columns=['lol'])
pla['BPM'] = BPM['BPM']
pla['Temp'] = Ptemp['Temp']

pla = pla['2020-05-08 12:00:00':'2020-05-09 15:00:00']


#pla.index = pla.index + timedelta(hours=1)


if inspect == True:
    fig2, axes = plt.subplots(nrows=3, ncols=1, figsize = (10, 8))
    pla.plot(subplots=True, ax=axes)
    plt.tight_layout()
    fig2.savefig('figs\Plaato_view.png')





fig8, ax2 = plt.subplots(figsize=(10, 8))

plottype = '-'

al = 1 

ax3 = ax2.twinx()
ax3.spines['right']
BIRFA = ax3.plot(bir['co2'], plottype ,  color = 'r', label = 'BIR CO2 [ppm]', alpha = al)
BIRfa = ax3.plot(bir['co2'].rolling(100).mean(), plottype ,  color = 'black',label = 'BIR CO2 [ppm] rolling average', alpha = al)

#BIRFA = ax3.plot(atm_bir['co2'].rolling(100).mean(), plottype ,  color = 'r', label = 'BN05 CO2 [ppm]', alpha = al)

PlaFA = ax2.plot(pla['BPM'], plottype,  color = 'b', label = 'Plaato BPM', alpha = al)

ax2.tick_params(axis='y', colors='b')


ax3.tick_params(axis='y', colors='r')
ax3.grid(None)


lns =  BIRFA + PlaFA + BIRfa
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=0,fancybox=True, framealpha=1)

plt.title('Fermentation Activity (FA)')


plt.gcf().autofmt_xdate()
plt.tight_layout()
fig8.savefig('figs/FA_plots.png')






######################## Resampling and Intepolation ################################

li = []

bir = bir.tz_localize(None)
pla = pla.tz_localize(None)
head_pla = head_pla.tz_localize(None)

#bar = bar.tz_localize(None)

head_pla = head_pla.drop(columns=['epoch', 'co2'])


pla['SG-plaat'] = pla['SG']
pla['Temp-plaat'] = pla['Temp']
pla = pla.drop(columns=['SG', 'Temp'])
bir = bir.drop(columns=['epoch', 'pres', 'temp_ext'])



tilt['SG-tilt'] = tilt['SG']
tilt['Temp-tilt'] = tilt['Temp']


tilt = tilt.drop(columns=['SG', 'Temp'])


li.append(pla)
li.append(bir)
li.append(head_pla)
li.append(tilt)



df_mv = pd.concat(li, axis=0, ignore_index=False, sort=True)

df_int = df_mv.resample('0.1min').mean().interpolate().dropna()


fig11, axes = plt.subplots(nrows=2, ncols=4, figsize = (14, 9))
df_int.plot(subplots=True, ax=axes)
plt.tight_layout()
plt.savefig('figs\InterpolatedOverview.png')


######################### write csv ###################



df_int['dates'] = df_int.index

df_int.to_csv(r'data/df_int.csv', index=False)



###################### correlation matrix 2##################


f = plt.figure(figsize=(10, 8))
plt.matshow(df_int.corr(), fignum=f.number, cmap='rocket')
plt.xticks(range(df_int.shape[1]), df_int.columns, fontsize=14, rotation=45, )
plt.yticks(range(df_int.shape[1]), df_int.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.savefig('figs\correlationMatrix.png')




################### fits

df_int = df_int[20:1400]

df_int['rol'] = df_int['co2'].expanding(1).sum()


fitLin(df_int['rol'], df_int['SG-plaat'], 1 , True, 'BIR vs Plaato fit', (10, 8))
'''