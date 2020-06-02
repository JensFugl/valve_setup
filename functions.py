# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:57:10 2020

@author: Jens Ringsholm
"""



import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import csv
import matplotlib.dates as mdates


import seaborn as sns
sns.set_style("darkgrid" )
sns.set(rc={'axes.facecolor':'#eef1f4',})
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.color_palette("husl", 10)


def plot_multi(data,  title=None, cols=None, **kwargs):

    from pandas import plotting

    # Get default color style from pandas
    if cols is None: cols = data.columns
    save = True
    if cols is None: save = False
    if len(cols) == 0: return
    colors = getattr(getattr(plotting, '_matplotlib').style, '_get_standard_colors')(num_colors=len(cols))
    
    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.tick_params(axis='y', colors=colors[0 % len(colors)])
    plt.title(title)
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right']
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.tick_params(axis='y', colors=colors[n % len(colors)])
        ax_new.grid(None)
        
        # Legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc='best')
    plt.tight_layout()
    if save == True:
        plt.savefig('figs\{}.png'.format(title))
    return ax

def OpenVolumeData(filename):
    device = []
    water_w = []
    weight_b = []
    weight_a = []
    m_t = []
    ph = []
    rho_w = 0.9978
    with open(filename,"r") as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for curline in csvfile:
            if curline == 1:
                pass
            else:
                for row in plots: 
                    device.append(float(row[0]))
                    water_w.append(float(row[1])/100)
                    weight_b.append(float(row[2])/100)
                    weight_a.append(float(row[3])/100)
                    m_t.append(float(row[4])/100)
        V_l = np.array(water_w)/1000 * np.ones_like(water_w)*rho_w
        
    return V_l, m_t, ph


################# plot func #################

def plot_Brew(x, name):
    
    fig1, ax1 = plt.subplots(figsize=(12, 6)) 
    #hours = mdates.HourLocator(interval = 5)
    #h_fmt = mdates.DateFormatter('%H:%M')
    ax1.plot(x[0],  'tab:red',  label = 'Airlock')
    ax1.plot(x[1], 'tab:blue',  label = 'In-Tank')
    ax1.set_xlabel('Time (hh:mm)', size=16)
    ax1.set_ylabel(name, size=16)
    #plt.tick_params(axis='both', which='major', labelsize=14)
    #ax1.xaxis.set_major_locator(hours)
    #ax1.xaxis.set_major_formatter(h_fmt)
    #fig1.autofmt_xdate()
    ax1.legend(loc='best')
    ax1.grid(True)
    plt.tight_layout()
    #plt.title(name)


    

###################### Linear fit ####################################################

def fitLin(x, y, sx, save_plot, name, size):


    def lin(x, a, b):
        return a*x + b
    
    sg = np.ones_like(x)*sx
    init_vals = [1,1]
    best_vals, covar = curve_fit(lin, x, y, sigma=sg, p0=init_vals, absolute_sigma = True)

    variance = np.cov(x,y)
    mean = np.mean(y)
    sd = np.std(y)
    
    residuals = y- lin(x, best_vals[0], best_vals[1])    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    
    
    fig1, ax1 = plt.subplots(figsize=size) 
    
    ax1.plot(x,y, '.', label='Data')
    ax1.plot(x, lin(x, best_vals[0],best_vals[1]),label=
                 'Fit: \n y = {0:8.2f}x + {1:8.2f} \n $R^2$: {2:8.8f}'
                 .format(best_vals[0],best_vals[1], r_squared))    
    ax1.legend()
    plt.title(name)    
    plt.tight_layout()
    
    if save_plot == True:
        fig1.savefig('figs/{}.png'.format(name))

    return best_vals



def fitexp(x, y, sx, save_plot, name, size):

    def exp(x, a, b, c):
        return a * np.exp(-b * x) + c

    sg = np.ones_like(x)*sx
    init_vals = [1, 1, 1]
    best_vals, covar = curve_fit(exp, x, y, sigma=sg, p0=init_vals, absolute_sigma = True)


    residuals = y- exp(x, best_vals[0], best_vals[1], best_vals[2])    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    
    
    fig1, ax1 = plt.subplots(figsize = size) 
    
    ax1.plot(x,y, '.')
    ax1.plot(x, exp(x, best_vals[0],best_vals[1], best_vals[2]),
             label='Fit: \n y = ${0:8.2f} * {1:8.2f}^x + {2:8.2f}$ \n $R^2$: {3:8.8f}'
                 .format(best_vals[0],best_vals[1], best_vals[2], r_squared))    
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    if save_plot == True:
        fig1.savefig('figs/{}.png'.format(name))
    
    return best_vals
    


def fit2exp(x, y, sx, save_plot, name, size):

    def exp2(x, a, b, c):
        return a**(b**x)+ c

    sg = np.ones_like(x)*sx
    init_vals = [1, 1, 1]
    best_vals, covar = curve_fit(exp2, x, y, sigma=sg, p0=init_vals, absolute_sigma = True)


    residuals = y- exp2(x, best_vals[0], best_vals[1], best_vals[2])    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    
    fig1, ax1 = plt.subplots(figsize = size) 
    
    ax1.plot(x,y, '.')
    ax1.plot(x, exp2(x, best_vals[0],best_vals[1], best_vals[2]),
             label='Fit: \n y = ${0:8.2f}^{1:8.2f}^x + {2:8.2f}$ \n $R^2$: {3:8.8f}'
                 .format(best_vals[0],best_vals[1], best_vals[2], r_squared))    
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    if save_plot == True:
        fig1.savefig('figs/{}.png'.format(name))
    
    return best_vals
    


def fit2dpol(x, y, sx, save_plot, name, size):

    def dpol(x, a, b, c):
        return a*x**2 + b*x + c

    sg = np.ones_like(x)*sx
    init_vals = [1, 1, 1]
    best_vals, covar = curve_fit(dpol, x, y, sigma=sg, p0=init_vals, absolute_sigma = True)


    residuals = y- dpol(x, best_vals[0], best_vals[1], best_vals[2])    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)    
    
    fig1, ax1 = plt.subplots(figsize = size) 
    
    ax1.plot(x,y, '.')
    ax1.plot(x, dpol(x, best_vals[0],best_vals[1], best_vals[2]),
             label='Fit: \n y = ${0:8.2f}*x^2 * {1:8.2f}^x + {2:8.2f}$ \n $R^2$: {3:8.8f}'
                 .format(best_vals[0],best_vals[1], best_vals[2], r_squared))    
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    if save_plot == True:
        fig1.savefig('figs/{}.png'.format(name))
    
    return best_vals


def fit3dpol(x, y, sx, save_plot, name, size):

    def d3pol(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    sg = np.ones_like(x)*sx
    init_vals = [1, 1, 1, 1]
    best_vals, covar = curve_fit(d3pol, x, y, sigma=sg, p0=init_vals, absolute_sigma = True)


    residuals = y- d3pol(x, best_vals[0], best_vals[1], best_vals[2], best_vals[3])    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)    
    
    fig1, ax1 = plt.subplots(figsize = size) 
    
    ax1.plot(x,y, '.')
    ax1.plot(x, d3pol(x, best_vals[0],best_vals[1], best_vals[2], best_vals[3]),
             label='Fit: \n y = ${0:8.2f}*x^3 {0:8.2f}*x^2 * {1:8.2f}^x + {2:8.2f}$ \n $R^2$: {3:8.8f}'
                 .format(best_vals[0],best_vals[1], best_vals[2], r_squared))    
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    if save_plot == True:
        fig1.savefig('figs/{}.png'.format(name))
    
    return best_vals


