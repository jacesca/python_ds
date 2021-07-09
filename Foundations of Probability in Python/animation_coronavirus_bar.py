# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:31:09 2020

@author: jacesca@gmail.com
Source:
    https://towardsdatascience.com/bar-chart-race-in-python-with-matplotlib-8e687a5c8a41
"""

###############################################################################
## Importing libraries
###############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation

from IPython.display import HTML

###############################################################################
## Main part of the code
###############################################################################
print("****************************************************")
topic = "1. Read and Transform data"; print("** %s\n" % topic)

df = pd.read_csv('city_populations.csv', 
                 usecols=['name', 'group', 'year', 'value'])
print("Data: \n{}\n".format(df.head()))


current_year = 2018
top10 = (df[df['year'].eq(current_year)]
       .sort_values(by='value', ascending=True)
       .head(10))
print("Data: \n{}\n".format(top10))

#Unique counntries
print("Unique countries: \n{}\n".format(df.group.unique()))
colors = dict(zip(['India', 'Europe', 'Asia', 'Latin America', 'Middle East', 'North America', 'Africa'],
                  ['#adb0ff', '#ffb3ff', '#90d595', '#e48381', '#aafbff', '#f7bb5f', '#eafb50']))
group_lk = df.set_index('name')['group'].to_dict()


min_year = df.year.min()
max_year = df.year.max()


print("****************************************************")
topic = "2. Prepare the animation"; print("** %s\n" % topic)

def draw_barchart(year):
    top10 = df[df['year'].eq(year)].sort_values(by='value', ascending=True).tail(10)
    
    ax.clear()
    ax.barh(top10['name'], top10['value'], color=[colors[group_lk[x]] for x in top10['name']])
    
    dx = top10['value'].max() / 200
    for i, (value, name) in enumerate(zip(top10['value'], top10['name'])):
        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')
        ax.text(value-dx, i-.25, group_lk[name], size=10, color='#444444', ha='right', va='baseline')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    
    # ... polished styles
    ax.text(1, 0.4, year, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.text(0, 1.10, 'Population (thousands)', transform=ax.transAxes, size=12, color='#777777')
    
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    
    ax.set_yticks([])
    
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    
    ax.text(0, 1.15, 'The most populous cities in the world from {} to {}'.format(min_year, max_year),
            transform=ax.transAxes, size=20, weight=600, ha='left')
    ax.text(1, 0, 'by @pratapvardhan; credit @jburnmurdoch', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)
    
#draw_barchart(2018)

fig, ax = plt.subplots(figsize=(11, 5.5))
animator = animation.FuncAnimation(fig, draw_barchart, frames=range(min_year, max_year+1), interval=40, repeat_delay=2000)
plt.subplots_adjust(left=.05, bottom=.05, right=None, top=.8, wspace=None, hspace=None)
plt.show()
#HTML(animator.to_jshtml()) 
# or use animator.to_html5_video() or animator.save()

print("****************************************************")
topic = "End."; print("** %s\n" % topic)
