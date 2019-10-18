import plotly.express as px
import plotly.graph_objects as go
import math
from random import randint

import time

x_list = []
y_list = []
z_list = []
d = 3
for x in range(0, 105, d):
    for y in range(0, 105 - x, d):
            x_list.append(x / 100)
            y_list.append(y / 100)
            z_list.append((100 - x- y) / 100)



phaseToColDict = {0:'#8dd3c7', 1:'#ffffb3', 2:'#bebada', 3:'#fb8072', 4:'#80b1d3', 5:'#fdb462', 6: '#eeeeee'}
fig = go.Figure(go.Scatterternary({
    'mode': 'markers',
    'a': x_list,
    'b': y_list,
    'c': z_list,
    'text': [math.floor(i%6.0) for i in range(len(x_list))],
    'marker': {
        'symbol': 100,
        'color':[phaseToColDict[math.floor(i%6.0)] for i in range(len(x_list))],
        'size': 4,
        'line': { 'width': 2 }
    }
}))


fig.update_layout({
    'title': 'Ternary Scatter Plot',
    'ternary':
        {
        'sum':1,
        'aaxis':{'title': 'X', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'baxis':{'title': 'W', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'caxis':{'title': 'S', 'min': 0.01, 'linewidth':2, 'ticks':'outside' }
    },
    'showlegend': False
})

fig.show()


