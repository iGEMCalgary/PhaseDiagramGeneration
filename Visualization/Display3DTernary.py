import plotly.graph_objects as go
from random import randint

# Normal vectors of planes that construct the triangle
# Is in the form ax + by + cz + d = 0
plane1 = (-(3 ** (1/2) / 2), 0.5, 0, 0)
plane2 = (-(3 ** (1/2) / 2), -0.5, 0, (3 ** (1/2) / 2))
plane3 = (0, 1, 0, 0)
planes = (plane1, plane2, plane3)
x_p, y_p, z_p = [], [], []

n_points = 50

for plane in planes:
    for i in range(n_points):
        z_p.append(randint(0, 100) / 10)
        if plane == plane1:
            xVal = randint(0, 50) / 100
        elif plane == plane2:
            xVal = randint(50, 100) / 100
        else:
            xVal = randint(0, 100) / 100
        x_p.append(xVal)
        y_p.append((-plane[-1] - plane[0]*xVal) * (1 / plane[1]))

fig = go.Figure(data=[go.Mesh3d(x=x_p, y=y_p, z=z_p,
                   opacity=0.4,
                   color='cyan')])
fig.show()
