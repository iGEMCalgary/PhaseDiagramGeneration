from plotly import graph_objects as go
import pandas as pd


def genScatterGradient(fileName):
    data_in = pd.read_csv(fileName).values.tolist()
    x = []
    y = []
    z = []
    i2 = 0
    allow = "0001" * int(len(data_in) / 4)
    for i in range(0, len(data_in)):
        if data_in[i][1] * 10200 / 0.71 - 250 < data_in[i][2] < data_in[i][1] * 10200 / 0.71 + 250:
            if allow[i2] != 0:
                x.append(data_in[i][1])
                y.append(data_in[i][2])
                z.append(data_in[i][3])
            i2 += 1

    """
    prod_x = []
    prod_y = []
    prod_z = []
    norm = (10200, -0.71, 0)
    for vec_index in range(len(x)):
        c = x[vec_index] * 0.71 + y[vec_index] * 10200
        c /= 0.71**2 + 10200**2
        PROD = (-(norm[0] * c), -(norm[1] * c), -(norm[2] * c))
        prod_x.append(x[vec_index] + PROD[0])
        prod_y.append(y[vec_index] + PROD[1])
        prod_z.append(z[vec_index] + PROD[2])
    """
    """
    hyp = ((0.71**2 + 10200**2))**0.5
    print(hyp)
    rot_matrix = [(0.71/hyp, 10200/hyp),
                  (-10200/hyp, 0.71/hyp)]

    rot_x = []
    rot_y = []
    rot_z = []
    for vec_index2 in range(len(x)):
        rot_x.append(x[vec_index2] * rot_matrix[0][0] + y[vec_index2] * rot_matrix[0][1])
        rot_y.append(x[vec_index2] * rot_matrix[1][0] + y[vec_index2] * rot_matrix[1][1])
    """

    fig = go.Figure(data=
                    [go.Scatter3d(z=z, x=x, y=y, mode='markers',
                                  marker=dict(size=6, color=z, opacity=1,
                                              colorscale=[[0, 'gold'], [0.5, 'mediumturquoise'], [1, 'magenta']],
                                              showscale=True))])

    fig.update_layout(scene=dict(
        xaxis_title='GAMMA',
        yaxis_title='COST',
        zaxis_title='MEAN ERROR RATE'))

    fig.show()


def project_onPlane(x, y, z):
    return


genScatterGradient("T300-OUT_Error.csv")