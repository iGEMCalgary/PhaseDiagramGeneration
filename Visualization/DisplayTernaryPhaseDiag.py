from itertools import permutations
from plotly import graph_objects as go
import pandas as pd

# Each color for different phases. Uses the colors in increasing order
colors = ['#65c6bb', '#87d37c', '#9b59b6', '#d64541', '#ffffcc', '#2c3e50']

def FormatDataFromCSV(stringfileName):
    """
    The format of the csv should have the column order: x, y, z, temp, phase.
    :param stringfileName: the fileName of csv
    :return: a tuple:
    first element is temperature
    second element is a list of tuples in form: (x, y, z, PHASE)
    """

    # Assume all data points given are at same temperature!

    data_list = pd.read_csv(stringfileName).values.tolist()
    return (data_list[0][3], list(map(lambda x : [round(x[0], 5), round(x[1], 5), round(x[2], 5), round(x[4], 5)], data_list)))



def DisplayTernaryPhaseScatter(data_in, temp):
    """
    Will display a ternary phase diagram as a scatter plot at a given temperature
    :param data_in: a list of tuples. Each tuple in form: (x, y, z, PHASE)
    :param temp: float kalvin temperature of the phase slice
    :return: nothing
    """
    all_phase_list = list(map(lambda x : x[3], data_in))
    phase_set = set(all_phase_list)
    n_unique_phases = len(phase_set)

    if n_unique_phases > 6:
        raise Exception("Too many phases")

    phaseToColDict = {list(phase_set)[i] : colors[i] for i in range(n_unique_phases)}

    fig = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': list(map(lambda x : x[0], data_in)),
        'b': list(map(lambda x : x[1], data_in)),
        'c': list(map(lambda x : x[2], data_in)),
        'text': all_phase_list,
        'marker': {
            'symbol': 0,
            'color': [phaseToColDict[phase] for phase in all_phase_list],
            'size': 9,
        }
    }))

    fig.show()
    return



# Find all points that are on boundaries on phases for the contour plot
def GetContourPoints(data_in):
    """
    :param data_in: a list of tuples in form: (x, y, z, PHASE)
    :param temp: the temperature of this slice
    :return:  A subset of all data points, and only those that are on boundaries
    """

    boundary_points = [] # Form: (x, y, z, phase)

    permuted = list(permutations([0, .01, -.01]))
    phaseDict = {(p[0], p[1], p[2]) : p[3] for p in data_in }
    print([0.01, 0.99, 0.0] in data_in)

    #for p in data_in:
    for p in data_in:

        # Check if on edge
        if p[0] == 0 or p[1] == 0 or p[2] == 0:
            boundary_points.append(p)
            continue

        # Get all 6 adjacent
        this_phase = p[3]
        for permutation in permuted:
           #print(phaseDict[round(p[0] + permutation[0], 5), round(p[1] + permutation[1], 5), round(p[2] + permutation[2], 5)])
            if (round(p[0] + permutation[0], 5), round(p[1] + permutation[1], 5), round(p[2] + permutation[2], 5)) in phaseDict:
                if phaseDict[round(p[0] + permutation[0], 5), round(p[1] + permutation[1], 5), round(p[2] + permutation[2], 5)] != this_phase:
                    boundary_points.append(p)
                break
    return boundary_points



def DisplayTernaryPhaseContour(data_in, temp):
    """
    Will display a ternary phase diagram as a contour plot at a given temperature
    :param data_in: a list of tuples. Each tuple in form: (x, y, z, PHASE)
    :param temp: float kalvin temperature of the phase slice
    :return: nothing
    """
    colorIT = iter(colors)

    all_phase_list = list(map(lambda x : x[3], data_in))
    phase_set = set(all_phase_list)

    contour_subset = GetContourPoints(data_in)
    print([0.35, 0.39, 0.26, 3.0] in contour_subset)
    print(contour_subset)
    print(len(contour_subset))


    figure = go.Figure()
    colorIT = iter(colors)

    for phase in list(phase_set):

        phase_points = [p for p in contour_subset if p[3] == phase]

        figure.add_trace(go.Scatterternary(
            text = phase,
            a = list(map(lambda x: x[0], phase_points)),
            b = list(map(lambda x: x[1], phase_points)),
            c = list(map(lambda x: x[2], phase_points)),
            mode='lines',
            line=dict(color='#444'),
            fill='toself',
            fillcolor=colorIT.__next__()
        ))

    figure.show()


# RUN FUNCTIONS
data_read = FormatDataFromCSV('limestone T2 Gen.csv')
DisplayTernaryPhaseScatter(data_read[1], data_read[0])

