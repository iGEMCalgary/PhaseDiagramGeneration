from itertools import permutations
from plotly import graph_objects as go
import pandas as pd
import random

class Display:
    # Each color for different phases. Uses the colors in increasing order
    colors = ['#65c6bb', '#87d37c', '#9b59b6', '#d64541', '#ffffcc', '#2c3e50']
    phases = ()

    def FormatDataFromCSV(self, stringfileName):
        """
        The format of the csv should have the column order: x, y, z, temp, phase.
        :param stringfileName: the fileName of csv
        :return: a tuple:
        first element is temperature
        second element is a list of tuples in form: (x, y, z, PHASE)
        """

        # Assume all data points given are at same temperature!

        data_list = pd.read_csv(stringfileName).values.tolist()
        return (data_list[0][3], [p for p in list(map(lambda x : [round(x[0], 5), round(x[1], 5), round(x[2], 5), round(x[4], 5)], data_list)) if p[-1] != 5])



    def DisplayTernaryPhaseScatter(self, data_in, temp):
        """
        Will display a ternary phase diagram as a scatter plot at a given temperature
        :param data_in: a list of tuples. Each tuple in form: (x, y, z, PHASE)
        :param temp: float kalvin temperature of the phase slice
        :return: nothing
        """
        all_phase_list = list(map(lambda x : x[3], data_in))
        phase_set = set(all_phase_list)
        self.phases = list(phase_set)
        n_unique_phases = len(phase_set)

        print(phase_set)

        if n_unique_phases > 6:
            raise Exception("Too many phases")

        phaseToColDict = {list(phase_set)[i] : self.colors[i] for i in range(n_unique_phases)}

        fig = go.Figure(go.Scatterternary({
            'mode': 'markers',
            'a': list(map(lambda x : x[2], data_in)),
            'b': list(map(lambda x : x[0], data_in)),
            'c': list(map(lambda x : x[1], data_in)),
            'text': all_phase_list,
            'marker': {
                'symbol': 0,
                'color': [phaseToColDict[phase] for phase in all_phase_list],
                'size': 9,
            }
        }))

        fig.update_layout({
            'title': 'Ternary Scatter Plot',
            'ternary':
                {
                    'sum': 1,
                    'aaxis': {'title': 'Surfactant'},
                    'baxis': {'title': 'Water'},
                    'caxis': {'title': 'Oil'}
                },
        })

        fig.show()
        return


    def Display3DScatter(self, data_in):
        # Display the data in
        return
