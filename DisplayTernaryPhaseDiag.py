from itertools import permutations
from plotly import graph_objs as go
import pandas as pd
import chart_studio.plotly as py
import plotly.io as pio
import plotly
import math
import os
import time


class Display:
    # Each color for different phases. Uses the colors in increasing order
    colors = ['#46576C', '#EFF0F1', '#C2EABD', '#60B4C1', '#ffffcc', '#2c3e50']
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
            'b': list(map(lambda x : x[1], data_in)),
            'c': list(map(lambda x : x[0], data_in)),
            'text': all_phase_list,
            'marker': {
                'symbol': 0,
                'color': [phaseToColDict[phase] for phase in all_phase_list],
                'size': 14,
            }
        }))

        fig.update_layout({
            'title': 'Ternary Scatter Plot',
            'ternary':
                {
                    'sum': 1,
                    'aaxis': {'title': 'Surfactant'},
                'baxis': {'title': 'Oil'},
                'caxis': {'title': 'Water'}
            },
        })
        plotly.offline.plot(fig, filename='name.html')


        return


    def Display3DScatter(self):
        totalStartTime=time.time()
        phase_set = {1, 2, 3, 4}
        self.phases = list(phase_set)
        n_unique_phases = len(phase_set)

        if n_unique_phases > 99:
            raise Exception("Too many phases")

        phaseToColDict = {list(phase_set)[i] : self.colors[i] for i in range(n_unique_phases)}
        x = []
        y = []
        z = []
        colors_ = []

        root3_2 = (3**(1/2)/2)

        for t in range(295, 346, 2):
            data_list = pd.read_csv('PD_P2_MLP/Extrapolated Data P2/PD_P2_T' + str(t) + '.csv').values.tolist()
            for p in data_list:
                x.append(p[0]*0.5 + p[2])
                y.append(p[0] * root3_2)
                z.append(t)
                colors_.append(phaseToColDict[p[-1]])

        #GETTING RID OF UNWANTED PHASES
        unwantedPhases = [self.colors[i] for i in []]

        unwantedPhaseIndices = [index for index, value in enumerate(colors_) if value in unwantedPhases]
        unwantedPhaseIndices.reverse()
        for i in unwantedPhaseIndices:
            del x[i]
            del y[i]
            del z[i]
            del colors_[i]
		
        #save images for a camera view
        camR = math.sqrt(5-0.375**2)
        camX = camR
        camY = 0
		
        fig2 = go.Figure(data=[go.Scatter3d(
			        x=x,
			        y=z,
                    z=y,
	                mode='markers',
                    marker=dict(
                    size=3,
                    color=colors_,
                    opacity=0.4
                    )
                )])
		
        deltaTime = time.time() - totalStartTime
        print("Data Loaded. Time: {:5.3f} minutes.".format(deltaTime/60.0))
		
		rotationStage = 0		#The stage of the rotation
		frameStage = 0			#The index of the first frame you need to generate
		rotationTicks=100		#Number of ticks in one rotation 
								#(if you want the full rotation you need rotationTicks + 1)
								#Like this is a pic at 0 degrees and 360 degrees
							
        for i in range(rotationTicks):
            if i > rotationStage:
                startTime=time.time()

                camera = dict(up=dict(x=0, y=0, z=1),
		    				  center=dict(x=0, y=0, z=0),
		    				  eye=dict(x=camX, y=camY, z=0.375))

                fig2['layout'].update(scene=dict(
                    camera=camera,
             	    xaxis=dict(showbackground=False, showticklabels=False, nticks=6, range=[0,1]),
		    		yaxis=dict(showbackground=False, showticklabels=False, nticks=6, range=[290,345]),
		            zaxis=dict(showbackground=False, showticklabels=False, range=[0,root3_2])),
		    		autosize=False,
		    		width=1280,
		    		height=1280)
				
                #fig2.show()
				
                fig2.write_image(os.path.join("Visualization","gifConstruction","ternaryPlot_{}.png".format(i+frameStage)))
                #fig2.close()
                #fig2.show()
                deltaTime = time.time() - startTime
                print("Frame {} generated. Time: {:5.3f} minutes.".format(i+frameStage, deltaTime/60.0))
			
            #Update camera angle
            camR = math.sqrt(5-0.375**2)
            camX = camR
            camY = 0

            for i in range(100):
                if i < (rotationTicks/2):
                    camX = camR-(i%(rotationTicks/2))*(2*camR)/(rotationTicks/2)
                    camY = math.sqrt(max([0,5-0.375**2-camX**2]))
                else:
                    camX = -camR+(i%(rotationTicks/2))*(2*camR)/(rotationTicks/2)
                    camY = -math.sqrt(max([0,5-0.375**2-camX**2]))    
                print("i = {}, X = {}, Y = {}".format(i, camX, camY))
        return


    def reduce_to_2D(self):
        return