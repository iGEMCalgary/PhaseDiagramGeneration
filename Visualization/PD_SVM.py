
#Imports
from __future__ import absolute_import, division, print_function, unicode_literals

#Imports for loading and saving files
import pathlib
import os
import pickle
import re

#For reading csv files
import numpy as np
import pandas as pd

#For plots
import statistics
import time
import math
from numpy.polynomial.polynomial import polyfit
import plotly.graph_objects as go


#For ML
from sklearn.svm import SVC
from sklearn import metrics

#For Dates
import datetime as dt
from datetime import timedelta
from datetime import date



#Global Settings
####################################################################################
dataPath = r"limestone T2.csv"
phaseDiagramSavePath = r"limestone T2 Gen.csv"
tempConst = 293.15
####################################################################################

#determine the difference between the current time and the given start time
def timeEndTime(startTime):
	endTime = time.time()
	deltaTime = endTime - startTime
	if deltaTime % 60 < 1:
		timeString = "Time: {:5.3f} milliseconds.".format((deltaTime%60)*1000)
	else:
		timeString = "Time: {} minutes, {:5.3f} seconds.".format(math.floor(deltaTime/60.0), deltaTime % 60)
	
	return timeString

#Get an input from the user in the range of [lowerLimit, upperLimit]
def getUserInput(lowerLimit, upperLimit, prompt):
	while(True):
		try:
			print(prompt)
			userInput = input()
			if int(userInput) >= lowerLimit and int(userInput) <= upperLimit:
				return int(userInput)
		except ValueError:
			print("Invalid Input")

#Load the input data and preprocess it
def loadData():

	#Path for data readable csv file
	#dataPath = r"limestoneXL3.csv"
	#dataPath = r"limestone T0.csv"

	#Get the input data and preprocess it	
	print("Loading phase diagram data...")
	startTime = time.time()
		
	phaseDf = downloadData(dataPath)
	phaseDf, uniquePhases = formatData(phaseDf)	
		
	print("Phase diagram data loaded from {}. ".format(dataPath) + timeEndTime(startTime))
	
	return [phaseDf, uniquePhases]

#Download input csv data from a given data path and return the dataframe
def downloadData(dataPath):
	headers = ['Solid', 'Water', 'Gas', 'Temp', 'Phase']
	csvDataframe = pd.read_csv(dataPath, names=headers)
	
	return csvDataframe

#Format the dataframe for normalization and splitting
def formatData(phaseDf):
	#Get the headers of the dataframe
	phaseDfHeaders = list(phaseDf.columns)
	phaseCol = phaseDf[phaseDfHeaders[len(phaseDfHeaders)-1]]
	
	#Convert last column into consecutive binary column
	
	uniquePhases = getUniqueElems(phaseCol)
	uniquePhases.sort()
	
	#phaseCol = [uniquePhases.index(phasePoint) for phasePoint in phaseCol]
	
	phaseCol = [phasePoint/15. for phasePoint in phaseCol]
	
	phaseDf[phaseDfHeaders[len(phaseDfHeaders)-1]] = phaseCol
	
	return [phaseDf, uniquePhases]
	
#Get a unique list of each element in the given list
def getUniqueElems(givenList):
	uniqueList = []
	for elem in givenList:
		if elem not in uniqueList:
			uniqueList.append(elem)
	return uniqueList
	
#Split the given data into training, validation, and test sets
#Ensure at least one of every phase is in the train set.
def splitData(phaseDf, trainTestFrac=0.8):
	#Get the unique phases of the original data.
	phaseDfHeaders = list(phaseDf.columns)
	uniquePhases = getUniqueElems(phaseDf[phaseDfHeaders[len(phaseDfHeaders)-1]])
		
	#Shuffle and split the input data into training and testing
	randState = 0
	trainValDf = phaseDf.sample(frac=trainTestFrac, random_state=randState)
	while len(uniquePhases) != len(getUniqueElems(trainValDf[phaseDfHeaders[len(phaseDfHeaders)-1]])):
		randState = randState + 1
		trainValDf = phaseDf.sample(frac=trainTestFrac, random_state=randState)
	
	testDf = phaseDf.drop(trainValDf.index)
	
	print("The number of training data used is {}.".format(len(trainValDf)))
	
	#Split target header off from phase Df
	headers = list(phaseDf.columns)
	targetHeader = headers[len(headers)-1]
	
	trainTarget = trainValDf.pop(targetHeader)
	testTarget = testDf.pop(targetHeader)
	"""
	#Scale the dataframe with min max scaling
	xScaler = MinMaxScaler()
	trainValDf = pd.DataFrame(xScaler.fit_transform(trainValDf))
	testDf = pd.DataFrame(xScaler.transform(testDf))
	"""
	return [trainValDf, trainTarget, testDf, testTarget]
	

#Create and train the SVM
def getModel(trainX, trainY, uniquePhases):
	startTime = time.time()
	print("Creating SVM model.")
	phaseModel = SVC(kernel='rbf', gamma=0.015, C=10000.)	#For T2 - 92.865
	#phaseModel = SVC(kernel='rbf', gamma=0.1, C=10000.)	#For T1 - 92.340
	#phaseModel = SVC(kernel='rbf', gamma=0.0035, C=10000.) #For T0 - 94.753%
	
	phaseModel.fit(trainX, trainY)
	print("SVM Model created. " + timeEndTime(time.time()))
	
	return phaseModel

#Evaluate the performance of the model on the test data
def evaluateModel(model, xDataTest, yDataTest):
	#Evaluate the final results
	yPred = model.predict(xDataTest)
	acc = metrics.accuracy_score(yDataTest, yPred)
	print("Loss (Accuracy of Phase Classification): ", acc)

#Predict and evaluate the performance of the model on the given labeled data
def evaluatePredictions(model, xData, yData):
	startTime = time.time()
		
	# Use the model to predict the output-signals.
	yPredictions = model.predict(xData)
	
	#Plot how the predicted values compare to the truth
	calculateError(yData, yPredictions)
	
	print("Evaluation complete. " + timeEndTime(startTime))
	
#Print the most egregious model predictions
def	printEgregiousPredictions(yData, yPredictions):
	#These will be false positives with a high confidence
	egregiousConfidence = 0.8
	
	print("\nThe predictions with a confidence of > {}%:".format(egregiousConfidence*100))
	egregiousPredictionsList = []
	for testLab, testOut in zip(yData, yPredictions):
		testPred, testConf = getPrediction(testOut)
		if testPred != testLab and testConf >= egregiousConfidence:	
			egregiousPredictionsList.append((testOut, testPred))
			print("prediction = {},	true value = {}, Confidence = {:5.3f}%".format(
				   testPred, testLab, testConf*100))
				   
	print("Above is Displaying {} of {} ({:5.3f}%) test predictions.".format(
		  len(egregiousPredictionsList), len(yPredictions), len(egregiousPredictionsList)/len(yPredictions)*100))



#Calculate the accuracy of a set
def calculateError(yData, yPredictions):

	correctCount = 0
	for testLab, testOut in zip(yData, yPredictions):
		testPred = getPrediction(testOut)[0]
		if testLab == testPred:
			correctCount = correctCount + 1
			
	acc = correctCount/len(yData)
	print("Accuracy = {:5.3f}%".format(acc*100))


#Determine the classification and confidence level of a prediction
def getPrediction(nnOutput):
	"""
	prediction = np.argmax(nnOutput)
	confidence = nnOutput[prediction]
	
	return [prediction, confidence]
	"""
	return [nnOutput, 1.0]


#Use the trained model to densely create the data points for a phase diagram
def createPhaseDiagram(phaseModel, phaseDfParam):
	
	#Generate the phase diagram
	phaseDf = generatePDDF(phaseModel, phaseDfParam, 0.01)
	
	#Save the final output dataframe to a csv file.
	print("Saving generated phase diagram.")
	phaseDf.to_csv(phaseDiagramSavePath, index=False, header=False)
	print("Phase diagram saved at {}. ".format(phaseDiagramSavePath))
	
	#Display the generated phase diagram
	phaseDf = generatePDDF(phaseModel, phaseDfParam, 0.01)
	phaseDf.pop(list(phaseDf.columns)[-2:-1][0])
	displayTernaryPD(phaseDf)
	
	
#Generate a phase diagram dataframe 
def generatePDDF(phaseModel, phaseDfParam, denseInterval = 0.01):
	#input dataframe format
	headers = list(phaseDfParam.columns)
	phaseDf = phaseDfParam.copy()
	phaseDf = phaseDf.iloc[-1:-1,]
	#print("THE FOLLOWING DATAFRAME SHOULD BE EMPTY.")
	#print(phaseDf)		

	#Interval of data points
	print("Generating phase diagram.")
	startTime = time.time()
	
	generatedTable = [[] for i in range(len(headers))]
	
	i = j = 0
	while i <= 1.0:		
		while j <= 1.0 - i:
			generatedTable[0].append(i)
			generatedTable[1].append(j)
			generatedTable[2].append(1.0-i-j)
			generatedTable[3].append(tempConst)
			generatedTable[4].append(getPrediction(phaseModel.predict(
									 pd.DataFrame([[i,j,1.0-i-j,tempConst]]))[0])[0])
			
			j = j + denseInterval
		j = 0
		i = i + denseInterval

	

	#Put the final output csv together
	for i in range(len(headers)):
		phaseDf[headers[i]] = generatedTable[i]
	
	print("Phase diagram generated. " + timeEndTime(startTime))
	return phaseDf
	
	
#Display a phase diagram ternary contour and scatter plot 
def displayTernaryPD(phaseDf):
	#Format the data
	#The total number of phases in existence	
	totalPhases = 6 
	rawDataScatter = []
	rawDataContour = [[str(i)] for i in range(totalPhases)]
	
	for i in phaseDf.index:
		a,b,c,p = [phaseDf.iloc[i,j] for j in range(4)]
		#if 0 in [a,b,c] or p != phaseDf.iloc
		if True:
			rawDataScatter.append({'Species 1':	a,
								   'Species 2':	c,
								   'Species 3':	b,
								   'Phase':		p})
			rawDataContour[int(phaseDf.iloc[i,3])].append([phaseDf.iloc[i,0], phaseDf.iloc[i,2], phaseDf.iloc[i,1]])
		
	
	#Display scatter plot
	
	phaseToColDict = {0:'#8dd3c7', 1:'#ffffb3', 2:'#bebada', 3:'#fb8072', 4:'#80b1d3', 5:'#fdb462'}
	fig = go.Figure(go.Scatterternary({
		'mode': 'markers',
		'a': [i for i in map(lambda x: x['Species 1'], rawDataScatter)],
		'b': [i for i in map(lambda x: x['Species 2'], rawDataScatter)],
		'c': [i for i in map(lambda x: x['Species 3'], rawDataScatter)],
		'text': [i for i in map(lambda x: x['Phase'], rawDataScatter)],
		'marker': {
			'symbol': 100,
			'color':[phaseToColDict[phase] for phase in map(lambda x: x['Phase'], rawDataScatter)],
			'size': 4,
			'line': { 'width': 2 }
		}
	}))
	
	fig.update_layout({
		'ternary': {
			'sum': 100,
			'aaxis': makeAxis('Species 1', 0),
			'baxis': makeAxis('<br>Species 2', 45),
			'caxis': makeAxis('<br>Species 3', -45)
		},
		'annotations': [{
		  'showarrow': False,
		  'text': 'Simple Ternary Plot with Markers',
			'x': 0.5,
			'y': 1.3,
			'font': { 'size': 15 }
		}]
	})

	fig.show()
	
	
	#Display contour plot 
	
	#remove missing phases
	missingPhases = []
	for phase in rawDataContour:
		if len(phase) == 1:
			missingPhases.append(int(phase[0][0]))
			
	colors = ['#8dd3c7','#ffffb3','#bebada',
			  '#fb8072','#80b1d3','#fdb462']
			  
	for missingPhase in missingPhases:
		colors.pop(missingPhase)
		rawDataContour.pop(missingPhase)
	colors_iterator = iter(colors)

	fig = go.Figure()

	for rawDataPhase in rawDataContour:
		print(len(rawDataPhase))
		a = [innerData[0] for innerData in rawDataPhase[1:]]
		#a.append(rawDataPhase[1][0]) # Closing the loop

		b = [innerData[1] for innerData in rawDataPhase[1:]]
		#b.append(rawDataPhase[1][1]) # Closing the loop

		c = [innerData[2] for innerData in rawDataPhase[1:]]
		#c.append(rawDataPhase[1][2]) # Closing the loop

		fig.add_trace(go.Scatterternary(
			text = rawDataPhase[0],
			a=a, b=b, c=c, mode='lines',
			line=dict(color='#444', shape='spline'),
			fill='toself',
			fillcolor = colors_iterator.__next__()
		))

	fig.update_layout(title = 'Ternary Contour Plot')
	fig.show()
	
	
#Make the axes for the ternary phase diagram
def makeAxis(title, tickangle):
    return {
      'title': title,
      'titlefont': { 'size': 20 },
      'tickangle': tickangle,
      'tickfont': { 'size': 15 },
      'tickcolor': 'rgba(0,0,0,0)',
      'ticklen': 5,
      'showline': True,
      'showgrid': True
    }	
	
	
	
	
	
	
	
	
	
	
	
	
	

#DATA PREPROCESSING
####################################################################################

#Load and reformat the input data
phaseDf, uniquePhases = loadData()

#Split the given data into training, validation, and test data
trainDataX, trainDataY, testDataX, testDataY = splitData(phaseDf, trainTestFrac=0.05)

####################################################################################



#CREATE AND EVALUATE THE SVM
####################################################################################

#Create the model
phaseModel = getModel(trainDataX, trainDataY, uniquePhases)

#Calculate the test error of the model
evaluateModel(phaseModel, testDataX, testDataY)

#Make predictions for the training data
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating Training Data Performance~~~~~~~~~~~~~~~~~~~~~~~~~")
evaluatePredictions(phaseModel, trainDataX, trainDataY)

#Make predictions for the testing data
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating Testing Data Performance~~~~~~~~~~~~~~~~~~~~~~~~~")
evaluatePredictions(phaseModel, testDataX, testDataY)

####################################################################################


#PRODUCE FINAL PHASE DIAGRAM POINTS
####################################################################################
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~Fully Interpolating Phase Diagram~~~~~~~~~~~~~~~~~~~~~~~~~")
createPhaseDiagram(phaseModel, phaseDf)

####################################################################################


