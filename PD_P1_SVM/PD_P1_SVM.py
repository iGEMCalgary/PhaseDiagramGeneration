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

#For plots and evaluation
import matplotlib.pyplot as plt
import statistics
import time
import math
from numpy.polynomial.polynomial import polyfit
import matplotlib.ticker as plticker
import plotly.graph_objects as go

#For ML
from sklearn.svm import SVC
from sklearn import metrics

#For Dates
import datetime as dt
from datetime import timedelta
from datetime import date

#determine the difference between the current time and the given start time
def timeEndTime(startTime):
	endTime = time.time()
	deltaTime = endTime - startTime
	if deltaTime % 60 < 1:
		timeString = "Time: {:5.3f} milliseconds.".format((deltaTime%60)*1000)
	else:
		timeString = "Time: {} minutes, {:5.3f} seconds.".format(math.floor(deltaTime/60.0), deltaTime % 60)
	
	return timeString

#Load the input data and preprocess it
def loadData(dataPath, undesirablePhases=[]):
	#Get the input data and preprocess it	
	print("Loading phase diagram data...")
	startTime = time.time()
		
	phaseDf = downloadData(dataPath)
	for deletPhase in undesirablePhases:
		phaseDf = phaseDf.loc[phaseDf.loc[:,'Phase'] != deletPhase,:]

	phaseDf, uniquePhases, tempConst = formatData(phaseDf)	
		
	print("Phase diagram data for T = {}K loaded from {}. ".format(tempConst, dataPath) + timeEndTime(startTime))
	
	return [phaseDf, uniquePhases, tempConst]

#Get the file names of every csv or specific csv files in a given folder path
def filesInFolder(folderPath, specificFiles=[]):	
	fileList=[]
	for filename in os.listdir(folderPath):
		if (len(specificFiles)==0 or filename in specificFiles) and filename[-4:]=='.csv':
			fileList.append(os.path.join(folderPath, filename))
	return fileList

#Download input csv data from a given data path and return the dataframe
def downloadData(dataPath):
	headers = ['X', 'Y', 'Z', 'Temp', 'Phase']
	csvDataframe = pd.read_csv(dataPath, names=headers)
	return csvDataframe

#Format the dataframe for normalization and splitting
def formatData(phaseDf):
	#Get the headers of the dataframe
	phaseDfHeaders = list(phaseDf.columns)
	phaseCol = phaseDf[phaseDfHeaders[len(phaseDfHeaders)-1]]
	
	#Get the temperature of the phase diagram
	tempConst = list(phaseDf[phaseDfHeaders[3]])[0]
	
	#Convert last column into consecutive binary column
	uniquePhases = getUniqueElems(phaseCol)
	uniquePhases.sort()
	
	#Format the phase data
	phaseCol = [phasePoint for phasePoint in phaseCol]

	phaseDf[phaseDfHeaders[len(phaseDfHeaders)-1]] = phaseCol
	
	return [phaseDf, uniquePhases, tempConst]
	
#Get a unique list of each element in the given list
def getUniqueElems(givenList):
	return list(set(givenList))
	
#Split the given data into training, validation, and test sets
#Ensure at least one of every phase is in the train set.
def splitData(phaseDf, trainTestFrac=0.8):
	#Get the unique phases of the original data.
	phaseDfHeaders = list(phaseDf.columns)
	uniquePhases = getUniqueElems(phaseDf[phaseDfHeaders[len(phaseDfHeaders)-1]])
		
	#Shuffle and split the input data into training and testing so that at least
	#one data point from each phase is in the training data.
	randState = 0
	trainValDf = phaseDf.sample(frac=trainTestFrac, random_state=randState)
	while len(uniquePhases) != len(getUniqueElems(trainValDf[phaseDfHeaders[len(phaseDfHeaders)-1]])):
		randState = randState + 1
		trainValDf = phaseDf.sample(frac=trainTestFrac, random_state=randState)
	print("The number of training data used is {}.".format(len(trainValDf)))

	#Create testing data
	testDf = phaseDf.drop(trainValDf.index)
	
	#Split target header off from phase Df
	headers = list(phaseDf.columns)
	targetHeader = headers[len(headers)-1]
	trainTarget = trainValDf.pop(targetHeader)
	testTarget = testDf.pop(targetHeader)

	return [trainValDf, trainTarget, testDf, testTarget]

#Define your custom hyperparameters for the model
def specifyHyperParameters(temperature, default = (0.1,10000.)):
	"""
	Temp	Training	Testing
	300.15:	84.848% 	93.750%
	310.15: 91.228%		78.571%
	315.15:	98.246%		71.429%
	328.15:	91.228%		57.143%
	343.15:	87.719%		78.571%
	"""
	hpDict = {300.15:(0.1, 10000.), 310.15:(3.2, 10000.),
			  315.15:(5., 10000.), 328.15:(1.9, 10000.),
			  343.15:(1.0, 10000.)}
	if temperature in list(hpDict.keys()):
		return hpDict[temperature]
	return default

#Create and train the SVM
def getModel(trainX, trainY, uniquePhases, hyperParams):
	startTime = time.time()
	print("\nCreating SVM model.")
	
	#The model is an SVM with an rbf kernel
	phaseModel = SVC(kernel='rbf', gamma=hyperParams[0], C=hyperParams[1])
	
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
def evaluatePredictions(model, xData, yData, plot=True):
	startTime = time.time()
		
	# Use the model to predict the output-signals.
	yPredictions = model.predict(xData)
	
	#Plot how the predicted values compare to the truth
	calculateError(yData, yPredictions)
	if plot:
		plotPredVsTrueScatter(yData, yPredictions)
		plotErrDistr(yData, yPredictions)
	
	print("Evaluation complete. " + timeEndTime(startTime))

#Calculate the accuracy of a set
def calculateError(yData, yPredictions):
	correctCount = 0
	for testLab, testOut in zip(yData, yPredictions):
		testPred = getPrediction(testOut)[0]
		if testLab == testPred:
			correctCount = correctCount + 1
			
	acc = correctCount/len(yData)
	print("Accuracy = {:5.3f}%".format(acc*100))

#Plot the test labels vs the test predictions
def plotPredVsTrueScatter(yData, yPredictions):
	targetHeader = 'Phase'
	testLabels = yData
	testPredictions = [getPrediction(nnOut)[0] for nnOut in yPredictions]
	
	plt.scatter(testLabels, testPredictions, alpha=0.1, s=100)
	plt.xlabel("True Values [{}]".format(targetHeader).replace(".", " "))
	plt.ylabel("Predictions [{}]".format(targetHeader).replace(".", " "))
	plt.axis('equal')
	plt.axis('square')	
	
	xRange = (plt.xlim()[1] - plt.ylim()[0])
	yRange = (plt.ylim()[1] - plt.ylim()[0])
	
	plt.xlim(plt.xlim()[0] - xRange*0.05, plt.xlim()[1] + xRange*0.05)
	plt.ylim(plt.ylim()[0] - yRange*0.05, plt.ylim()[1] + yRange*0.05)
	#Plots the linear line of best fit
	_ = plt.plot([-30, 30],
				 [-30, 30], color='green')	
	b, m = polyfit(testLabels, testPredictions, 1)
	print("b = {}, m = {} for linear regression of predicted values vs true values".format(b, m))
	plt.plot(testLabels, [b+m*tlabel for tlabel in testLabels], '-', color='red')
	plt.show()
	plt.close()
	
	return (b, m)

#Plot the error distribution of some test predictions
def plotErrDistr(yData, yPredictions, ):
	#Compute the number of errors for each potential label 
	uniquePhases = getUniqueElems(yData)
	error = []
	for testLab, testOut in zip(yData, yPredictions):
		testPred = getPrediction(testOut)[0]
		if testLab != testPred:
			error.append(testLab)
	
	#Plot the histogram
	plt.hist(error, bins = len(uniquePhases), rwidth = 0.90)
	plt.xlabel("True Phase Type")
	_ = plt.ylabel("Count (No. of Predictions)")
	plt.show()
	plt.close()


#Determine the classification and confidence level of a prediction
def getPrediction(modelOutput):
	return [modelOutput, 1.0]
	
#Plot a subplot for the the model's classification outputs for a given prediction list
def plotNNOutput(labeledPredList):
	displayGridLength = 5
	displayGridWidth = 6
	
	totalI = 0
	while totalI < len(labeledPredList):
		for i in range(displayGridLength*displayGridWidth):
			if totalI == len(labeledPredList):
				break
			totalI = totalI + 1
		plt.show()
		plt.close()

#Use the trained model to densely create the data points for a phase diagram
def createPhaseDiagram(phaseModel, phaseDfParam, tempConst, totalPhases, savePath):
	#Generate the phase diagram
	phaseDf = generatePDDF(phaseModel, phaseDfParam, tempConst, 0.01)
	
	#Save the final output dataframe to a csv file.
	print("Saving generated phase diagram.")
	phaseDiagramSavePath = savePath[:-4] + '_Gen' + savePath[-4:]
	phaseDf.to_csv(phaseDiagramSavePath, index=False, header=False)
	print("Phase diagram saved at {}. ".format(phaseDiagramSavePath))
	
	#Display the generated phase diagram
	#phaseDf = generatePDDF(phaseModel, phaseDfParam, tempConst, 0.005)
	phaseDf.pop(list(phaseDf.columns)[-2:-1][0])
	displayTernaryPD(phaseDf, totalPhases, tempConst)
	
#Generate a phase diagram dataframe 
def generatePDDF(phaseModel, phaseDfParam, tempConst, denseInterval = 0.01):
	#Get an empty dataframe of the same form as the input dataframe
	headers = list(phaseDfParam.columns)
	phaseDf = phaseDfParam.copy()
	phaseDf = phaseDf.iloc[-1:-1,]

	#Generate the phase diagram
	print("Generating phase diagram.")
	startTime = time.time()
	generatedTable = [[] for i in range(len(headers))]
	i = j = 0
	while round(i,5) <= 1.0:		
		while round(j,5) <= round(1.0 - i, 5):
			generatedTable[0].append(round(i,5))
			generatedTable[1].append(round(j,5))
			generatedTable[2].append(round(1.0-i-j,5))
			generatedTable[3].append(tempConst)
			generatedTable[4].append(round(getPrediction(phaseModel.predict(
									 pd.DataFrame([[i,j,1.0-i-j,tempConst]]))[0])[0],5))
			j = j + denseInterval
		j = 0
		i = i + denseInterval

	#Put the final output csv together
	for i in range(len(headers)):
		phaseDf[headers[i]] = generatedTable[i]
	
	print("Phase diagram generated. " + timeEndTime(startTime))
	return phaseDf
	
	
#Display a phase diagram ternary contour and scatter plot 
def displayTernaryPD(phaseDf, overallPhases, tempConst):
	#Format the data
	#The total number of phases in existence	
	totalPhases = overallPhases
	rawDataScatter = []
	rawDataContour = [[str(i+1)] for i in range(totalPhases)]
	
	for i in phaseDf.index:
		a,b,c,p = [phaseDf.iloc[i,j] for j in range(4)]
		rawDataScatter.append({'Species 1':	c,
							   'Species 2':	b,
							   'Species 3':	a,
							   'Phase':		p})
		rawDataContour[int(phaseDf.iloc[i,3])-1].append([phaseDf.iloc[i,0], phaseDf.iloc[i,2], phaseDf.iloc[i,1]])
		
	#Display scatter plot
	displayTernaryPDScatter(rawDataScatter, tempConst)
	
	#Display contour plot 
	#displayTernaryPDContour(rawDataContour)
	
#Display the generated phase diagram as a scatterplot
def displayTernaryPDScatter(rawDataScatter, tempConst):
	phaseToColDict = {0:'#8dd3c7', 1:'#ffffb3', 2:'#bebada', 3:'#fb8072', 4:'#80b1d3', 5:'#fdb462'}
	fig = go.Figure(go.Scatterternary(		
		{
		'mode': 'markers',
		'a': [i for i in map(lambda x: x['Species 1'], rawDataScatter)],
		'b': [i for i in map(lambda x: x['Species 2'], rawDataScatter)],
		'c': [i for i in map(lambda x: x['Species 3'], rawDataScatter)],
		'text': [i for i in map(lambda x: x['Phase'], rawDataScatter)],
		'marker': {
			'symbol': 100,
			'color':[phaseToColDict[phase] for phase in map(lambda x: x['Phase'], rawDataScatter)],
			'size': 2,
			'line': {'width': 2}
		}
	}))
	
	fig.update_layout({
		'ternary': {
			'sum': 100,
			'aaxis': makeAxis('A: Surfactant', 0),
			'baxis': makeAxis('<br>B: Water', 45),
			'caxis': makeAxis('<br>C: Oil', -45)
		},
		'annotations': [{
		  'showarrow': False,
		  'text': 'Simple Ternary Plot with Markers',
			'x': 0.5,
			'y': 1.3,
			'font': { 'size': 15 }
		}],
		'title':"Temp = {}K".format(tempConst)
	})

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

#Display the generated phase diagram as a countour plot
def displayTernaryPDContour(rawDataContour):
	#remove missing phases
	missingPhases = []
	for phase in rawDataContour:
		if len(phase) == 1:
			missingPhases.append(int(phase[0][0]))
			
	colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072',
			  '#80b1d3','#fdb462','#476264','#787851',
			  '#489ad3','$e01523','#ef328f','#919191'][:len(rawDataContour)]
			  
	for missingPhase in missingPhases:
		colors.pop(missingPhase)
		rawDataContour.pop(missingPhase)
	colors_iterator = iter(colors)

	fig = go.Figure()

	for rawDataPhase in rawDataContour:
		a = [innerData[0] for innerData in rawDataPhase[1:]]
		b = [innerData[1] for innerData in rawDataPhase[1:]]
		c = [innerData[2] for innerData in rawDataPhase[1:]]

		fig.add_trace(go.Scatterternary(
			text = rawDataPhase[0],
			a=a, b=b, c=c, mode='lines',
			line=dict(color='#444', shape='spline'),
			fill='toself',
			fillcolor = colors_iterator.__next__()
		))

	fig.update_layout(title = 'Ternary Contour Plot')
	fig.show()
	

#Do for each file in the folder
inputFolderPath = r"Experimental Data P1"
outputFolderPath = r"Interpolated Data P1" 
for inputFilePath in filesInFolder(inputFolderPath, specificFiles=[]):
	print("\n############################################################################################")
	#DATA PREPROCESSING
	####################################################################################
	#Load and reformat the input data
	phaseDf, uniquePhases, tempConst = loadData(inputFilePath, undesirablePhases=[5])

	#Split the given data into training, validation, and test data
	trainDataX, trainDataY, testDataX, testDataY = splitData(phaseDf, trainTestFrac=0.80)
	####################################################################################


	#CREATE AND EVALUATE THE SVM
	####################################################################################
	#Create the model
	phaseModel = getModel(trainDataX, trainDataY, uniquePhases, specifyHyperParameters(tempConst))

	#Calculate the test error of the model
	evaluateModel(phaseModel, testDataX, testDataY)

	#Make predictions for the training data
	print("\n~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating Training Data Performance~~~~~~~~~~~~~~~~~~~~~~~~~")
	evaluatePredictions(phaseModel, trainDataX, trainDataY, plot=False)

	#Make predictions for the testing data
	print("\n~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating Testing Data Performance~~~~~~~~~~~~~~~~~~~~~~~~~")
	evaluatePredictions(phaseModel, testDataX, testDataY, plot=False)
	####################################################################################


	#PRODUCE FINAL PHASE DIAGRAM POINTS
	####################################################################################
	print("\n~~~~~~~~~~~~~~~~~~~~~~~~~Fully Interpolating Phase Diagram~~~~~~~~~~~~~~~~~~~~~~~~~")
	#Specify the total phases of the entire system
	totalPhases = 4
	outputFilePath = os.path.join(outputFolderPath, inputFilePath[len(inputFolderPath)+1:])
	createPhaseDiagram(phaseModel, phaseDf, tempConst, totalPhases, outputFilePath)
	####################################################################################
	print("\n############################################################################################\n")