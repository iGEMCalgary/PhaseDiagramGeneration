
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
import matplotlib.pyplot as plt
import statistics
import time
import math
from numpy.polynomial.polynomial import polyfit
import matplotlib.ticker as plticker

#For ML
from sklearn.svm import SVC
from sklearn import metrics

#For Dates
import datetime as dt
from datetime import timedelta
from datetime import date



#Global Settings
####################################################################################
#Temperature of the input data
tempConst = 293.15

#Path for data readable csv file
dataPath = r"Input\inputPD T{}.csv".format(math.floor(tempConst))

#Save Path for the generated data.
phaseDiagramSavePath = r"Output\outputPD T{}.csv".format(math.floor(tempConst))
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
	
	#gamma and C must be determined through heuristic methods
	phaseModel = SVC(kernel='rbf', gamma=0.1, C=10000.)
	
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
	plotPredVsTrueScatter(yData, yPredictions)
	plotErrDistr(yData, yPredictions)
	
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

	plotNNOutput(egregiousPredictionsList)


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
def getPrediction(nnOutput):
	"""
	prediction = np.argmax(nnOutput)
	confidence = nnOutput[prediction]
	
	return [prediction, confidence]
	"""
	return [nnOutput, 1.0]
	
#Plot a subplot for the the NN's classification outputs for a given prediction list
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
def createPhaseDiagram(phaseModel, phaseDfParam):
	#input dataframe format
	inputDfHeaders = list(phaseDfParam.columns)
	phaseDf = phaseDfParam.copy()
	phaseDf = phaseDf.iloc[-1:-1,]
	print("THE FOLLOWING DATAFRAME SHOULD BE EMPTY.")
	print(phaseDf)	
	
	#Interval of data points
	print("Generating phase diagram.")
	startTime = time.time()
	denseInterval = 0.01
	
	generatedTable = [[] for i in range(len(inputDfHeaders))]
	#tempConst = 273.15
	
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
	for i in range(len(inputDfHeaders)):
		phaseDf[inputDfHeaders[i]] = generatedTable[i]
	
	#Save the final output dataframe to a csv file.
	print("Saving generated phase diagram.")
	phaseDf.to_csv(phaseDiagramSavePath, index=False, header=False)
	print("Phase diagram generated and saved at {}. ".format(phaseDiagramSavePath) + timeEndTime(startTime))
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

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


