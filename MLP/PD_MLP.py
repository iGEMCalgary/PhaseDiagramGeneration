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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.losses

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

	#Paths for data readable csv files
	dataPathList = [r"Input\outputPD TN.csv"]

	#Get the input data and preprocess it	
	print("Loading phase diagram data...")
	startTime = time.time()
		
	phaseDf = downloadData(dataPathList)
	phaseDf, uniquePhases = formatData(phaseDf)	
		
	print("Phase diagram data loaded from {}. ".format(dataPathList) + timeEndTime(startTime))
	
	return [phaseDf, uniquePhases]

#Download input csv data from a given data path and return the dataframe
def downloadData(dataPathList):
	headers = ['Solid', 'Water', 'Gas', 'Temp', 'Phase']
	
	readCSVList = []
	for dataPath in dataPathList:
		readCSVList.append(pd.read_csv(dataPath, names=headers))
	
	csvDataframe = pd.concat(readCSVList, ignore_index = True, sort=False)
	
	return csvDataframe

#Format the dataframe for normalization and splitting
def formatData(phaseDf):
	#Get the headers of the dataframe
	phaseDfHeaders = list(phaseDf.columns)
	phaseCol = phaseDf[phaseDfHeaders[len(phaseDfHeaders)-1]]
	
	#Convert last column into consecutive binary column
	uniquePhases = getUniqueElems(phaseCol)
	uniquePhases.sort()
	phaseCol = [phasePoint/1. for phasePoint in phaseCol]
	
	#phaseCol = [uniquePhases.index(phasePoint) for phasePoint in phaseCol]	
	
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
	
	#Scale the dataframe with min max scaling
	xScaler = MinMaxScaler()
	trainValDf = pd.DataFrame(xScaler.fit_transform(trainValDf))
	testDf = pd.DataFrame(xScaler.transform(testDf))
	
	return [trainValDf, trainTarget, testDf, testTarget]

#Create and train the neural network
def getModel(trainX, trainY, uniquePhases):
	#Choose whether or not to load a previously trained model
	print("Enter in 1 if you wish to load an existing model h5 file, 2 if a new model must be created.")
	loadModelSetting = getUserInput(1, 2, "Enter in a number between 1 and 2:")
	
	#Paths for saving models and traing history
	modelSavePath = r"phaseModel.h5"
	trainingHistoryPath = r"trainingHistory.csv"
	
	#Build model
	phaseModel = buildModel([len(trainX.keys())], len(uniquePhases))
	print(phaseModel.summary())
	
	#Load model if it exists
	if loadModelSetting == 1:
		if os.path.isfile(modelSavePath) and os.path.isfile(trainingHistoryPath):
			phaseModel = keras.models.load_model(modelSavePath)
			trainingHistoryDf = pd.read_csv(trainingHistoryPath)
			print("Phase model loaded from: ", modelSavePath)
	#Else build the model from scratch
	else:
		#Train the model
		#Train model
		custEpochs=100000
		custBatchSize=1024
		startTime = time.time()
		trainingHistory = phaseModel.fit(trainX, trainY, epochs=custEpochs, batch_size=custBatchSize, shuffle=True,
										 validation_split=0.2, verbose=0, callbacks=createModelCallbackFunctions(modelSavePath))
		
		print("\nModel training complete. " + timeEndTime(startTime))
	
		#Record training history	
		trainingHistoryDf = pd.DataFrame(trainingHistory.history)
		trainingHistoryDf['epoch'] = trainingHistory.epoch
		
		print("Saving training history...")
		trainingHistoryDf.to_csv(trainingHistoryPath)
	
	#Display training history	
	print("\nEnd training loss values:")
	print(trainingHistoryDf.tail())
	plotHistory(trainingHistoryDf, 'Phase Classification')
	
	return phaseModel
	
#Build the model
def buildModel(inputShape, outputDimension):
	#Create the model; a keras sequential model
	model = keras.Sequential()
	
	#The GRU output dimensions are 256 days in a batch time sequence, 
	#the activation is the default tanh, and return the full sequence.
	model.add(layers.Dense(256, activation='softmax', input_shape=inputShape))
	model.add(layers.Dropout(0.2))

	#Output layer is a dense layer.
	model.add(layers.Dense(outputDimension, activation='softmax'))
	
	#Compile the model 
	optimizer = keras.optimizers.Adam()
	model.compile(loss='sparse_categorical_crossentropy',
				  optimizer = optimizer,
				  metrics=['accuracy'])
	return model


#Create the callback functions of the model
def createModelCallbackFunctions(modelSavePath):
	#Callback for early stopping
	earlyStopPatience = 5000
	callbackEarlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = earlyStopPatience)
	
	#Callback for checkpointing/model saving.
	callbackCheckpoint = keras.callbacks.ModelCheckpoint(filepath=modelSavePath, monitor = 'val_loss', save_best_only=True)
	
	#training message callback
	trainMessage = PrintDot()
	
	#Compile callback list
	callbackList = [callbackEarlyStop, callbackCheckpoint, trainMessage]
	
	return callbackList


#Train the model and plot the history
#Display training progress by printing a single dot for each completed epoch 
#Called on every batch
class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 1000 == 0: print('')
		if epoch % 100 == 0: print('.', end='')

#Plot the Error vs Training Epoch for the model
def plotHistory(histDf, targetHeader):	
	
	#Create the graph for the trainging and value MAE
	plt.figure(dpi = 120)
	plt.xlabel('Epoch')
	plt.ylabel('Sparse Categorical Crossentropy Error [{}]'.format(targetHeader))
	plt.plot(histDf['epoch'], histDf['loss'], label = 'Train Sparse Categorical Crossentropy', linewidth = 1, )
	plt.plot(histDf['epoch'], histDf['val_loss'], label = 'Val Sparse Categorical Crossentropy', linewidth = 1)
	plt.ylim(0, max(max(histDf['loss'][math.floor(len(histDf['loss'])*0.1):]), 
				max(histDf['val_loss'][math.floor(len(histDf['loss'])*0.1):]), 
				max(max(histDf['loss']), max(histDf['val_loss']))*0.01))
	
	plt.legend()
	
	#Show the graphs
	plt.show()
	plt.close()

#Evaluate the performance of the model on the test data
def evaluateModel(model, xDataTest, yDataTest):
	#Evaluate the final results
	loss, acc = model.evaluate(x=xDataTest,
								y=yDataTest)	
	print("Loss (Accuracy of Phase Classification): ", acc)

#Predict and evaluate the performance of the model on the given labeled data
def evaluatePredictions(model, xData, yData):
	startTime = time.time()
		
	# Use the model to predict the output-signals.
	yPredictions = model.predict(xData)
	
	#Plot how the predicted values compare to the truth
	printEgregiousPredictions(yData, yPredictions)
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
	prediction = np.argmax(nnOutput)
	confidence = nnOutput[prediction]
	
	return [prediction, confidence]

#Plot a subplot for the the NN's classification outputs for a given prediction list
def plotNNOutput(labeledPredList):
	displayGridLength = 5
	displayGridWidth = 6
	
	totalI = 0
	while totalI < len(labeledPredList):
		for i in range(displayGridLength*displayGridWidth):
			plt.subplot(displayGridLength, displayGridWidth, i+1)
			if totalI == len(labeledPredList):
				break
			plotArrayHistogram(labeledPredList[totalI])
			totalI = totalI + 1
		plt.show()
		plt.close()

#Plot (but don't show) the NN output for a single prediction with indicated correct answer
def plotArrayHistogram(outputSingle):
	nnOutput, trueVal = outputSingle
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	curPlot = plt.bar(range(len(nnOutput)), nnOutput, color="#777777")
	plt.ylim([0, 1])
	
	prediction = np.argmax(nnOutput)
	curPlot[prediction].set_color('red')
	curPlot[trueVal].set_color('blue')

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
	
	
	scaler = MinMaxScaler()

	ogTempRange = [[273.15],[293.15]]
	scaler.fit_transform(ogTempRange)
	
	tempSamples = [[253.15 + 5*i] for i in range(13)] 
	tempSamplesCopy = tempSamples.copy()
	
	tempSamples = scaler.transform(tempSamples)
	
	for scaledTemp in tempSamples:
		realTemp = scaler.inverse_transform([scaledTemp])[0][0]
		scaledTemp = scaledTemp[0]
		
		generatedTable = [[] for i in range(len(inputDfHeaders))]

		i = j = 0
		while i <= 1.0:		
			while j <= 1.0 - i:
				generatedTable[0].append(i)
				generatedTable[1].append(j)
				generatedTable[2].append(1.0-i-j)
				generatedTable[3].append(realTemp)
				generatedTable[4].append(getPrediction(phaseModel.predict(
										 pd.DataFrame([[i,j,1.0-i-j,scaledTemp]]))[0])[0])
				
				j = j + denseInterval
			j = 0
			i = i + denseInterval

		#Put the final output csv together
		for i in range(len(inputDfHeaders)):
			phaseDf[inputDfHeaders[i]] = generatedTable[i]
		
		#delete all unecessary data fields
		for header in inputDfHeaders[3:-1]:
			phaseDf.pop(header)
		
		#Save the final output dataframe to a csv file.
		print("Saving generated phase diagram.")
		print("Phase diagram for T = {}K generated and saved. ".format(realTemp) + timeEndTime(startTime))
		
		phaseDiagramSavePath = r"Output\Output2 T{}.csv".format(math.floor(realTemp))
		phaseDf.to_csv(phaseDiagramSavePath, index=False, header=False)
		
		phaseDf = phaseDfParam.copy()
		phaseDf = phaseDf.iloc[-1:-1,]
	
	
	

#DATA PREPROCESSING
####################################################################################

#Load and reformat the input data
phaseDf, uniquePhases = loadData()

#Split the given data into training, validation, and test data
trainDataX, trainDataY, testDataX, testDataY = splitData(phaseDf, trainTestFrac=0.8)

####################################################################################



#CREATE AND EVALUATE THE MLP ANN
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



