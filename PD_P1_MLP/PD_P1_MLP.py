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
	phaseCol = [phasePoint-1 for phasePoint in phaseCol]

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

#Create and train the SVM
def getModel(trainX, trainY, uniquePhases, tempConst, customSettings={}):
	#Paths for saving models and traing history
	
	
	#Specify settings of the model training
	defaultSettings = {'trainingMaxEpochs': 100000,
					   'trainingPatience' : 50000,
					   'trainingBatchSize': 32,
					   'modelSavePath'	  :	r"phaseModel_T{}.h5".format(math.floor(tempConst)),
					   'trainingSavePath' :	r"trainingHistory.csv"}
	#Build model
	phaseModel = buildModel([len(trainX.keys())], len(uniquePhases))
	print(phaseModel.summary())	
	
	useArg = lambda x: customSettings[x] if x in list(customSettings.keys()) else defaultSettings[x]
	custEpochs = useArg('trainingMaxEpochs')
	custBatchSize = useArg('trainingBatchSize')
	custPatience = useArg('trainingPatience')
	modelSavePath = useArg('modelSavePath')
	trainingHistoryPath = useArg('trainingSavePath')
	
	#Train the model
	startTime = time.time()
	if os.path.isfile(modelSavePath) and os.path.isfile(trainingHistoryPath):
		phaseModel = keras.models.load_model(modelSavePath)
		trainingHistoryDf = pd.read_csv(trainingHistoryPath)
		print("Phase model loaded from: ", modelSavePath)
	else:
		trainingHistory = phaseModel.fit(trainX, trainY, epochs=custEpochs, batch_size=custBatchSize, 
										 shuffle=True, validation_split=0.2, verbose=0, 
										 callbacks=createModelCallbackFunctions(modelSavePath, custPatience))
		print("\nModel training complete. " + timeEndTime(startTime))

		#Record training history	
		trainingHistoryDf = pd.DataFrame(trainingHistory.history)
		trainingHistoryDf['epoch'] = trainingHistory.epoch
	
		print("Saving training history...")
		trainingHistoryDf.to_csv(trainingHistoryPath)
	
	plotHistory(trainingHistoryDf, 'Phase Classification')
	
	return phaseModel

#Build the model
def buildModel(inputShape, outputDimension):
	#Create the model; a keras sequential model
	model = keras.Sequential()
		
	#Standard feedforward MLP
	model.add(layers.Dense(512, activation='softmax', input_shape=inputShape))
	model.add(layers.Dropout(0.05))

	#Output layer is a dense layer.
	model.add(layers.Dense(outputDimension, activation='softmax'))
	
	#Compile the model 
	optimizer = keras.optimizers.Adam()
	model.compile(loss='sparse_categorical_crossentropy',
				  optimizer = optimizer,
				  metrics=['accuracy'])
	return model

#Evaluate the performance of the model on the test data
def evaluateModel(model, xDataTest, yDataTest):
	#Evaluate the final results
	loss, acc = model.evaluate(x=xDataTest,
							   y=yDataTest)	
	print("Loss (Accuracy of Phase Classification): ", acc)

#Create the callback functions of the model
def createModelCallbackFunctions(modelSavePath, earlyStopPatienceP):
	#Callback for early stopping
	earlyStopPatience = earlyStopPatienceP
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
		if epoch % 5000 == 0: print("Epoch {} completed.".format(epoch))

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
	prediction = np.argmax(modelOutput)
	confidence = modelOutput[prediction]
	
	return [prediction, confidence]
	
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
							   'Phase':		p+1})
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
			'aaxis': makeAxis('A: Water', 0),
			'baxis': makeAxis('<br>B: Oil', 45),
			'caxis': makeAxis('<br>C: Surfactant', -45)
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
	phaseDf, uniquePhases, tempConst = loadData(inputFilePath, undesirablePhases=[])

	#Split the given data into training, validation, and test data
	trainDataX, trainDataY, testDataX, testDataY = splitData(phaseDf, trainTestFrac=0.80)
	####################################################################################


	#CREATE AND EVALUATE THE SVM
	####################################################################################
	#Create the model
	phaseModel = getModel(trainDataX, trainDataY, uniquePhases, tempConst)

	#Calculate the test error of the model
	evaluateModel(phaseModel, testDataX, testDataY)

	#Make predictions for the training data
	print("\n~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating Training Data Performance~~~~~~~~~~~~~~~~~~~~~~~~~")
	evaluatePredictions(phaseModel, trainDataX, trainDataY, plot=True)

	#Make predictions for the testing data
	print("\n~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating Testing Data Performance~~~~~~~~~~~~~~~~~~~~~~~~~")
	evaluatePredictions(phaseModel, testDataX, testDataY, plot=True)
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