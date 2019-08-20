from KNN.KNN_Classifier import KNN_Classifier
from DisplayTernaryPhaseDiag import Display
from SVM.Resampling_SVM import ResampleSVM
from KNN.Resampling_KNN import Resampling
import os

#ALL_DATA_PATH = 'RESULTS1_temp2.csv'
ALL_DATA_PATH = "PD_P1_SVM/Experimental Data P1/ER2_T343.csv"

display = Display()
KNN = KNN_Classifier()




#display.Display3DScatter()
data_read = display.FormatDataFromCSV(ALL_DATA_PATH)
"""

svmResample = ResampleSVM(data_read[1])
knnResample = Resampling(data_read[1], 10, 5)
display.DisplayTernaryPhaseScatter(data_read[1], data_read[0])
#display.DisplayTernaryPhaseScatter(KNN.KNN(6, data_read[1], 0.005), data_read[0])
"""
#display.Display3DScatter()
data_read = display.FormatDataFromCSV(ALL_DATA_PATH)
display.DisplayTernaryPhaseScatter(data_read[1], data_read[0])

ALL_DATA_PATH = "RESULTS1_temp3.csv"
data_read = display.FormatDataFromCSV(ALL_DATA_PATH)
display.DisplayTernaryPhaseScatter(data_read[1], data_read[0])

