from KNN.KNN_Classifier import KNN_Classifier
from DisplayTernaryPhaseDiag import Display
from SVM.Resampling_SVM import ResampleSVM
from KNN.Resampling_KNN import Resampling

ALL_DATA_PATH = 'RESULTS1_temp2.csv'

display = Display()
KNN = KNN_Classifier()




#display.Display3DScatter()


data_read = display.FormatDataFromCSV(ALL_DATA_PATH)
svmResample = ResampleSVM(data_read[1])
knnResample = Resampling(data_read[1], 10, 5)
display.DisplayTernaryPhaseScatter(data_read[1], data_read[0])
#display.DisplayTernaryPhaseScatter(KNN.KNN(6, data_read[1], 0.005), data_read[0])
display.Display3DScatter()

#display.DisplayTernaryPhaseScatter(data_read[1], data_read[0])

