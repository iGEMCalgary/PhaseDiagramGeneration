from KNN_Classifier import KNN_Classifier
from Resampling import Resampling
from DisplayTernaryPhaseDiag import Display

ALL_DATA_PATH = 'RESULTS1_temp5.csv'

display = Display()
KNN = KNN_Classifier()


# RUN FUNCTIONS
data_read = display.FormatDataFromCSV(ALL_DATA_PATH)
print(data_read)

data_copy = data_read[1].copy()

resample = Resampling(data_read[1], 10, data_read[0])

optimalK = resample.find_optimal_k( max_k=20 )
print(optimalK)
print("Min Error: " + str(min(resample.errors)))
resample.display_ERR_over_K()

#display.DisplayTernaryPhaseScatter(data_read[1], data_read[0])
display.DisplayTernaryPhaseScatter(KNN.KNN(9, data_read[1], 0.01), data_read[0])
