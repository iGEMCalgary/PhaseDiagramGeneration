from KNN.KNN_Classifier import KNN_Classifier
from KNN.Resampling_KNN import Resampling
from DisplayTernaryPhaseDiag import Display
import numpy as np
import pandas as pd
import os

ALL_DATA_PATH = 'RESULTS1_temp4.csv'

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
generated_data = KNN.KNN(9, data_read[1], 0.01)
display.DisplayTernaryPhaseScatter(generated_data, data_read[0])

#save generated data
saving_data = []
for data_point in generated_data:
	data_point_temp=list(data_point)
	data_point_temp.insert(3,328.15)
	saving_data.append(data_point_temp)

pd.DataFrame(np.array(saving_data)).to_csv(os.path.join("Experimental " + 
"Interpolated Results", ALL_DATA_PATH[:-4]+"_Gen.csv"), index=False, header=False)
