from Resampling_SVM import ResampleSVM

import pandas as pd
from multiprocessing import Pool
import time

def FormatDataFromCSV(stringfileName):
    """
    The format of the csv should have the column order: x, y, z, temp, phase.
    :param stringfileName: the fileName of csv
    :return: a tuple:
    first element is temperature
    second element is a list of tuples in form: (x, y, z, PHASE)
    """

    # Assume all data points given are at same temperature!

    data_list = pd.read_csv(stringfileName).values.tolist()
    return (data_list[0][3], [p for p in list(map(lambda x : [x[0], x[1], x[2], x[4]], data_list))])



start = time.time()

ALL_DATA_PATH = "Improved_T300.csv"
data_read = FormatDataFromCSV(ALL_DATA_PATH)
SVM = ResampleSVM(data_read[1])
print("Data read in: " + str(time.time() - start) + " seconds")
pool = Pool()
all = []
for C in range(200, 25000, 200):
    multiple_results = [pool.apply_async(SVM.get_avg_kFold_partitn, (g/100, C)) for g in range(2, 150, 1)]
    all += [res.get(timeout=1) for res in multiple_results]
    print("Cost iteration took " + str(time.time() - start) + " seconds")
print("All parameter calculations finished at: " + str(time.time() - start) + " seconds")


sort_time = time.time()
all_sorted = sorted(all, key=lambda x:x[-1])
print("Sorted the error and parameter combination list in " + str(time.time() - sort_time) + " seconds")


data_out = {}
data_out["g"] = []
data_out["C"] = []
data_out["ERR"] = []
for e in all_sorted:
    data_out["g"].append(e[0])
    data_out["C"].append(e[1])
    data_out["ERR"].append(e[2])

df = pd.DataFrame(all_sorted, columns=['g', 'C', 'ERR'])
df.to_csv("T300-Error-ConfidenceApplied.csv")
print("Data exported and finished program in " + str(time.time() - start) + " seconds")
print("Finished all three loops in " + str(time.time() - start) + " seconds")
