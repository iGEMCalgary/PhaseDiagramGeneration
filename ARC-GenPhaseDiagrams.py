import pandas as pd
from multiprocessing import Pool
from Resampling_SVM import ResampleSVM


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



parameter_comb = [(x, (10200/0.71)*x) for x in [v/100 for v in range(1, 180, 3)]]
train = FormatDataFromCSV("Improved_T300.csv")
SVM = ResampleSVM(train)
pool = Pool()
multiple_results = [pool.apply_async(SVM.generate_all_phase, (train, comb[0], comb[1], 0.01)) for comb in parameter_comb]
all = [res.get(timeout=1) for res in multiple_results]

data_out = {}
data_out['x'] = []
data_out['y'] = []
data_out['z'] = []
data_out['PARAM_INDEX'] = []
data_out['PHASE'] = []
index = 0
for phaseDIAG in all:
    for p in phaseDIAG:
        data_out['x'].append(p[0])
        data_out['y'].append(p[1])
        data_out['z'].append(p[2])
        data_out['PARAM_INDEX'].append(index)
        data_out['PHASE'].append(p[3])
    index += 1

df = pd.DataFrame(data_out, columns=['x', 'y', 'z', 'PARAM_INDEX', 'PHASE'])
df.to_csv("ALL-Parameter-Phases.csv", index=False)
