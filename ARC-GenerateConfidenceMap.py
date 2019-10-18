from SVM.Resampling_SVM import ResampleSVM
import time
from multiprocessing import Pool
import pandas as pd
import math

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


def Fn_switches(phase_list):
    # Return a confidence score solely on the number of switches within the phase list
    n_switches = 0
    last_phase = phase_list[0]
    for i in range(1, len(phase_list)):
        if last_phase != phase_list[i]: n_switches += 1
        last_phase = phase_list[i]
    return 1 / (1 + math.exp(0.5*(n_switches - 3.5)))


def Fn_distinct_phases(phase_list):
    x = len(set(phase_list))
    if x < 2:
        return 0
    return 0.25 * (x - 2)**0.5


total = time.time()

parameter_comb = [(x, (10200/0.71)*x) for x in [v/100 for v in range(1, 180)]]

for s in ['310', '323', '343']:
    start = time.time()
    ALL_DATA_PATH = "PD_P1_SVM/Experimental Data P1/ER2_T" + s + ".csv"
    data_read = FormatDataFromCSV(ALL_DATA_PATH)
    SVM = ResampleSVM(data_read[1])
    get_results = [SVM.generate_all_phase(data_read[1], comb[0], comb[1], 0.01) for comb in parameter_comb]

    # Now get_results is a list of phase diagrams of increasing flexibility! Large list huh

    print("Finished generating 180 phase diagrams of increasing overfitness in " + str(time.time() - start) + " seconds.")
    start = time.time()
    all_dict = []
    for slice in get_results:
        all_dict.append({(p[0], p[1], p[2]) : p[-1] for p in slice})

    data_out = {}
    data_out['x'] = []
    data_out['z'] = []
    data_out['y'] = []
    data_out['CONFD'] = []

    for i in range(len(get_results[0])):
        data_out['x'].append(get_results[0][i][0])
        data_out['y'].append(get_results[0][i][1])
        data_out['z'].append(get_results[0][i][2])
        phase_list = [slice[(get_results[0][i][0],get_results[0][i][1], get_results[0][i][2])] for slice in all_dict]

        fminusg = Fn_switches(phase_list) - Fn_distinct_phases(phase_list)
        if str(fminusg)[:4] == '0.85':
            fminusg = 1
        data_out['CONFD'].append(max(fminusg, 0))

    df = pd.DataFrame(data_out, columns=['x', 'y', 'z', 'CONFD'])
    df.to_csv("Confidence" + s + ".csv")
    print("Finished loop " + s + " and calculated confidences in " + str(time.time() - start) + " seconds.")
print("Total run time: " + str(time.time() - total) + " seconds.")

