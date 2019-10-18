#from DisplayTernaryPhaseDiag import Display
from SVM.Resampling_SVM import ResampleSVM
from DisplayTernaryPhaseDiag import Display
import pandas as pd
from plotly import graph_objects as go
import plotly
display = Display()
SVM = ResampleSVM(display.FormatDataFromCSV("PD_P1_SVM/Experimental Data P1/ER2_T300.csv")[1])



#online = [(p[1], p[2]) for p in data_in if p[1] * 10200/0.71 - 250 < p[2] < p[1] * 10200/0.71 + 250]
D = {(p[1], p[2], p[3]): p[4] for p in pd.read_csv("Confidence300.csv").values.tolist()}

data_out = {}
data_out["surf"] = []
data_out["oil"] = []
data_out["water"] = []
for p in SVM.generate_all_phase(SVM.data_in, 1.39, 2400, 0.005):
    if p[-1] == 1 and 0.06 <= p[0] <= 0.55:
        if (p[0], p[1], p[2]) in D and D[(p[0], p[1], p[2])] == 1.0:
            data_out["surf"].append(p[2])
            data_out["oil"].append(p[1])
            data_out["water"].append(p[0])

df = pd.DataFrame(data_out, columns=['water', 'oil', 'surf'])
df.to_csv("Winsor1-WaterRestricted-CONF1.csv")


def display_All_T_Confidence():
    """
    FOR ConFIDENCE HEAT MAP
    """
    for s in ('300', '310', '323', '343'):
        SVM = ResampleSVM(display.FormatDataFromCSV("PD_P1_SVM/Experimental Data P1/ER2_T" + s +".csv"))
        display.DisplayTernaryPhaseScatter(display.FormatDataFromCSV("PD_P1_SVM/Experimental Data P1/ER2_T" + s + ".csv")[1], 10, s + " Lab Data")
        optimal = pd.read_csv("T" + s + "-OUT.csv").values.tolist()[0][1:3]
        print("Optimal Parameters used: ")
        print(optimal)
        display.DisplayTernaryPhaseScatter(SVM.generate_all_phase(SVM.data_in[1], optimal[0], optimal[1],.01), 7, s + " SVM Optimized")
        data_in = pd.read_csv("Confidence" + s +".csv").values.tolist()

        conf = list(map(lambda x:x[4], data_in))
        fig = go.Figure(go.Scatterternary({
            'mode': 'markers',
            'a': list(map(lambda x:x[3], data_in)),
            'b': list(map(lambda x:x[2], data_in)),
            'c': list(map(lambda x:x[1], data_in)),
            'text': conf,
            'marker': {
                'symbol': 0,
                'color': conf,
                'size': 7,
                'colorbar': dict(title="Colorbar"),
                'colorscale': "Viridis"}
            })
        )

        fig.update_layout(
            title=go.layout.Title(
                text="Confidence T-"+s))
        plotly.offline.plot(fig, filename="ConfidenceT" + s + ".html")


#display_All_T_Confidence()