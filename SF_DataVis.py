from plotly import graph_objects as go
import pandas as pd


year = []
avg_sea_temp = []
PDI = []


LIST = pd.read_csv("SF_data.csv").values.tolist()
for triple in LIST:
    year.append(triple[0])
    avg_sea_temp.append(triple[1])
    PDI.append(triple[2])

fig = go.Figure(data = go.Scatter(x = sorted(avg_sea_temp), y=sorted(PDI), mode='lines+markers'))
fig.update_layout(
    title=go.layout.Title(
        text="Correlation between Hurricane Intensity and Sea Surface Temperature",
        xref="paper",
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Mean Sea Surface Temperature (F)",
            font=dict(
                family="Open Sans",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Mean Hurricane Intensity (PDI)",
            font=dict(
                family="Open Sans",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)
fig.show()