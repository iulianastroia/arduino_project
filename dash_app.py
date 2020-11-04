import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('dataset.txt',
                 names=["Day", "Humidity", "Temperature"],
                 header=None)

df[['Day', 'Humidity']] = df["Day"].str.split(" ", 1, expand=True)
df[['Humidity', 'Temperature']] = df["Humidity"].str.split(" ", 1, expand=True)

df['Day'] = pd.to_datetime(df['Day'], format='%d/%m/%Y',dayfirst=True)

df['Humidity'] = df['Humidity'].astype(float)
df['Temperature'] = df['Temperature'].astype(float)

df=df.groupby("Day").mean().reset_index()
print(df)

fig = go.Figure(data=go.Scatter(x=df['Day'], y=df['Temperature'], mode='lines+markers'))
fig.show()
fig = go.Figure(data=go.Scatter(x=df['Day'], y=df['Humidity'], mode='lines+markers'))
fig.show()