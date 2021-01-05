import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
from dash.dependencies import Output, Input
import plotly.graph_objects as go
from maindash import app


def create_data():
    df = pd.read_csv('dataset.txt',
                     names=["day", "Humidity", "Temperature"],
                     header=None)

    df[['day', 'Humidity']] = df["day"].str.split(" ", 1, expand=True)
    df[['Humidity', 'Temperature']] = df["Humidity"].str.split(" ", 1, expand=True)

    df['day'] = pd.to_datetime(df['day'], format='%d/%m/%Y', dayfirst=True)

    df['Humidity'] = df['Humidity'].astype(float)
    df['Temperature'] = df['Temperature'].astype(float)

    df = df.groupby("day").mean().reset_index()
    return df


def create_figure(column_name):
    df = create_data()
    figure = go.Figure()
    trace = go.Scatter(x=df['day'], y=df[column_name], name=str(column_name) + ' values',
                       mode='lines+markers', marker_color='rgba(50,100,250,.8)')

    layout = dict(title="Plot showing " + str(column_name) + " values over time", xaxis_title="Time",
                  yaxis_title=column_name, showlegend=True, font_family='Bahnschrift')

    figure.add_trace(trace)
    figure.update_layout(layout)
    return figure


def homepage_layout():
    app.layout = html.Div(id='appmain', style={'font-family': 'Bahnschrift'},
                          children=[
                              html.A("View table data", href='http://127.0.0.1:5000/table', target="_blank"),
                              html.Br(),
                              html.Br(),
                              html.A("Go to predictions", href='http://127.0.0.1:5000/predictions', target="_blank"),
                              html.Br(),
                              html.Br(),
                              html.A("Go to future predictions", href='http://127.0.0.1:5000/future', target="_blank"),
                              html.H1(children='Plotting Air Quality',
                                      style={'color': 'rgba(50,100,250,.8)', 'text-align': 'center'}),
                              dcc.Dropdown(id='dropdown',
                                           placeholder='Select a value for the y axis',
                                           options=[{'label': "Temperature", 'value': "Temperature"},
                                                    {'label': "Humidity", 'value': "Humidity"}]
                                           ),
                              html.Div(dcc.Graph(id='figure')),
                              dcc.Interval(
                                  id='interval-component',
                                  interval=1 * 5000,  # in milliseconds
                                  n_intervals=0
                              ),
                              html.Div(id='page-content')
                          ]
                          )
    return app.layout


@app.callback(Output(component_id='figure', component_property='figure'),
              [Input(component_id='dropdown', component_property='value'), Input('interval-component', 'n_intervals')])
def choose_yaxis(column_name, n):
    if column_name is not None:
        figure = create_figure(column_name)
    else:
        figure = create_figure('Temperature')
    return figure
