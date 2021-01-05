import pandas as pd
import dash
import dash_html_components as html
import json
from dash.dependencies import Input, Output
import dash_core_components as dcc
from dash_table import DataTable

from homepage import create_data

app = dash.Dash(__name__)



# read json file for first load
df = create_data()

def table_layout():
    app.layout = html.Div(style={'fontFamily': 'Bahnschrift'},
                          children=[
                              html.H1(children='Temperature and Humidity Data',
                                      style={'color': 'rgba(0,102,102,1)', 'textAlign': 'center'}),
                              DataTable(
                                  id='table',

                                  # center text in cell
                                  style_cell={'textAlign': 'center', 'border': '1px solid black'},

                                  # header color and font
                                  style_header={
                                      'backgroundColor': 'rgba(0,210,210,1)', 'fontWeight': 'bold'
                                  },

                                  style_table={'margin': 'auto',
                                               'width': '80%',
                                               'height': '80%',
                                               'padding': '50px'},

                                  # create stripped datatable
                                  style_data_conditional=[
                                      # selected cell
                                      {
                                          "if": {"state": "selected"},  # 'active' | 'selected'
                                          "backgroundColor": "rgba(0,170,170,1)",
                                          "border": "1px solid blue",
                                      },

                                      # even cells have different color
                                      {
                                          'if': {'row_index': 'even'},
                                          'backgroundColor': 'rgba(193,245,245,1)'
                                      }
                                  ],

                                  columns=[{"name": column_name, "id": column_name} for column_name in df.columns],

                                  # get values from dataframe for the rows
                                  data=df.to_dict('records'),

                                  # don't display border
                                  style_as_list_view=True)
                              ,
                              # used to update the contents of the table after updating txt file
                              dcc.Interval(id='interval', interval=1000, n_intervals=0),
                          ])
    return app.layout


@app.callback([Output("table", "data")],
              [Input('interval', 'n_intervals')])
def update_table(n_intervals):
    # update df from file
    df = create_data()
    # return updated values
    return [df.to_dict('records')]


if __name__ == "__main__":
    app.run_server(debug=True, port=8022)