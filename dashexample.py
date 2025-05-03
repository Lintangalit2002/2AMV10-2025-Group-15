#%% Import libraries
from dash import Dash, html, dash_table
import pandas as pd
import plotly.express as px


#%% Import data
df = pd.read_csv("data/ml-32m/tags.csv")
df = df.head(100)

#%% Pandas processing

#%% Run the app
app = Dash()

# Requires Dash 2.17.0 or later
app.layout = [
    html.Div(children='Hello World'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10)
    ]

if __name__ == '__main__':
    app.run(debug=True)