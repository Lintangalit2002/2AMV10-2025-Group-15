#%% Import libraries
from dash import Dash, dcc, html, Input, Output, callback, dash_table, State
import pandas as pd
import plotly.express as px


#%% Import data



ratings_df = pd.read_csv("data/ml-32m/ratings.csv")
movie_df = pd.read_csv("data/ml-32m/movies.csv")

df = movie_df
df = df.head(100)

#%% Pandas processing
movie_Series = movie_df['title'] #Series of movies titles + movieIDs
movie_list = movie_Series.values #List of movie titles

#%% Run the app
app = Dash()

# Requires Dash 2.17.0 or later
app.layout = [
    html.H6("Insert Movie and year:"),
    html.Div([
        dcc.Input(
        id = "movie-field",
        type = 'text',
        ),
        dcc.Input(
        id = "year-field",
        type = 'text',
        )
    ]),

    html.Button('Search', id='submit-values', n_clicks=0),

    dash_table.DataTable(data=df.to_dict('records'), page_size=10),

    html.Div(id = 'movie-name', children="Movie Here")
    ]

#Button Logic
@callback(
    Output('movie-name', 'children'),
    Input('submit-values', 'n_clicks'),
    State('movie-field', 'value'),
    State('year-field', 'value'),
    prevent_initial_call=True
)
def update_output(n_clicks,movie,year):
    movieID = 0
    title = movie + " (" + year + ")"

    if title in movie_list:
        movieID = list(movie_Series).index(title)

    if movieID != 0:
        return 'The movieID is {}'.format(
            movieID,
        )
    else:
        return 'Movie Not Found'



if __name__ == '__main__':
    app.run(debug=True)
# %%
