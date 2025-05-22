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
    html.H1("Insert Movie and year:"),
    #TODO:Insert suggested dropdown so users dont have to know the specific movie
    html.Div([
        
        dcc.Input(
        id = "movie-field",
        type = 'text',
        ),
        dcc.Input(
        id = "year-field",
        type = 'text',
        ),
        html.Button('Search', id='submit-values', n_clicks=0)
    ]),

    #dash_table.DataTable(data=df.to_dict('records'), page_size=10),

    html.Div(id = 'movie-name', children="The movieID is:"),

    dcc.Graph(id="histogram_ratings", figure=None)
    #TODO:Add more graphs
    ]

#Button Logic, can add more input fields and output graphs etc. later
@callback(
    Output('movie-name', 'children'),
    Output('histogram_ratings', 'figure'),
    Input('submit-values', 'n_clicks'),
    State('movie-field', 'value'),
    State('year-field', 'value'),
    prevent_initial_call=True
) #Must have one parameter for each input or state
def update_output(n_clicks,movie,year):
    movieId = 0
    title = movie + " (" + year + ")"

    if title in movie_list:
        #Get movieID
        movieId = list(movie_Series).index(title) + 1 #Index of data starts with 1

        #Histogram Plot Creation
        ratings = ratings_df.loc[ratings_df['movieId'] == movieId]
        histogram = px.histogram(ratings, x="rating")

        #TODO: Insert more plots below

    #Must have one return variable for each defined Output
    if movieId != 0:
        return 'The movieID is: {}'.format(
            movieId,
        ), histogram
    else:
        return 'Movie Not Found', None



if __name__ == '__main__':
    app.run(debug=True)
# %%
