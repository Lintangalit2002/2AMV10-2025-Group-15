#%% Import libraries
from dash import Dash, dcc, html, Input, Output, callback, dash_table, State
import pandas as pd
import plotly.express as px
import numpy as np


#%% Import data
ratings_df = pd.read_csv("data/ml-32m/ratings.csv")
movie_df = pd.read_csv("data/ml-32m/movies.csv")

#%% Pandas processing
#Add/edit new columns
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s') #change timestamp to datetime
ratings_df['review_year'] = ratings_df['timestamp'].dt.year #add review year


movie_Series = movie_df['title'] #Series of movies titles + movieIDs
movie_list = movie_Series.values #List of movie titles

#%%sandboxing cell for development purposes, delete or comment out before commit
#ratings = ratings_df.loc[ratings_df['movieId'] == 5445]

# movie = 'Heat'
# year = '1995'
# title = movie + " (" + year + ")"

# movieID = movie_df.loc[movie_df['title'] == title, 'movieId'].iloc[0]
# movieID




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

    dcc.Graph(id="histogram_ratings", figure=None),
    dcc.Graph(id="ratings_over_time", figure=None)
    #TODO:Add more graphs
    ]

#Button Logic, can add more input fields and output graphs etc. later
@callback(
    Output('movie-name', 'children'),
    Output('histogram_ratings', 'figure'),
    Output('ratings_over_time', 'figure'),
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
        movieId = movie_df.loc[movie_df['title'] == title, 'movieId'].iloc[0]

        #Rating Histogram Plot Creation
        ratings = ratings_df.loc[ratings_df['movieId'] == movieId]
        rating_histogram = px.histogram(ratings, x="rating", title='Rating Distribution')


        #Rating Over Time Plot Creation
        #Group Timestamps
        yearly_ratings = ratings.groupby(['review_year']).mean()
        rating_over_time_movie = px.line(yearly_ratings, x=yearly_ratings.index, y='rating', 
                                         range_y = [0,5], title='Rating Over Time')

        #TODO: Insert more plots

    #Must have one return variable for each defined Output
    if movieId != 0:
        return 'The movieID is: {}'.format(
            movieId,
        ), rating_histogram, rating_over_time_movie
    else:
        return 'Movie Not Found', None, None



if __name__ == '__main__':
    app.run(debug=True)
# %%
