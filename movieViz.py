from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, callback
import pandas as pd
import plotly.express as px
import numpy as np

#%% Import data
df_ratings = pd.read_csv("data/ml-32m/ratings.csv").head(1000)
df_movies = pd.read_csv("data/ml-32m/movies.csv").head(1000)
df_tags = pd.read_csv("data/ml-32m/tags.csv").head(1000)
df_links = pd.read_csv("data/ml-32m/links.csv").head(1000)

#%% Pandas processing
#Add/edit new columns
df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], unit='s') #change timestamp to datetime
df_ratings['review_year'] = df_ratings['timestamp'].dt.year #add review year


movie_Series = df_movies['title'] #Series of movies titles + movieIDs
movie_list = movie_Series.values #List of movie titles

#Natasha part
df_genre = pd.merge(df_ratings, df_movies, on="movieId")
df_genre['genres'] = df_genre['genres'].str.split('|')
df_genre = df_genre.explode('genres')

all_genres = sorted(df_genre['genres'].unique())
top_10_genres = df_genre['genres'].value_counts().head(10).index.tolist()
#%%sandboxing cell for development purposes, delete or comment out before commit
import numpy as np
import pandas as pd
import plotly.express as px

x0 = np.random.randn(250)
# Add 1 to shift the mean of the Gaussian distribution
x1 = np.random.randn(500) + 1

df = pd.DataFrame(dict(
    series=np.concatenate((["a"] * len(x0), ["b"] * len(x1))), 
    data=np.concatenate((x0, x1))
))

df

px.histogram(df, x="data", color="series", barmode="overlay")

#%% Run the app
app = Dash()

# Requires Dash 2.17.0 or later
app.layout = [
    #Movie Specific Section
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
    dcc.Graph(id="ratings_over_time", figure=None),
    #TODO:Add more graphs

    #Genre Section
    html.Div([
    html.H1("Movie Ratings by Genre"),

    # Row with label and reset button side by side
    html.Div([
        html.Label("Select genres to compare:", style={'marginRight': '10px'}),
        html.Button("Reset", id='reset-button', n_clicks=0)
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),

    # Genre checklist
    dcc.Checklist(
        id='genre-selector',
        options=[{'label': genre, 'value': genre} for genre in all_genres],
        value=top_10_genres,
        inline=True
    ),

    dcc.Graph(id='genre-rating-graph'),
    dcc.Graph(id='genre_histogram'),

    html.Hr(),
    html.H3("Tags Data Table"),
    dash_table.DataTable(data=df_tags.to_dict('records'), page_size=10)
])]


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
        movieId = df_movies.loc[df_movies['title'] == title, 'movieId'].iloc[0]

        #Rating Histogram Plot Creation
        ratings = df_ratings.loc[df_ratings['movieId'] == movieId]
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
    
# Callback to update genres when reset is clicked
@app.callback(
    Output('genre-selector', 'value'),
    Input('reset-button', 'n_clicks'),
    prevent_initial_call=True
)
def reset_genres(n_clicks):
    return top_10_genres

# Callback to update graph
@app.callback(
    Output('genre-rating-graph', 'figure'),
    Output('genre_histogram','figure'),
    Input('genre-selector', 'value')
)
def update_graph(selected_genres):
    if not selected_genres:
        return px.bar(title="No genres selected"), None
    
    filtered_df = df_genre[df_genre['genres'].isin(selected_genres)]
    genre_avg_rating = filtered_df.groupby('genres')['rating'].mean().reset_index()
    genre_avg_rating = genre_avg_rating.rename(columns={'rating': 'average_rating'})
    genre_avg_rating = genre_avg_rating.sort_values(by='average_rating', ascending=False)

    #rating distribution of selected genres
    genre_rating_histogram = px.histogram(filtered_df, x="rating", color='genres', barmode='overlay', title='Rating Distribution')

    genre_rating_bar_chart = px.bar(
        genre_avg_rating,
        x='genres',
        y='average_rating',
        text='average_rating',
        title='Average Rating per Selected Genre',
        labels={'genres': 'Genre', 'average_rating': 'Average Rating'}
    )
    genre_rating_bar_chart.update_traces(texttemplate='%{text:.2f}', textposition='inside')
    return genre_rating_bar_chart, genre_rating_histogram




if __name__ == '__main__':
    app.run(debug=True)