from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, callback
import pandas as pd
import plotly.express as px
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.manifold import TSNE

#%% Import data
df_ratings = pd.read_csv("data/ml-32m/ratings.csv").head(10000)
df_movies = pd.read_csv("data/ml-32m/movies.csv").head(10000)
df_tags = pd.read_csv("data/ml-32m/tags.csv").head(10000)
df_links = pd.read_csv("data/ml-32m/links.csv").head(10000)
df_genre = pd.read_csv("data/df_genre_ratings.csv").head(10000)
df_combined = pd.read_csv("data/df_combined.csv").head(10000)

# determine unique genres and languages
df_combined['genres'] = df_combined['genres'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
all_unique_genres = sorted(set(g for sublist in df_combined['genres'] for g in sublist if isinstance(sublist, list)))
unique_languages = sorted(df_combined['original_language'].dropna().unique())


#%% Pandas processing
#Add/edit new columns
df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], unit='s') #change timestamp to datetime
df_ratings['review_year'] = df_ratings['timestamp'].dt.year #add review year

movie_Series = df_movies['title'] #Series of movies titles + movieIDs
movie_list = movie_Series.values #List of movie titles

all_genres = sorted(df_genre['genres'].unique())
top_10_genres = df_genre['genres'].value_counts().head(10).index.tolist()
#%%sandboxing cell for development purposes, delete or comment out before commit
#ratings = ratings_df.loc[ratings_df['movieId'] == 5445]

movie = 'Heat'
year = '1995'
title = movie + " (" + year + ")"

movieID = df_movies.loc[df_movies['title'] == title, 'movieId'].iloc[0]
movieID

#%% Run the app
app = Dash()

# Requires Dash 2.17.0 or later
app.layout = [
    dcc.Store(id='tsne-data'),
    
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

    # html.Hr(),
    # html.H3("Tags Data Table"),
    # dash_table.DataTable(data=df_tags.to_dict('records'), page_size=10),

    
    html.Hr(),
    html.H1("Movie Map"),
    html.Div([
        html.Div([
            html.H1("Insert Movie and year:"),
            dcc.Input(
            id = "movie-field",
            type = 'text',
            ),
            dcc.Input(
            id = "year-field",
            type = 'text',
            ),
            html.Button('Search', id='submit-values', n_clicks=0),
            
            html.Button('Find in Graph', id='find-in-graph',n_clicks = 0), #give coordinates of movie in graph if it exists

            html.Div(id = 'movie-name', children="The movieID is:"),
            html.Div(id = 'movie-coordinates', children='The coordinates are: (x,y)'),

            dcc.Graph(id="histogram_ratings", figure=None),
            dcc.Graph(id="ratings_over_time", figure=None),
        ]),

        #dash_table.DataTable(data=df.to_dict('records'), page_size=10),

        
        # filters on the left
        html.Div([
            html.Label("Filter by genre:"),
            dcc.Dropdown(
                id='genre-filter',
                options=[{'label': g, 'value': g} for g in all_unique_genres],
                multi=True,
                placeholder="Select genre(s)..."
            ),
            html.Label("Original Language:"),
            dcc.Dropdown(
                id='language-filter',
                options=[{'label': l, 'value': l} for l in unique_languages],
                multi=True,
                placeholder="Select language(s)..."
            ),
            html.Label("Release Year:"),
            dcc.RangeSlider(
                id='year-filter',
                min=df_combined['year'].min(),
                max=df_combined['year'].max(),
                step=1,
                value=[df_combined['year'].min(), df_combined['year'].max()],
                marks={y: str(y) for y in range(int(df_combined['year'].min()), int(df_combined['year'].max()) + 1, 10)}
            ),
            html.Label("Runtime (minutes):"),
            dcc.RangeSlider(
                id='runtime-filter',
                min=int(df_combined['runtime'].min()),
                max=int(df_combined['runtime'].max()),
                step=1,
                value=[int(df_combined['runtime'].min()), int(df_combined['runtime'].max())],
                marks={
                    int(df_combined['runtime'].min()): str(int(df_combined['runtime'].min())),
                    int(df_combined['runtime'].max()): str(int(df_combined['runtime'].max()))
                }
            ),
            html.Label("Budget ($):"),
            dcc.RangeSlider(
                id='budget-filter',
                min=int(df_combined['budget'].min()),
                max=int(df_combined['budget'].max()),
                step=1_000_000,
                value=[int(df_combined['budget'].min()), int(df_combined['budget'].max())],
                marks={
                    int(df_combined['budget'].min()): f"${int(df_combined['budget'].min()):,}",
                    int(df_combined['budget'].max()): f"${int(df_combined['budget'].max()):,}"
                }
            ),
            html.Label("Average Rating:"),
            dcc.RangeSlider(
                id='rating-filter',
                min=0,
                max=5,
                step=0.1,
                value=[0, 5],
                marks={i: str(i) for i in range(6)}
            ),
        ], style={'flex': '1', 'padding': '20px', 'minWidth': '300px'}),

        # plot and recomputing t-sne on the right
        html.Div([
            dcc.Graph(id='tsne-plot', figure=None),
            html.Div(id='tsne-click-info', style={'marginTop': '10px', 'fontSize': '16px'}),

            html.Hr(),
            html.H3("t-SNE Settings"),
            html.Label("Select features to include:"),
            dcc.Checklist(
                id='tsne-features',
                options=[
                    {'label': 'Genres', 'value': 'genres'},
                    {'label': 'Original Language', 'value': 'language'},
                    {'label': 'Runtime', 'value': 'runtime'},
                    {'label': 'Budget', 'value': 'budget'},
                    {'label': 'Average Rating', 'value': 'average_rating'},
                    {'label': 'Year', 'value': 'year'}
                ],
                value=['genres', 'language', 'runtime', 'budget', 'average_rating', 'year']
            ),
            html.Div([
                html.Label("Perplexity (5–100):"),
                dcc.Input(
                    id='tsne-perplexity',
                    type='number',
                    min=5,
                    max=100,
                    step=1,
                    value=30,
                    style={'width': '80px'}
                ),
                html.Div("Must be between 5 and 100", style={'fontSize': '12px', 'color': '#666'})
            ]),
            html.Button("Compute t-SNE", id='tsne-run-button', n_clicks=0),
            html.Div(id='tsne-status', style={'marginTop': '10px', 'color': 'green'}),
        ], style={'flex': '3', 'padding': '20px'})
    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'})


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
    Input('genre-selector', 'value')
)
def update_graph(selected_genres):
    if not selected_genres:
        return px.bar(title="No genres selected")
    
    filtered_df = df_genre[df_genre['genres'].isin(selected_genres)]
    genre_avg_rating = filtered_df.groupby('genres')['rating'].mean().reset_index()
    genre_avg_rating = genre_avg_rating.rename(columns={'rating': 'average_rating'})
    genre_avg_rating = genre_avg_rating.sort_values(by='average_rating', ascending=False)

    fig = px.bar(
        genre_avg_rating,
        x='genres',
        y='average_rating',
        text='average_rating',
        title='Average Rating per Selected Genre',
        labels={'genres': 'Genre', 'average_rating': 'Average Rating'}
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')
    return fig

# For t-SNE
@app.callback(
    Output('tsne-plot', 'figure'),
    Input('tsne-data', 'data'),
    Input('genre-filter', 'value'),
    Input('rating-filter', 'value'),
    Input('year-filter', 'value'),
    Input('runtime-filter', 'value'),
    Input('budget-filter', 'value'),
    Input('language-filter', 'value')
)
def update_tsne_plot(data, selected_genres, rating_range, year_range, runtime_range, budget_range, selected_languages):
    if not data or 'x' not in data:
        return px.scatter(title="Run t-SNE to generate plot.")

    df_valid = pd.DataFrame(data)

    if isinstance(df_valid['genres'].iloc[0], str):
        df_valid['genres'] = df_valid['genres'].apply(ast.literal_eval)

    df_valid = df_valid[
        (df_valid['average_rating'] >= rating_range[0]) &
        (df_valid['average_rating'] <= rating_range[1]) &
        (df_valid['year'] >= year_range[0]) &
        (df_valid['year'] <= year_range[1]) &
        (df_valid['runtime'] >= runtime_range[0]) &
        (df_valid['runtime'] <= runtime_range[1]) &
        (df_valid['budget'] >= budget_range[0]) &
        (df_valid['budget'] <= budget_range[1])
    ]

    if selected_genres:
        df_valid = df_valid[df_valid['genres'].apply(lambda g: all(genre in g for genre in selected_genres))]
    if selected_languages:
        df_valid = df_valid[df_valid['original_language'].isin(selected_languages)]

    fig = px.scatter(
        df_valid,
        x='x',
        y='y',
        hover_name='title',
        hover_data={
            'x': False, 'y': False,
            'genres': True,
            'year': True,
            'average_rating': True,
            'runtime': True,
            'budget': True
        },
        color='average_rating',
        title='Movie Map (Filtered)'
    )
    fig.update_layout(dragmode='zoom')
    fig.update_traces(marker=dict(size=6))
    return fig


@app.callback(
    Output('tsne-data', 'data'),
    Output('tsne-status', 'children'),
    Input('tsne-run-button', 'n_clicks'),
    State('tsne-features', 'value'),
    State('tsne-perplexity', 'value'),
    prevent_initial_call=True
)
def update_tsne(n_clicks, selected_features, perplexity):
    df = df_combined.copy()
    feature_blocks = []

    if 'genres' in selected_features:
        df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else str(x).split('|'))
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(df['genres'])
        genre_df = pd.DataFrame(genre_matrix, index=df.index)
        feature_blocks.append(genre_df.values)

    if 'language' in selected_features:
        language_df = pd.get_dummies(df['original_language'], prefix='lang')
        feature_blocks.append(language_df.values)

    if any(f in selected_features for f in ['runtime', 'budget', 'average_rating', 'year']):
        numeric_cols = [col for col in ['runtime', 'budget', 'average_rating', 'year'] if col in selected_features]
        numeric_df = df[numeric_cols].fillna(0)
        feature_blocks.append(numeric_df.values)

    if not feature_blocks:
        return dash.no_update, "No features selected. Please select at least one."

    full_feature_matrix = np.hstack(feature_blocks)
    scaled = StandardScaler().fit_transform(full_feature_matrix)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(scaled)

    return (
        {
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "title": df['title'].tolist(),
            "genres": df['genres'].tolist(),
            "year": df['year'].tolist(),
            "average_rating": df['average_rating'].tolist(),
            "runtime": df['runtime'].tolist(),
            "budget": df['budget'].tolist(),
            "original_language": df['original_language'].tolist()
        },
        f"t-SNE completed with {len(df)} points using perplexity {perplexity}."
    )


@app.callback(
        Output('movie-coordinates','children'),
        Input('find-in-graph','n_clicks'),
        State('tsne-data','data'),
        State('movie-field','value'),
        prevent_initial_call=True
)
def find_movie(n_clicks,data,movie_name):
    df_find = pd.DataFrame(data)

    x = df_find.loc[df_find['title'] == movie_name, 'x']
    y = df_find.loc[df_find['title'] == movie_name, 'y']

    return "The coordinates are: (" + str(x.iloc[0]) + "," + str(y.iloc[0]) + ")"

#%%
if __name__ == '__main__':
    app.run(debug=True)