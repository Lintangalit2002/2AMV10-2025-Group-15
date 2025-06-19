#%% Imports
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, callback, MATCH, ALL
import pandas as pd
import plotly.express as px
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

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

min_year, max_year = df_combined['year'].min(), df_combined['year'].max()
min_budget,max_budget = df_combined['budget'].min(), df_combined['budget'].max()
min_rating,max_rating = df_combined['average_rating'].min(), df_combined['average_rating'].max()
min_runtime,max_runtime = df_combined['runtime'].min(), df_combined['runtime'].max()




#%%sandboxing cell for development purposes, delete or comment out before commit

#%% Run the app
app = Dash()

# Requires Dash 2.17.0 or later
app.layout = [
    dcc.Store(id='tsne-data'),
    dcc.Store(id='highlighted-movie'),
    
    html.Div([
        html.H1("Movie Exploration Tool", style={
            'textAlign': 'center',
            'fontSize': '36px',
            'marginBottom': '10px'
        }),
        html.Hr(style={'borderTop': '3px solid black'}),
        
        html.H2("Movie Ratings by Genre", style={
            'textAlign': 'center',
            'fontSize': '26px',
            'marginBottom': '20px'
        }),

    html.Div([
        html.Label("Select genres to compare:", style={'marginRight': '10px'}),
        html.Button("Reset", id='reset-button', n_clicks=0),
        html.Button("Select All", id='select-all-genres', n_clicks=0, style={'marginLeft': '10px'}),
        html.Button("Clear All", id='clear-all-genres', n_clicks=0, style={'marginLeft': '10px'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '10px'}),

    dcc.Checklist(
        id='genre-selector',
        options=[{'label': genre, 'value': genre} for genre in all_genres],
        value=top_10_genres,
        inline=True
    ),

    dcc.Graph(id='genre-rating-graph'),
    dcc.Graph(id='genre-histogram'),

    html.Hr(),
    html.H2("Movie Map", style={
        'textAlign': 'center',
        'fontSize': '26px',
        'marginBottom': '20px'
    }),

    html.Div([
        # left: ratings
        html.Div([
            html.H2("Insert Movie and year:", style={
                'fontSize': '26px',
                'marginBottom': '10px'
            }),
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
            
            html.Div([
            dcc.Graph(id="histogram_ratings", figure=None),
            dcc.Graph(id="ratings_over_time", figure=None),
            ]),
        ],style={'flex': '1', 'padding': '20px', 'minWidth': '400px'}), 

        # right: t-sne, settings, filters
        html.Div([
            dcc.Graph(id='tsne-plot', figure=None),
            html.Div(id='tsne-click-info', style={'marginTop': '10px', 'fontSize': '16px'}),

            html.Hr(),
            html.Div([
                html.Div([
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
                ], style={'flex': '1', 'padding': '10px', 'minWidth': '300px', 'maxWidth': '500px', 'alignItems': 'center'}),

                html.Div([
                    html.H3("Movie Map Filters"),
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
                        marks={},
                        tooltip={"placement": "bottom", "always_visible": True}
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
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
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
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
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
                ], style={'flex': '2', 'padding': '10px', 'maxWidth': '500px'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'})
        ], style={'flex': '2'})
    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'}),

    html.Div([
                html.H1("Selected Movies:"),
                html.Div(id='checkbox-container'),
                html.H2("Similarity Feature Weights"),
                html.Div([
                    
                    dcc.Input(
                    id = "budget-weight",
                    type = 'number',
                    placeholder="Budget Weight...",
                    value = 1
                    ),
                    
                    dcc.Input(
                    id = "runtime-weight",
                    type = 'number',
                    placeholder="Runtime Weight...",
                    value = 1
                    ),
                    dcc.Input(
                    id = "year-weight",
                    type = 'number',
                    placeholder="Year Weight...",
                    value = 1
                    ),
                    dcc.Input(
                    id = "rating-weight",
                    type = 'number',
                    placeholder="Rating Weight...",
                    value = 1
                    ),
                    dcc.Input(
                    id = "genre-weight",
                    type = 'number',
                    placeholder="Genre Weight...",
                    value = 1
                    )
                ]),
                html.Div([
                    html.Div(children=[
                    dcc.Graph(id='similarity_heatmap', style={'display': 'inline-block'}),
                    html.Button("Find Most and Least Similar Movies", id = "find-similar"),
                    html.Div(id = "most-similar", children=[
                        html.P("Most Similar:"),
                        html.P("Least Similar:")
                        ])
                    ])
                ],style={'flex': '3', 'padding': '20px'})
                
            ])


])]


#--------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------Button Logic Start-------------------------------------------------------------------#



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
    Input('select-all-genres', 'n_clicks'),
    Input('clear-all-genres', 'n_clicks'),
    prevent_initial_call=True
)
def update_genre_selection(reset_clicks, select_all_clicks, clear_all_clicks):
    triggered_id = ctx.triggered_id

    if triggered_id == 'reset-button':
        return top_10_genres
    elif triggered_id == 'select-all-genres':
        return all_genres
    elif triggered_id == 'clear-all-genres':
        return []
    return dash.no_update

# Callback to update genre graph
@app.callback(
    Output('genre-rating-graph', 'figure'),
    Output('genre-histogram', 'figure'),
    Input('genre-selector', 'value')
)
def update_graph(selected_genres):
    if not selected_genres:
        empty_df = pd.DataFrame({'genres': [], 'average_rating': []})
        empty_hist_df = pd.DataFrame({'rating': [], 'genres': []})

        empty_fig1 = px.bar(empty_df, x='genres', y='average_rating', title="No genres selected")
        empty_fig2 = px.histogram(empty_hist_df, x='rating', color='genres', barmode='stack', title="No genres selected")

        return empty_fig1, empty_fig2

    filtered_df = df_genre[df_genre['genres'].isin(selected_genres)]
    genre_avg_rating = filtered_df.groupby('genres')['rating'].mean().reset_index()
    genre_avg_rating = genre_avg_rating.rename(columns={'rating': 'average_rating'})
    genre_avg_rating = genre_avg_rating.sort_values(by='average_rating', ascending=False)

    # Rating distribution of selected genres
    genre_rating_histogram = px.histogram(
        filtered_df, 
        x="rating", 
        color='genres', 
        barmode='stack', 
        title='Rating Distribution'
    )
    genre_rating_histogram.update_layout(title_x=0.5)
    
    fig = px.bar(
        genre_avg_rating,
        x='genres',
        y='average_rating',
        text='average_rating',
        title='Average Rating per Selected Genre',
        labels={'genres': 'Genre', 'average_rating': 'Average Rating'}
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')
    fig.update_layout(title_x=0.5)

    return fig, genre_rating_histogram

# For t-SNE
@app.callback(
    Output('tsne-plot', 'figure'),
    Input('tsne-data', 'data'),
    Input('genre-filter', 'value'),
    Input('rating-filter', 'value'),
    Input('year-filter', 'value'),
    Input('runtime-filter', 'value'),
    Input('budget-filter', 'value'),
    Input('language-filter', 'value'),
    Input('highlighted-movie', 'data')  # New input
)
def update_tsne_plot(data, selected_genres, rating_range, year_range, runtime_range, budget_range, selected_languages, highlighted_title):
    if not data or 'x' not in data:
        return px.scatter(title="Run t-SNE to generate plot.")

    df_valid = pd.DataFrame(data)

    if isinstance(df_valid['genres'].iloc[0], str):
        df_valid['genres'] = df_valid['genres'].apply(ast.literal_eval)

    # Filter data
    df_filtered = df_valid[
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
        df_filtered = df_filtered[df_filtered['genres'].apply(lambda g: all(genre in g for genre in selected_genres))]
    if selected_languages:
        df_filtered = df_filtered[df_filtered['original_language'].isin(selected_languages)]

    # Base scatter plot    
    fig = px.scatter(
        df_filtered,
        x='x',
        y='y',
        hover_name='title',
        text='title',
        hover_data={
            'x': False, 'y': False,
            'genres': True,
            'year': True,
            'average_rating': True,
            'runtime': True,
            'budget': True
        },
        color='average_rating',
        title='Movie Map (Filtered)',
    )
    fig.update_traces(mode='markers', marker=dict(size=6))

    # Highlight the selected movie
    if highlighted_title and highlighted_title in df_filtered['title'].values:
        row = df_filtered[df_filtered['title'] == highlighted_title].iloc[0]

        # Add main scatter layer (fade if highlighted)
        fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))

        # Add overlay marker for highlighted movie
        fig.add_scatter(
            x=[row['x']],
            y=[row['y']],
            mode='markers+text',
            name='Selected Movie',
            text=[row['title']],
            textposition='top center',
            marker=dict(
                color='black',
                size=10,
                symbol='circle-open-dot',
                line=dict(color='red', width=2)
            ),
            hoverinfo='skip',
            showlegend=True
        )

    fig.update_layout(dragmode='zoom')
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
        return Dash.no_update, "No features selected. Please select at least one."

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

#Find movie written in field on the graph
@app.callback(
    Output('movie-coordinates', 'children'),
    Output('highlighted-movie', 'data'),
    Input('find-in-graph', 'n_clicks'),
    State('tsne-data', 'data'),
    State('movie-field', 'value'),
    prevent_initial_call=True
)
def find_movie(n_clicks, data, movie_name):
    df_find = pd.DataFrame(data)

    if movie_name not in df_find['title'].values:
        return "Movie not found in t-SNE data.", None

    row = df_find[df_find['title'] == movie_name].iloc[0]
    x, y = row['x'], row['y']

    return [f"Movie found and highlighted at" + str(x) +","+ str(y), movie_name]

@app.callback(
    Output('checkbox-container', 'children'),
    Input('tsne-plot','selectedData'),
    State('tsne-data','data'),
    prevent_initial_call=True
)
def generate_checkboxes(selectedData,tsne_data):
    df = pd.DataFrame(tsne_data)

    point_indices = [point['text'] for point in selectedData['points']]

    rows = df.loc[df['title'].isin(point_indices)]
    movies = rows['title'].to_list()

    checkbox_items = movies
    
    return [
        dcc.Checklist(
            options=[{'label': item, 'value': item}],
            value=[item],
            id={'type': 'dynamic-checkbox', 'index': i}
        ) for i, item in enumerate(checkbox_items)
    ]

@app.callback(
    Output('similarity_heatmap','figure'),
    Input({'type': 'dynamic-checkbox', 'index': ALL}, 'value'),
    State('tsne-plot','selectedData'),
    State('tsne-data','data'),
    State('budget-weight','value'),
    State('runtime-weight','value'),
    State('year-weight','value'),
    State('rating-weight','value'),
    State('genre-weight','value'),
    prevent_initial_call=True
)
def read_checkbox_values(values,selectedData,tsne_data,budget_weight,runtime_weight,year_weight,rating_weight,genre_weight):
    # values is a list of selected values per checkbox
    values = [item[0] for item in values if item]
    number_weights = [budget_weight,runtime_weight,year_weight,rating_weight]

    for i in number_weights:
        if i is None:
            i = 1
    
    if genre_weight is None:
        genre_weight = 1

    df = pd.DataFrame(tsne_data)

    point_indices = [point['text'] for point in selectedData['points']]

    rows = df.loc[df['title'].isin(point_indices)]
    rows = rows[rows['title'].isin(values)] #filter based on checkbox


    #calculate jaccard similarity of genres
    genre_similarity = []
    for i in range(0,rows.shape[0]):
        similarity_current_row = []
        for j in range(0,rows.shape[0]):
            similarity_current_row.append(jaccard_similarity(rows['genres'].iloc[i], rows['genres'].iloc[j]))
        genre_similarity.append(similarity_current_row)

    rows = rows.drop(['title','genres','x','y','original_language'], axis = 1)

    #normalize data with original min max values
    rows['year'] = (rows['year'] - min_year) / (max_year-min_year)
    rows['budget'] = (rows['budget'] - min_budget) / (max_budget-min_budget)
    rows['runtime'] = (rows['runtime'] - min_runtime) / (max_runtime-min_runtime)
    rows['average_rating'] = (rows['average_rating'] - min_rating) / (max_rating-min_rating)



    numerical_similarity = []
    for i in range(0,rows.shape[0]):
        
        numerical_similarity_current_row = []
        for j in range(0,rows.shape[0]):
            first_row = (rows.iloc[i] * number_weights)
            second_row = (rows.iloc[j] * number_weights)
            similarity = cosine_similarity([first_row.values],[second_row.values])[0][0]
            numerical_similarity_current_row.append(similarity)
        numerical_similarity.append(numerical_similarity_current_row)

    similarity_matrix = np.add(np.multiply(numerical_similarity,4.0),np.multiply(genre_similarity,genre_weight))
    similarity_matrix = np.divide(similarity_matrix,4+genre_weight)

    similarity_heatmap = px.imshow(similarity_matrix,x=values,y=values,zmin=0,zmax=1,title="Similarity Heatmap")

    return similarity_heatmap

def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0
    return len(intersection) / len(union)

@app.callback(
    Output('most-similar', 'children'),
    Input('find-similar','n_clicks'),
    State('tsne-plot','selectedData'),
    State('tsne-data','data'),
    prevent_initial_call=True
)
def find_similar_movies(n_clicks, selectedData,tsne_data):
    df = pd.DataFrame(tsne_data)

    html_output = []

    point_indices = [point['text'] for point in selectedData['points']]

    rows = df.loc[df['title'].isin(point_indices)]

    x0 = rows['x'].mean()
    y0 = rows['y'].mean()

    print(rows['title'].values) 

    df = df[~df['title'].isin(rows['title'].values)]

    # Compute Euclidean distance
    df['distance'] = np.sqrt((df['x'] - x0)**2 + (df['y'] - y0)**2)

    print(df['distance'])

    # 3 closest points
    closest_3 = df.nsmallest(3, 'distance')

    # 3 farthest points
    farthest_3 = df.nlargest(3, 'distance')

    print(closest_3)

    print(closest_3['title'].values)

    html_output.append(html.H3('Most Similar Movies:'))
    for row in closest_3['title'].values:
        html_output.append(html.P(row))

    html_output.append(html.H3('Least Similar Movies:'))
    for row in farthest_3['title'].values:
        html_output.append(html.P(row))

    print(html_output)

    return html_output


#%%
if __name__ == '__main__':
    app.run(debug=True)