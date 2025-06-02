from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx
import pandas as pd
import plotly.express as px

#%% Import data
df_tags = pd.read_csv("data/ml-32m/tags.csv").head(1000)
df_movies = pd.read_csv("data/ml-32m/movies.csv").head(1000)
df_ratings = pd.read_csv("data/ml-32m/ratings.csv").head(1000)
df_links = pd.read_csv("data/ml-32m/links.csv").head(1000)

#%% Processing
df_genre = pd.merge(df_ratings, df_movies, on="movieId")
df_genre['genres'] = df_genre['genres'].str.split('|')
df_genre = df_genre.explode('genres')

all_genres = sorted(df_genre['genres'].unique())
top_10_genres = df_genre['genres'].value_counts().head(10).index.tolist()

#%% App
app = Dash()

app.layout = html.Div([
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

    html.Hr(),
    html.H3("Tags Data Table"),
    dash_table.DataTable(data=df_tags.to_dict('records'), page_size=10),
])


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

# Run
if __name__ == '__main__':
    app.run(debug=True)
