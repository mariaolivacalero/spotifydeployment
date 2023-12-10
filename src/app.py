import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import base64
from collections import namedtuple
import pandas as pd, numpy as np
import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import (
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    silhouette_score,
    v_measure_score,
    adjusted_rand_score
)

colorscale=[[False, '#eeeeee'], [True, '#76cf63']]

# Datos de ejemplo para las gráficas y la tabla
df = pd.read_csv('data/history.csv', index_col = 0)
filtered_data = pd.DataFrame()
#tracks playing time
df['secPlayed'] = df['msPlayed'] / 1000
df = df[df.columns[:-1].insert(4, df.columns[-1])] #moving seconds column to proper place
df = df[df.secPlayed > 60] #removing songs that were played for less than 60 secs 
                            # me quito 60000 reproducciones
df.shape
# converting ms to minute and extracting date from datetime column
df['mins_played'] = df.apply(lambda x: round(x['msPlayed']/60000,2), axis=1)
df['date'] = df.apply(lambda x: pd.to_datetime(x['datetime'][:10],format='%Y-%m-%d'),axis=1)

# calculate the daily streaming time length 
daily_length = df.groupby('date',as_index=True).sum()

# create new date series for displaying time series data
idx = pd.DataFrame(pd.date_range(min(df.date), max(df.date)),columns=['date'])
idx['date'] = idx.apply(lambda x: pd.to_datetime(x['date'],format='%Y-%m-%d'),axis=1)

# use new date series to display the daily streaming time
new_daily_length = pd.merge(idx, daily_length, how='left', left_on='date', right_on = 'date', copy=False)

# getting rid of columns except for date and time
new_daily_length = new_daily_length.drop(new_daily_length.loc[:, 'msPlayed':'time_signature'], axis=1)

artist = df.drop(columns=["date"])
artist_length = artist.groupby('artistName',as_index=False).sum()
artist_song_cnt = artist.groupby('artistName',as_index=False).agg({"trackName": "nunique"})

# merge artist_length and artist_song_cnt 
artist_length_uniqsong = pd.merge(artist_length,artist_song_cnt,how='left',on='artistName',copy=False)
artist_length_uniqsong.rename(columns={'trackName':'unique_track_number'},inplace=True)

# sorting the df by minutes played
max_time  = artist_length_uniqsong.sort_values(by=['mins_played'])

# top 30 artist I listen to (tail because the df is in ascending order of count of minutes)
most_heard_30 = max_time.tail(30)

def year_analysis(df):
    #identifying the month
    df['month'] = df.endTime.str.split('-').apply(lambda x: (x[0], x[1]))

    #listing months and features
    months = list(set(df.month.values))
    months.sort()
    features = ['danceability', 'energy', 'speechiness', 'instrumentalness', 'valence',"loudness","tempo"]

    #standardizing features (we're interested not in their absolute value, but in how each changed over time)
    for feature in features:
        df[f'{feature}_zscore'] = ( df[feature] - df[feature].mean() ) / df[feature].std()

    #making sure we standardized correctly: mean is 0 and std is 1
    df[[feature + '_zscore' for feature in features]].describe().loc['mean':'std'].T

    #features averages by month
    Month = namedtuple('Month', features)
    avg_features_months = []
    for month in months:
        df_month = df[df['month'] == month]
        avg_features = df_month.describe().loc['mean'][[feature + '_zscore' for feature in features]]
        month = Month(*avg_features)
        avg_features_months.append(month)

    #labelling months
    month_labels = [f'{month[1]}/{month[0]}' for month in months]
    month_labels_short = [m[:3]+m[-2:] for m in month_labels]

    features = ['valence', 'energy', 'danceability', 'speechiness', 'instrumentalness',"loudness","tempo"]
    x = [-1] + [x for x in range(13)]

    # Create traces for each feature
    traces = []
    for feature in features:
        y = [getattr(month, feature) for month in avg_features_months]
        traces.append(go.Scatter(x=x, y=y, mode='lines+markers', name=feature))

    # Create the layout
    layout = go.Layout(
        title='My mood this past year (According to Spotify)',
        xaxis=dict(
            title='Months',
            tickmode='array',
            tickvals=list(range(13)),
            ticktext=month_labels_short,
        ),
        yaxis=dict(
            title='Mood',
        ),
        showlegend=True,
    )

    # Add annotations
    annotations = [
        dict(x=1, y=0.4, xref="x", yref="y", text="Studying", showarrow=True, arrowhead=5, ax=0, ay=-40, font=dict(size=14)),
        dict(x=4.15, y=0.4, xref="x", yref="y", text="Travelling", showarrow=True, arrowhead=5, ax=0, ay=-40, font=dict(size=14)),
        dict(x=6.75, y=0.4, xref="x", yref="y", text="Graduating", showarrow=True, arrowhead=5, ax=0, ay=-40, font=dict(size=14)),
        dict(x=10.35, y=0.4, xref="x", yref="y", text="Back to school", showarrow=True, arrowhead=5, ax=0, ay=-40, font=dict(size=14)),
    ]

    layout['annotations'] = annotations

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    return fig

track_characteristics = df.groupby('trackName').agg({
    'artistName': 'first',
    'danceability': 'mean',
    'energy': 'mean',
    'key': 'mean',
    'loudness': 'mean',
    'mode': 'mean',
    'speechiness': 'mean',
    'acousticness': 'mean',
    'instrumentalness': 'mean',
    'liveness': 'mean',
    'valence': 'mean',
    'tempo': 'mean',
    'msPlayed': 'sum'
}).reset_index()

track_characteristics = track_characteristics.sort_values(by='msPlayed', ascending=False)

# Scatter plots using Plotly Express
scatter_loudness_energy = px.scatter(
    track_characteristics,
    x="loudness",
    y="energy",
    color="mode",
    color_continuous_scale="magma",
    title="Loudness vs. Energy",
)

scatter_valence_danceability = px.scatter(
    track_characteristics,
    x="valence",
    y="danceability",
    color="mode",
    color_continuous_scale="picnic",
    title="Valence vs. Danceability",
)

scatter_valence_energy = px.scatter(
    track_characteristics,
    x="valence",
    y="energy",
    color="mode",
    color_continuous_scale="speed",
    title="Valence vs. Energy",
)

scatter_loudness_valence = px.scatter(
    track_characteristics,
    x="loudness",
    y="valence",
    color="mode",
    color_continuous_scale="peach",
    title="Loudness vs. Valence",
)

scatter_loudness_liveness = px.scatter(
    track_characteristics,
    x="loudness",
    y="liveness",
    color="mode",
    color_continuous_scale="tempo",
    title="Loudness vs. Liveness",
)

scatter_danceability_tempo = px.scatter(
    track_characteristics,
    x="danceability",
    y="tempo",
    color="mode",
    color_continuous_scale="turbo",
    title="Danceability vs. Tempo",
)


## grafica historico diario
def display_year(
    z,
    year: int = None,
    month_lines: bool = True,
    fig=None,
    row: int = None
):
    
    if year is None:
        year = datetime.datetime.now().year
        
    d1 = datetime.date(year, 1, 1)
    d2 = datetime.date(year, 12, 31)

    number_of_days = (d2-d1).days + 1
    
    data = np.ones(number_of_days) * np.nan
    data[:len(z)] = z
    

    d1 = datetime.date(year, 1, 1)
    d2 = datetime.date(year, 12, 31)

    delta = d2 - d1
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_days =   [31,    28,    31,     30,    31,     30,    31,    31,    30,    31,    30,    31]
    if number_of_days == 366:  # leap year
        month_days[1] = 29
    month_positions = (np.cumsum(month_days) - 15)/7

    dates_in_year = [d1 + datetime.timedelta(i) for i in range(delta.days+1)] # list with datetimes for each day a year
    weekdays_in_year = [i.weekday() for i in dates_in_year] # gives [0,1,2,3,4,5,6,0,1,2,3,4,5,6,…] (ticktext in xaxis dict translates this to weekdays
    
    weeknumber_of_dates = []
    for i in dates_in_year:
        inferred_week_no = int(i.strftime("%V"))
        if inferred_week_no >= 52 and i.month == 1:
            weeknumber_of_dates.append(0)
        elif inferred_week_no == 1 and i.month == 12:
            weeknumber_of_dates.append(53)
        else:
            weeknumber_of_dates.append(inferred_week_no)
    
    text = [str(i) for i in dates_in_year] #gives something like list of strings like ‘2018-01-25’ for each date. Used in data trace to make good hovertext.
    #4cc417 green #347c17 dark green
    colorscale=[[False, '#eeeeee'], [True, '#76cf63']]
    
    # handle end of year
    

    data = [
        go.Heatmap(
            x=weeknumber_of_dates,
            y=weekdays_in_year,
            z=data,
            text=text,
            hoverinfo='text',
            xgap=3, # this
            ygap=3, # and this is used to make the grid-like apperance
            showscale=False,
            colorscale=colorscale
        )
    ]
    
        
    if month_lines:
        kwargs = dict(
            mode='lines',
            line=dict(
                color='#9e9e9e',
                width=1,
            ),
            hoverinfo='skip',
        )
        
        for date, dow, wkn in zip(
            dates_in_year, weekdays_in_year, weeknumber_of_dates
        ):
            if date.day == 1:
                data += [
                    go.Scatter(
                        x=[wkn-.5, wkn-.5],
                        y=[dow-.5, 6.5],
                        **kwargs,
                    )
                ]
                if dow:
                    data += [
                    go.Scatter(
                        x=[wkn-.5, wkn+.5],
                        y=[dow-.5, dow - .5],
                        **kwargs,
                    ),
                    go.Scatter(
                        x=[wkn+.5, wkn+.5],
                        y=[dow-.5, -.5],
                        **kwargs,
                    )
                ]
                    


import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Select the top 300 most listened songs
top_songs = track_characteristics.nlargest(300, 'msPlayed')

# Extract the relevant features for clustering
features = top_songs[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','msPlayed']]

# Scale the features
scaler = preprocessing.StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(
    data=pca_features,
    columns=['PC1', 'PC2'])


# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(pca_features)
    wcss.append(kmeans.inertia_)


# Train a KMeans clustering model with the optimal number of clusters
optimal_k = 5 
kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans_model.fit(pca_features)

# Assign cluster labels to the songs
cluster_labels = kmeans_model.labels_

# Add the cluster labels as a new column to the track_characteristics dataset
top_songs['cluster'] = cluster_labels




def display_year(
    z,
    year: int = None,
    month_lines: bool = True,
    fig=None,
    row: int = None
):
    
    if year is None:
        year = datetime.datetime.now().year
        
    d1 = datetime.date(year, 1, 1)
    d2 = datetime.date(year, 12, 31)

    number_of_days = (d2-d1).days + 1
    
    data = np.ones(number_of_days) * np.nan
    data[:len(z)] = z
    

    d1 = datetime.date(year, 1, 1)
    d2 = datetime.date(year, 12, 31)

    delta = d2 - d1
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_days =   [31,    28,    31,     30,    31,     30,    31,    31,    30,    31,    30,    31]
    if number_of_days == 366:  # leap year
        month_days[1] = 29
    month_positions = (np.cumsum(month_days) - 15)/7

    dates_in_year = [d1 + datetime.timedelta(i) for i in range(delta.days+1)] # list with datetimes for each day a year
    weekdays_in_year = [i.weekday() for i in dates_in_year] # gives [0,1,2,3,4,5,6,0,1,2,3,4,5,6,…] (ticktext in xaxis dict translates this to weekdays
    
    weeknumber_of_dates = []
    for i in dates_in_year:
        inferred_week_no = int(i.strftime("%V"))
        if inferred_week_no >= 52 and i.month == 1:
            weeknumber_of_dates.append(0)
        elif inferred_week_no == 1 and i.month == 12:
            weeknumber_of_dates.append(53)
        else:
            weeknumber_of_dates.append(inferred_week_no)
    
    text = [str(i) for i in dates_in_year] #gives something like list of strings like ‘2018-01-25’ for each date. Used in data trace to make good hovertext.
    #4cc417 green #347c17 dark green
    colorscale=[[False, '#eeeeee'], [True, '#76cf63']]
    
    # handle end of year
    

    data = [
        go.Heatmap(
            x=weeknumber_of_dates,
            y=weekdays_in_year,
            z=data,
            text=text,
            hoverinfo='text',
            xgap=3, # this
            ygap=3, # and this is used to make the grid-like apperance
            showscale=False,
            colorscale=colorscale
        )
    ]
    
        
    if month_lines:
        kwargs = dict(
            mode='lines',
            line=dict(
                color='#9e9e9e',
                width=1,
            ),
            hoverinfo='skip',
        )
        
        for date, dow, wkn in zip(
            dates_in_year, weekdays_in_year, weeknumber_of_dates
        ):
            if date.day == 1:
                data += [
                    go.Scatter(
                        x=[wkn-.5, wkn-.5],
                        y=[dow-.5, 6.5],
                        **kwargs,
                    )
                ]
                if dow:
                    data += [
                    go.Scatter(
                        x=[wkn-.5, wkn+.5],
                        y=[dow-.5, dow - .5],
                        **kwargs,
                    ),
                    go.Scatter(
                        x=[wkn+.5, wkn+.5],
                        y=[dow-.5, -.5],
                        **kwargs,
                    )
                ]
                    
                    
    layout = go.Layout(
        title='My Spotify Activity',
        height=250,
        yaxis=dict(
            showline=False, showgrid=False, zeroline=False,
            tickmode='array',
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            autorange="reversed",
        ),
        xaxis=dict(
            showline=False, showgrid=False, zeroline=False,
            tickmode='array',
            ticktext=month_names,
            tickvals=month_positions,
        ),
        font={'size':10, 'color':'#9e9e9e'},
        plot_bgcolor=('#fff'),
        margin = dict(t=40),
        showlegend=False,
    )

    if fig is None:
        fig = go.Figure(data=data, layout=layout)
    else:
        fig.add_traces(data, rows=[(row+1)]*len(data), cols=[1]*len(data))
        fig.update_layout(layout)
        fig.update_xaxes(layout['xaxis'])
        fig.update_yaxes(layout['yaxis'])

    
    return fig


def display_years(z, years):
    
    day_counter = 0
    
    fig = make_subplots(rows=len(years), cols=1, subplot_titles=years)
    for i, year in enumerate(years):
        d1 = datetime.date(year, 1, 1)
        d2 = datetime.date(year, 12, 31)
        
        number_of_days = (d2-d1).days + 1
        data = z[day_counter : day_counter + number_of_days]
        
        display_year(data, year=year, fig=fig, row=i)
        fig.update_layout(height=250*len(years))
        day_counter += number_of_days
    return fig



df['mins_played'] = df.apply(lambda x: round(x['msPlayed']/60000,2), axis=1)
df['date'] = df.apply(lambda x: pd.to_datetime(x['datetime'][:10],format='%Y-%m-%d'),axis=1)

# calculate the daily streaming time length 
daily_length = df.groupby('date',as_index=True).sum()

# create new date series for displaying time series data
idx = pd.DataFrame(pd.date_range(min(df.date), max(df.date)),columns=['date'])
idx['date'] = idx.apply(lambda x: pd.to_datetime(x['date'],format='%Y-%m-%d'),axis=1)

# use new date series to display the daily streaming time
new_daily_length = pd.merge(idx, daily_length, how='left', left_on='date', right_on = 'date', copy=False)

# getting rid of columns except for date and time
new_daily_length = new_daily_length.drop(new_daily_length.loc[:, 'msPlayed':'time_signature'], axis=1)

# Define the graphyears graph
graphyears = dcc.Graph(
    id='graphyears',
    config={'displayModeBar': False},
    style={'flex': '1', 'padding': '20px', 'float': 'left', 'width': '60%'}
)

year = dcc.Graph(
    id='year',
    config={'displayModeBar': False},
    style={'flex': '1', 'padding': '20px', 'float': 'left', 'width': '70%'}
)
playlists = {"playlist1":0,"playlist2":1,"playlist3":2,"playlist4":3, "playlist5":4}

# Lista de playlists 
playlist_options = [{'label': key, 'value': value} for key, value in playlists.items()]

# Logo de Spotify
logo = 'data/logo.png'  
encoded_logo = base64.b64encode(open(logo, 'rb').read()).decode('ascii')

# Dash APP
app = dash.Dash(__name__)
server = app.server

# Layout de la aplicación
app.layout = html.Div(children=[
    # Sección superior con logo y título
    html.Div([
        html.Img(src=f'data:image/png;base64,{encoded_logo}', style={'height': '100px'}),
        html.H1("My year on Spotify", style={'text-align': 'center', 'font-family': 'SpotifyFont'}),  # Use Spotify-like font
    ], style={'text-align': 'center', 'padding': '20px'}),

    # Primera fila
    html.Div([
        # Years graph en la izquierda
        year,

        # Dropdown y tabla en la derecha
        html.Div([
            dcc.Dropdown(
                id='playlist-dropdown',
                options=playlist_options,
                value=playlist_options[0]['value'],
                multi=False,
                style={'margin-bottom': '20px'}
            ),

            dash_table.DataTable(
                id='tabla',
                columns=[],
                data=[],
                style_table={
                    'height': '400px',
                    'overflowY': 'auto',
                    'backgroundColor': 'rgb(30 215 96)',
                },
                style_header={
                    'backgroundColor': 'rgb(30 215 96)',
                    'color': 'white',
                    'fontWeight': 'bold',
                },
                style_cell={
                    'backgroundColor': 'rgb(152,251,152)',
                    'color': 'black',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(144,238,144)',
                    },
                ],
                virtualization=True,
            ),
        ], style={'flex': '1', 'padding': '20px', 'float': 'right', 'width': '60%'}),
    ], style={'display': 'flex'}),

    # Segunda fila
    html.Div([
        # Graph years en la izquierda
        graphyears,

        # Most heard artists en la derecha
        dcc.Graph(
            figure = px.bar(most_heard_30, 
             x='mins_played', 
             y='artistName', 
             orientation='h',  # Set orientation to horizontal for a bar plot
             title='Top 30 Artists Heard',
             labels={'artistName': 'Artist Name', 'mins_played': 'Minutes Played'},
             color='mins_played',  # Use 'mins_played' as the color variable
             color_continuous_scale='greens')),
    ], style={'display': 'flex'}),

    # Tercera fila
    html.Div([
        dcc.Graph(figure=scatter_loudness_energy),
        dcc.Graph(figure=scatter_valence_danceability),
        dcc.Graph(figure=scatter_valence_energy),
    ], style={'display': 'flex'}),

    html.Div([
        dcc.Graph(figure=scatter_loudness_liveness),
        dcc.Graph(figure=scatter_danceability_tempo),
        dcc.Graph(figure=scatter_loudness_valence),
    ], style={'display': 'flex'})

])

           

# Callback para actualizar la tabla según la playlist seleccionada
# Callback to update the table based on the selected playlist
@app.callback(
    [Output('tabla', 'data'),
    Output('tabla', 'columns')],
    [Input('playlist-dropdown', 'value')]
)
def update_table(selected_playlist):
    print("Selected Playlist:", selected_playlist)
    
    if selected_playlist is None:
        print("No playlist selected.")
        return pd.DataFrame().to_dict('records'), []

    filtered_data = top_songs[top_songs['cluster'] == selected_playlist]
    filtered_data  = filtered_data[['artistName', 'trackName']]
    print("Filtered Data:")
    print(filtered_data)
    
    columns = [{"name": col, "id": col} for col in filtered_data.columns]
    
    return filtered_data.to_dict('records'), columns


z = [0] * 312
for i in new_daily_length["secPlayed"]:
    z.append(i)

fig = display_years(z, (2022, 2023))
app.layout['graphyears'].figure = fig

year = year_analysis(df)
app.layout["year"].figure = year

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)