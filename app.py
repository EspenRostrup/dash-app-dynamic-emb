import math
import numpy as np
from sklearn.manifold import TSNE

from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go

from gensim.models import KeyedVectors
from plotly.subplots import make_subplots

app = Dash(__name__,
        suppress_callback_exceptions=True)
server = app.server

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

def load_data(emb_type:str):
    if emb_type == "dw2v":
        timespans = list(zip(range(1867,2027,5)[:-1],range(1867,2027,5)[1:]))
        timespans = [f"{x}_{y}" for x,y in timespans]
    elif emb_type == "aw2v":
        INTERVAL = [1867, 1920, 1950, 1980, 1995, 2010, 2022]
        timespans = list(zip(INTERVAL[:-1],INTERVAL[1:]))
        timespans = [f"{x}_{y}" for x,y in timespans]
    kvs = [KeyedVectors.load(f"kv/{emb_type}/{emb_type}_emb_{x}.kv") for x in timespans]
    return kvs

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig


# dw2v_kvs = load_data("dw2v")
aw2v_kvs = load_data("aw2v")

app.layout = html.Div([
    html.H1(children='Dynamic word embeddings based on court rulings from Ugeskrift for Retvæsen'),
    html.Hr(),
    dcc.Markdown(
        """
        This application shows the relations between a queried word's embedding and a subset of the \"most related words\" through time. 
        
        Variations of this visualisation technique is used in several dynamic word embedding papers (see e.g. [Kurlkarni et al, 2015](https://dl.acm.org/doi/10.1145/2736277.2741627); [Hamilton et al, 2016](https://arxiv.org/abs/1605.09096); [Yao et al, 2018](https://arxiv.org/abs/1703.00607)). 
        
        - For each period, I sample the _N_ closest words to a word, where _N_ can be varied. 
        - The unique similar words' most recent embeddings are then collected in a list with the selected word's embeddings for each period. 
        - The embeddings are assigned reduced to a  two-dimensional space using the dimensional reduction technique made for visualising high dimensional data, t-SNE.
        
        Intuitively each similar word is depicted using the most recent embedding since that is the current understanding of the said word. 
        In reality, all word embeddings change over time, which would clutter the analysis if depicted, but should be kept in mind when analysing the results. 
        
        """
        ),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Dropdown(
            options=[],
            value='fod',
            multi=False,
            id='input',
            style = {"display":'none'}
        ),
    dcc.Graph(id='graph-with-qword',figure = blank_fig()),
    html.Hr(),
    html.Div([html.H4('Top N related words'),
    dcc.Slider(
        0,
        20,
        1,
        value=15,
        id='topn-slider'
    )]
    ),
    html.Div([html.H4('Type of embeddings'),
                dcc.RadioItems(
                ['AW2V'],
                'AW2V',
                id='emb-type',
                labelStyle={'display': 'block'}),
    html.Hr(style={"height":"3px"})
    ])
],style=
    {"margin":"30px 30px 30px 30px "})

@app.callback(Output('page-content', 'children'),[Input('url', 'pathname')],Input(component_id='emb-type',component_property='value'))
def generate_layout(url,emb_type):
    if emb_type == "AW2V":
            options = list(aw2v_kvs[0].key_to_index.keys())
    # elif emb_type == "DW2V":
    #     options = list(dw2v_kvs[0].key_to_index.keys())
    return html.Div([
        html.H4('Query Word'),
        dcc.Dropdown(
            options=options,
            value='fod',
            multi=False,
            id='input',
            style={"width":"500px"}
        ),
        html.Div(id='output')
    ])

def plot_words_plotly(word1, words, fitted,sims,fig,fig_placement=[1,1]):
    fig = fig.add_trace(go.Scatter(x=[],y=[],
                     mode='text'),
                     row=fig_placement[0],
                     col=fig_placement[1])
    fig.update_traces(marker=dict(color="rgba(255,255,255)",opacity=0))
    
    annotations = []
    isArray = type(word1) == list
    for i in range(len(words)):
        pt = fitted[i]

        ww,decade = [w.strip() for w in words[i].split("|")]
        decade = "-".join(decade.split("_"))
        
        color = "green"

        word = ww
        # word1 is the word we are plotting against
        if ww == word1 or (isArray and ww in word1):
            if len(decade) == 4:
                decade = int(decade)
                decade += 2 
            annotations.append((ww, decade, pt))
            # annotations.append(pt)
            word = decade
            color = 'black'
            sizing = 15
        else:
            sizing = 13
        fig.add_annotation(text = word,
                            row= fig_placement[0],
                            col= fig_placement[1],
                            x = pt[0], 
                            y = pt[1], 
                            showarrow=False,
                            font=dict(color=color,size=sizing)
                            )
    return fig, annotations

def plot_annotations_plotly(annotations,fig, fig_placement):
    # draw the movement between the word through the decades as a series of
    # annotations on the graph
    annotations.sort(key=lambda w: w[1], reverse=True)
    annotations = [x[-1] for x in annotations]
    def scale(x): 
        dist=math.dist(x[0],x[1])
        # if dist > 1: k = 1/dist 
        # else: k=1
        k=1
        return (x[1]-x[0])*(1-k)+x[0]
    prev = np.stack(annotations)[0]
    for x in np.stack(annotations)[1:]:
        coordinate_scaled_from = scale(np.asarray([prev,x]))
        coordinate_scaled_to = scale(np.asarray([x,prev]))
        coordinates = np.stack([coordinate_scaled_from,coordinate_scaled_to])
        fig.add_scatter(x=coordinates[:,0],
                        y=coordinates[:,1],
                        mode="lines",
                        line=dict(width=0.5,color="green"),
                        row= fig_placement[0],
                        col= fig_placement[1])
        prev=x

    return fig

def query_TSNE_plot_emb(query_word,                #Word that is queried
                        embeddings_time,           #List with KeyedVectors for each timeperiod 
                        top_n=3,                   #The top X most similar words for each time period
                        timespan=[],               #Custom list time period names mapping embeddings_time to a time period
                        aggregation_interval=5,    #If no custom list, aggregation interval is the aggregated years 
                                                   # following year start that is included in the time period
                        year_start=1867,           #Year that the timeperiod start
                        interval_sampled=1         #Only include every x time of the embedding
                        ):  
    if type(interval_sampled)==dict: interval_sampled=interval_sampled["interval_sampled"]       
    query_embeddings = {}
    list_of_unique_words = []
    query_similar_word = {}
    query_similar_word_score = {}
    for index, model in enumerate(embeddings_time[::interval_sampled]):
        if timespan ==[]:
            current_time_period = index*interval_sampled*aggregation_interval+year_start
        else: 
            current_time_period = timespan[index]
        try:
            query_word_embedding = model.get_vector(query_word,norm=True)
        except KeyError:
            raise PreventUpdate
             
        query_embeddings.update({f"{query_word}|{current_time_period}":query_word_embedding})
        most_sim_words = model.most_similar(query_word,topn=top_n)
        for sim_word in most_sim_words:
            if sim_word[0] in list_of_unique_words:
                continue
            if sim_word[1]<0.4:
                continue
            sim_word_embedding = embeddings_time[-1].get_vector(sim_word[0],norm=True)
            query_similar_word.update({f"{sim_word[0]}|{index}" : sim_word_embedding})
            query_similar_word_score.update({f"{sim_word[0]}|{index}" : sim_word[1]})
            list_of_unique_words.append(sim_word[0])
    embeddings_to_plot = query_embeddings
    embeddings_to_plot.update(query_similar_word) 
    return embeddings_to_plot, query_similar_word_score





@app.callback(
    Output(component_id='graph-with-qword', component_property='figure'),
    Input('input', 'value'),
    Input(component_id='topn-slider',component_property='value'),
    Input(component_id='emb-type',component_property='value')
)
def update_graph(query_word, top_n, embeddings_time, fig_placement = [1,1], **kwargs):
    if query_word is None: raise PreventUpdate
    if embeddings_time == "DW2V":
        embeddings_time =  dw2v_kvs 
        timespan=[]
    else:
        embeddings_time = aw2v_kvs
        INTERVAL = [1867, 1920, 1950, 1980, 1995, 2010, 2022]
        timespan = list(zip(INTERVAL[:-1],INTERVAL[1:]))
        timespan = [f"{x}_{y}" for x,y in timespan]
    
    fig = make_subplots(1,1)
    fig['layout'].update(margin=dict(l=30,r=30,b=20,t=20))
    embeddings_to_plot, similar_word_score = query_TSNE_plot_emb(
                                                query_word,
                                                embeddings_time,
                                                top_n,
                                                timespan = timespan,
                                                **kwargs)
    #FIT TSNE
    mat = np.array([embeddings_to_plot[word] for word in embeddings_to_plot])
    model = TSNE(n_components=2, random_state=0, learning_rate=200, init='pca')
    fitted = model.fit_transform(mat)
    #PLOT
    fig, annotations = plot_words_plotly(query_word,
                                        list(embeddings_to_plot.keys()),
                                        fitted,
                                        similar_word_score,
                                        fig,
                                        fig_placement)
    fig = plot_annotations_plotly(annotations,fig,fig_placement)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
