# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas
import math
import numpy as np
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
from plotly.subplots import make_subplots
import pandas as pd

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

def load_data():
    timespans = list(zip(range(1867,2027,5)[:-1],range(1867,2027,5)[1:]))
    timespans = [f"{x}_{y}" for x,y in timespans]
    dw2v_kws = [KeyedVectors.load(f"kw/dw2v_emb_{x}.kv") for x in timespans]
    return dw2v_kws

dw2v_kws = load_data()

app.layout = html.Div([
    dcc.Graph(id='graph-with-qword'),
    dcc.Input(
        id='text-box', value="mand",type="text"
    )
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
            sizing = 11
        else:
            sizing = sims[words[i]] * 17
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
        query_word_embedding = model.get_vector(query_word,norm=True)
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
    Input(component_id='text-box', component_property='value')
)
def update_graph(query_word, embeddings_time = dw2v_kws ,fig_placement = [1,1], **kwargs):
    fig = make_subplots(1,1)
    embeddings_to_plot, similar_word_score = query_TSNE_plot_emb(
                                                query_word,
                                                embeddings_time,
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
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
