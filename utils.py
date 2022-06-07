import numpy as np
import pandas as pd
import math
import copy
import pickle
import gensim
import scipy.sparse as ss
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


def load_key_to_index():
    key_to_index = pd.read_csv("dw2v/wordIDHash_ufr.csv")
    key_to_index = key_to_index.set_index("word")
    return key_to_index["index"].to_dict()

### FOR HAMILTON ET AL DYNAMIC EMBEDDINGS

def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """
    Original script: https://gist.github.com/zhicongchen/9e23d5c3f1e5b1293b16133485cd17d8 
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.

    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    #Refilling the normed vectors
    in_base_embed.wv.fill_norms(force = True)
    in_other_embed.wv.fill_norms(force = True)
    
    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.wv.get_normed_vectors()
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
    
    return other_embed

def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        # print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1,m2)

#utility functions for running the CD method
#loss: min 1/2 /sum_t | Yt - UtVt' |^2 + lam/2 /sum_t(|Ut|^2 + |Vt|^2) + 
#                                        tau/2 /sum_t>1(|Vt - Vt-1|^2 + |Ut - Ut-1|^2)
#                                        gam/2 /sum_t (|Ut - Vt|^2)


def update(U,Y,Vm1,Vp1,lam,tau,gam,ind,iflag):
    """Implementation is copied from https://github.com/yifan0sun/DynamicWord2Vec.
    Utility functions for running the CD method. 
    loss: min 1/2 /sum_t | Yt - UtVt' |^2 + lam/2 /sum_t(|Ut|^2 + |Vt|^2) + 
                                           tau/2 /sum_t>1(|Vt - Vt-1|^2 + |Ut - Ut-1|^2)
                                           gam/2 /sum_t (|Ut - Vt|^2)
    for the function, the equations are to update V. So:
    Y is n X b (b = batch size)
    r = rank
    U is n X r
    Vm1 and Vp1 are bXr. so they are b rows of V, transposed
    """

    UtU = np.dot(U.T,U) # rxr
    r = UtU.shape[0]    
    if iflag:   M   = UtU + (lam + 2*tau + gam)*np.eye(r) #Gradient of equation 8 in paper.
    else:       M   = UtU + (lam + tau + gam)*np.eye(r) #Gradient of equation 8 in paper.
       
    Uty = np.dot(U.T,Y) # rxb
    Ub  = U[ind,:].T   # rxb
    A   = Uty + gam*Ub + tau*(Vm1.T+Vp1.T)  # rxb
    Vhat = np.linalg.lstsq(M,A) #rxb
    return Vhat[0].T #bxr

def import_static_init(T,EMBEDDING_DIMENSIONS):
    """Implementation is copied from https://github.com/yifan0sun/DynamicWord2Vec"""
    with open(f'dw2v/static_embeddings/emb_static_ufr_EMB_{EMBEDDING_DIMENSIONS}.pkl','rb') as f:
        emb = pickle.load(f)
    if type(emb) == gensim.models.word2vec.Word2Vec:
        emb = emb.wv.vectors
    U = [copy.deepcopy(emb) for t in T]
    V = [copy.deepcopy(emb) for t in T]
    return U,V

def initvars(vocab_size,T,rank):
    # dictionary will store the variables U and V. tuple (t,i) indexes time t and word index i
    U,V = [],[]
    U.append(np.random.randn(vocab_size,rank)/np.sqrt(rank))
    V.append(np.random.randn(vocab_size,rank)/np.sqrt(rank))
    for t in range(1,T):
        U.append(U[0].copy())
        V.append(V[0].copy())
        print(t)
    return U,V

def getmat(f,v,rowflag):
    data = pd.read_csv(f)
    data["word"] = data["word"].astype(int)
    data["context"] = data["context"].astype(int)
    data = data.drop(columns="Unnamed: 0")
    data = data.values
    
    X = ss.coo_matrix((data[:,2],(data[:,0],data[:,1])),shape=(v,v))
   
    if rowflag: 
        X = ss.csr_matrix(X)
        #X = X[inds,:]
    else:
        X = ss.csc_matrix(X)
        #X = X[:,inds]
    
    return X#.todense()

def getbatches(vocab,b):
    batchinds = []
    current = 0
    while current<vocab:
        inds = range(current,min(current+b,vocab))
        current = min(current+b,vocab)
        batchinds.append(inds)
    return batchinds

#   THE FOLLOWING FUNCTION TAKES A WORD ID AND RETURNS CLOSEST WORDS BY COSINE DISTANCE
def getclosest(wid,U):
    C = []
    for t in range(len(U)):
        temp = U[t]
        K = cosine_similarity(temp[wid,:],temp)
        mxinds = np.argsort(-K)
        mxinds = mxinds[0:10]
        C.append(mxinds)
    return C
        
# THE FOLLOWING FUNCTIONS COMPUTES THE REGULARIZER SCORES GIVEN U AND V ENTRIES
def compute_symscore(U,V):
    return np.linalg.norm(U-V)**2

def compute_smoothscore(U,Um1,Up1):
    X = np.linalg.norm(U-Up1)**2 + np.linalg.norm(U-Um1)**2
    return X

### PLOT EMBEDDINGS

def plot_words_plotly(word1, words, fitted,sims,fig,fig_placement=[1,1]):
    fig = fig.add_trace(go.Scatter(x=[],y=[],
                     mode='text'),
                     row=fig_placement[0],
                     col=fig_placement[1])
    fig.update_traces(marker=dict(color="rgba(255,255,255)",opacity=0))
    
    annotations = []
    isArray = type(word1) == list
    print(f"plotting at: (x{fig_placement[1]}, y{fig_placement[0]})")
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

def create_subplot(query_word, embeddings_time, fig, fig_placement, **kwargs):
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