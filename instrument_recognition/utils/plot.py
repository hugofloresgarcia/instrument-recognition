import os
import itertools
import io
import datetime
import yaml

import numpy as np
import pandas as pd
import torch 
import matplotlib.pyplot as plt
import plotly.express as px
import PIL

#---------------------#
#     PLOT UTILS
#---------------------#

def plot_filterbank(fb, n_rows, n_cols, title='filterbank', figsize=(6.4, 4.8)):
    """
    plot a group of time domain filters
    params:
        fb (np.ndarray or torch.Tensor): filterbank w shape (filters, sample)
        n_rows: rows for plot
        n_cols: cols for plot

    returns:
        matplotlib figure
    """
    n_subplots = fb.shape[0]
    n = np.arange(fb.shape[1]) # time axis

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=True)
    fig.set_size_inches(figsize)
    fig.suptitle(title)

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.plot(fb[i][0])
        ax.set_xlabel('sample')

    fig.tight_layout()
    
    return fig

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    if len(class_names) > 20:
        figure = plt.figure(figsize=(32, 32))
    else:
        figure = plt.figure(figsize=(12, 12))
    plt.imshow(cm, cmap='viridis')
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=65)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "black" if cm[i, j] > threshold else "white"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = np.array(PIL.Image.open(buf))

    return image

def plotly_fig2array(fig, dims=(1200, 700)):
    """
    convert plotly figure to numpy array
    """
    import io
    from PIL import Image
    fig_bytes = fig.to_image(format="png", width=900, height=600)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)

def plotly_confusion_matrix(m, labels):
    import plotly.figure_factory as ff

    x = labels
    y = labels

    # change each element of z to type string for annotations
    m_text = [[str(y) for y in x] for x in m]

    # set up figure 
    fig = ff.create_annotated_heatmap(m, x=x, 
        y=y, annotation_text=m_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<b>Confusion matrix</b>')

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True

    return fig

def plotly_bce_classification_report(classification_report):
    """ a bar chart visualization of an sklearn classification_report
    """
    df = pd.DataFrame.from_dict(classification_report, orient='index')

    # df = pd.DataFrame.from_dict(classification_report, orient='index').reset_index()
    # df = df.melt(id_vars='index', value_vars=['precision', 'recall', 'f1-score'])

    # fig = px.bar(df, x="index", y='value', color='variable')
    ax = df.plot.bar(y=['precision', 'recall', 'f1-score'], subplots=True, figsize=(6.8*2, 4.8*2))
    if isinstance(ax, np.ndarray):
        ax = ax[0].figure
    else:
        ax = ax.figure
    return ax

def dim_reduce(emb, labels, n_components=3, method='umap', title=''):
    """
    dimensionality reduction for visualization!
    returns a plotly figure with a 2d or 3d dim reduction of ur data
    parameters:
        emb (np.ndarray): the samples to be reduced with shape (samples, features)
        labels (list): list of labels for embedding with shape (samples)
        method (str): umap, tsne, or pca
        title (str): title for ur figure
    returns:    
        fig (plotly figure): 
    """
    if method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=n_components)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f'dunno how to do {method}')
 
    proj = reducer.fit_transform(emb)

    if n_components == 2:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            instrument=labels
        ))
        fig = px.scatter(df, x='x', y='y', color='instrument',
                        title=title)

    elif n_components == 3:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            z=proj[:, 2],
            instrument=labels
        ))
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='instrument',
                        title=title)

    else:
        raise ValueError("cant plot more than 3 components")

    fig.update_traces(marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    return fig

def plot_piano_roll(m, labels):
    import plotly.figure_factory as ff
    
    y = list(range(len(m)))
    x = labels

    # change each element of z to type string for annotations
    # m_text = [[str(Y) for Y in y] for x in m]

    # set up figure 
    fig = ff.create_annotated_heatmap(m, x=x, y=y, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<b>piano roll</b>')

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True

    return fig
    