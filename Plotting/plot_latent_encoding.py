import numpy as np
import math
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from keras.models import load_model

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_latent_encoding(labels, tsne_vae, tsne_dis, n_components):
    # plot the cells with colours according to labels=clusters
    #pick the clusters/labels used
    #labels=clusters_kmeans_dis
    #n_components=50

    #plot VAE encoding
    #tsne_vae=sample_enc_tsne
    vae_df2 = pd.DataFrame( data={'x-TSNE': tsne_vae[:, 0], 'y-TSNE': tsne_vae[:, 1], 'Model': 'DiffVAE'})
    vae_df3 = pd.DataFrame( data={'clusters': labels})
    vae_df = vae_df2.join(vae_df3)

    #plot dis-VAE encoding
    #tsne_dis=sample_enc_tsne_dis
    dis_vae_df2 = pd.DataFrame( data={'x-TSNE': tsne_dis[:, 0], 'y-TSNE': tsne_dis[:, 1], 'Model': 'Disentangled-DiffVAE'})
    dis_vae_df3 = pd.DataFrame( data={'clusters': labels})
    dis_vae_df = dis_vae_df2.join(dis_vae_df3)

    result_df = pd.concat([vae_df, dis_vae_df])
    #result_df.to_csv('./Zebrafish/plots_data/tsne_results_' + str(n_components) + '.csv')
    #result_df = pd.read_csv('./Zebrafish/plots_data/tsne_results_' + str(n_components) +'.csv')

    sns_plot = sns.lmplot(data=result_df, x='x-TSNE', y='y-TSNE', col='Model', hue = 'clusters',col_wrap=2,
                              fit_reg=False, size=10, legend=False)
    plt.show()
    #return sns_plot


def plot_GE_latent_encoding(labels, VAE_result, dis_VAE_result, n_components, genes_names, markers_indexes):
    label_map=[labels[i,:] for i in np.arange(labels.shape[0])]
    label_maps=[str(genes_names[markers_indexes,:][i,2]) for i in np.arange(labels.shape[0])]
    Label_Maps={label_maps[i]:label_map[i] for i in np.arange(labels.shape[0])}

    vae_df2 = pd.DataFrame( data={'x-dim': VAE_result[:, 0], 'y-dim': VAE_result[:, 1], 'Model': 'DiffVAE'})
    vae_df3 = pd.DataFrame( data=Label_Maps)
    vae_df = vae_df2.join(vae_df3)

    dis_vae_df2 = pd.DataFrame( data={'x-dim': dis_VAE_result[:, 0], 'y-dim': dis_VAE_result[:, 1], 'Model': 'Disentangled-DiffVAE'})
    dis_vae_df3 = pd.DataFrame( data=Label_Maps)
    dis_vae_df = dis_vae_df2.join(dis_vae_df3)

    result_df = pd.concat([vae_df, dis_vae_df])

    for g in genes_names[markers_indexes,:][:,2]:
        sns_plot = sns.lmplot(data=result_df, x='x-dim', y='y-dim', col='Model', hue = g, col_wrap=2, palette='coolwarm',
                              fit_reg=False, size=10, legend=False)
        fig1 = sns_plot.fig
        fig1.suptitle(str(g))        
        sns_plot.savefig('./results/figures/tsne_results_zebrafish_' + str(g) + '.pdf')



    plt.show()
    