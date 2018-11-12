
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collections


def get_weights_latent_genes(model, h_layers,num_latent_dimensions, genes):
    decoder_weights = []

    for index in np.arange(h_layers+8,2*(h_layers+2)+8,2):
        decoder_weights.append(list(model.parameters())[index].cpu().detach().numpy())

    result = decoder_weights[0]
    for index in range(1, len(decoder_weights)):
        result = np.dot(result.T, decoder_weights[index].T)
        result=result.T
    W_genes = pd.DataFrame(result.T, index=range(num_latent_dimensions), columns=genes[1:]) 
    W_genes_abs = pd.DataFrame(abs(result.T), index=range(num_latent_dimensions), columns=genes[1:])  

    return W_genes, W_genes_abs

def zebrafish_compute_high_weight_genes_latent_dim(weights_latent_genes, latent_dimension, genes):
    latent_dim_weights = weights_latent_genes.iloc[latent_dimension:latent_dimension + 1].T
    latent_dim_weights = latent_dim_weights.sort_values(by=[latent_dimension], ascending=False)
    #latent_dim_weights.to_csv('results/HighWeightGenes/zebrafish_high_weight_genes_for_latent_dimension_' +
    #                          str(latent_dimension) + model + '.csv')
    return latent_dim_weights

def compare_genes(weights_latent_genes, cluster, relevance, genes, Map_comp_clust):
    # compare relevant genes between components for a specific cluster
    #get components related to the cluster
    Comp=np.where(Map_comp_clust['Cell type']==str(cluster))[0]
    #get weights for all genes in these components
    Weights=[]
    Names=[]
    common=list(genes[1:])
    C=[]
    for comp in Comp:
        W_genes= zebrafish_compute_high_weight_genes_latent_dim(weights_latent_genes, comp, genes).values
        Weights.append(W_genes)
        W_genes_names= zebrafish_compute_high_weight_genes_latent_dim(weights_latent_genes, comp, genes).index.values
        Names.append(W_genes_names)    #find which genes are common among components
        common=list(set(common) & set(W_genes_names[0:relevance]))
        C.append(common)
    return Weights, Names, C

