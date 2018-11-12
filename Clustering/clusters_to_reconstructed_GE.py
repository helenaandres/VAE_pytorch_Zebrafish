
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collections


def compute_diff_capacity_genes(latent_dim_data, labels, model, gene_names):
    # Compute the percentage distribution for the cells more than a standard deviation away from the mean
    latent_diff = np.zeros(shape=(latent_dim_data.shape[1], 7))
    print (latent_dim_data.shape[1])

    for latent_dim in range(latent_dim_data.shape[1]):

        latent_dim_across_cells = latent_dim_data[:, latent_dim]
        latent_dim_mean = np.mean(latent_dim_across_cells)
        latent_dim_std = np.std(latent_dim_across_cells)

        variable_cells = np.where((latent_dim_across_cells > latent_dim_mean + 2*latent_dim_std)|
                                  (latent_dim_across_cells < latent_dim_mean - 2*latent_dim_std))
        #print(variable_cells)

        variable_labels = labels[variable_cells]
        variable_cells = variable_labels.tolist()
        
        counter_dict = {x: variable_cells.count(x) for x in range(0, 6)}
        print(counter_dict)
        counter = np.array(list(counter_dict.values())) / float(len(variable_cells))
        #print(counter)
        counter = np.around(counter * 100.0, decimals=2)
        latent_diff[latent_dim][1:] = counter
        latent_diff[latent_dim][0] = int(latent_dim)

    latent_diff = pd.DataFrame(latent_diff, columns=['Genes', 'Names','1', '2','3', '4', '5'])
    latent_diff['Genes'] = latent_diff['Genes'].astype(int)
    latent_diff['Names'] = gene_names

    latent_diff = latent_diff.melt(id_vars=['Genes', 'Names'], value_vars=['1', '2','3', '4', '5'],
                                   var_name='Cell type', value_name='Percentage')
    

    return latent_diff





