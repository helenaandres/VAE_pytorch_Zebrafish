
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collections


def compute_diff_capacity_latent_dim(latent_dim_data, labels, model):
    # Compute the percentage distribution for the cells more than 2x standard deviation away from the mean
    latent_diff = np.zeros(shape=(latent_dim_data.shape[1], 6))
    print (latent_dim_data.shape[1])

    for latent_dim in range(latent_dim_data.shape[1]):

        latent_dim_across_cells = latent_dim_data[:, latent_dim]
        latent_dim_mean = np.mean(latent_dim_across_cells)
        latent_dim_std = np.std(latent_dim_across_cells)

        variable_cells = np.where((latent_dim_across_cells > latent_dim_mean + 2*latent_dim_std)|
                                  (latent_dim_across_cells < latent_dim_mean - 2*latent_dim_std))

        variable_labels = labels[variable_cells]
        variable_cells = variable_labels.tolist()
        
        counter_dict = {x: variable_cells.count(x) for x in range(0, 5)}
        counter = np.array(list(counter_dict.values())) / float(len(variable_cells))
        counter = np.around(counter * 100.0, decimals=2)
        latent_diff[latent_dim][1:] = counter
        latent_diff[latent_dim][0] = int(latent_dim)

    latent_diff = pd.DataFrame(latent_diff, columns=['Latent dimension', '1', '2',
                                                     '3', '4', '5'])
    latent_diff['Latent dimension'] = latent_diff['Latent dimension'].astype(int)

    latent_diff = latent_diff.melt(id_vars=['Latent dimension'], value_vars=['1', '2',
                                                                             '3', '4', '5'],
                                   var_name='Cell type', value_name='Percentage')
    print (latent_diff)
    sns.set(font_scale=2.5)
    flatui = ["#9b59b6", "#2ecc71", "#95a5a6", "#e74c3c", "#3498db", "#34495e", ]
    sns.set_palette(sns.color_palette(flatui))
    sns.set_style('darkgrid')
    g = sns.factorplot(x='Cell type', y='Percentage', col='Latent dimension', data=latent_diff, saturation=.5,
                       col_wrap=5,
                       kind="bar", ci=None, aspect=1.3, legend_out=True)

    g.set_xticklabels(rotation=70)

    plt.show()
    return latent_diff



def Map_Comp_clust(latent_diff):
    Map_components_clusters=np.array([latent_diff['Latent dimension'][np.where(latent_diff['Percentage']>50.)[0]],
                                 latent_diff['Cell type'][np.where(latent_diff['Percentage']>50.)[0]]])
    Map_comp_clust = pd.DataFrame(Map_components_clusters.T, columns=['Latent dimension', 'Cell type'])
    return Map_comp_clust






