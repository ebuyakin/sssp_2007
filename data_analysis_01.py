# analysis of data
"""
functions and algorithms for data analysis and presentation
"""

# standard import set
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

er = pd.read_csv('xer_02.csv')  # read data file

#%% coding of data fields for plots:
# legend 0 - not used, 1 - x, 2 - y, 3 - z

data_selection = {'factors_id': 0,
                  'distance_threshold_multiplier': 0,  # manipulated
                  'increment_threshold': 0,  # manipulated
                  'lexicalization_noise_level': 0,  # manipulated
                  'collocations_noise_level': 0,  # manipulated
                  'message_id': 0,
                  'message': 0,
                  'threshold_per_concept': 0,
                  'total_threshold': 0,
                  'semantic_noise': 0,
                  'syntactic_noise': 0,
                  'time': 1,
                  'sentence_set': 0,
                  'verdict': 0,
                  'distance': 0,
                  'length_of_sentence': 0,
                  'depth_of_sentence': 0,
                  'n_sub_trees': 0,
                  'n_head_words': 0,
                  'n_roots': 0,
                  'n_leafs': 0,
                  'verdict_success': 0,
                  'verdict_one_word': 0,
                  'verdict_failure': 0,
                  'n_distance': 0}


#%%  Histogram of a single data field.

factor = [d for d in data_selection if data_selection[d] == 1][0]
fig = px.histogram(er, x=factor, color='verdict', facet_row='verdict', nbins=20)
fig.show()

#%% Histogram of filtered data (by verdict field)

ers = er[er.verdict == 'success']
factor = [d for d in data_selection if data_selection[d] == 1][0]
fig = px.histogram(ers, x=factor, color='verdict')
fig.show()

#%% Data

f = er.groupby(factor).time.mean()
ert = er[er.distance_threshold_multiplier == 0.3].head()

#%% dependence between two factors

ers = er[er.verdict=='success']
c = ers.n_distance
ers.columns
ers[ers.distance=='1.9361']


factor_1 = [d for d in data_selection if data_selection[d] == 1][0]
factor_2 = [d for d in data_selection if data_selection[d] == 2][0]

fig = px.scatter(ers, x=factor_1, y=factor_2, color='ntr', trendline='ols')

fig.show()
#%%
# get rid of 'xxx'
er['n_distance'] = er.apply(lambda row: 0 if row['distance'] == 'xxx' else row['distance'], axis=1)
er['ntr'] = er.apply(lambda row: str(row['total_threshold']), axis=1)
er.to_csv('xer_03_1.csv')
#%%

er = pd.read_csv('xer_03.csv')
er['n_distance'] = er.apply(lambda row: 0 if row['distance']=='xxx' else float(row['distance']), axis=1)
er.n_distance.mean()
