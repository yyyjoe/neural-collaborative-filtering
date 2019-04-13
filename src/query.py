import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
from data import SampleGenerator
import config_factory
from argparse import ArgumentParser
import os
from os import path
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Load Data
ml1m_dir = 'data/ml-1m/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
# Reindex
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')

tmp=zip(ml1m_rating['itemId'],ml1m_rating['mid'])
poster_dict={}
for row in tmp:
    poster_dict[row[0]]=row[1]


ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))


poster_embedding_path="./data/ml1m_embeddings.npy"
poster_embeddings = np.load(poster_embedding_path)

ranking = np.zeros((len(item_id), len(item_id)))
similarity = cosine_similarity(poster_embeddings, poster_embeddings)
# ax enables access to manipulate each of subplots

randomIdx = np.random.randint(ml1m_rating.itemId.min(), high = ml1m_rating.itemId.max())

fig=plt.figure()
columns = 5
rows = 1
row = similarity[randomIdx, :]
ranking[randomIdx, :] = np.flip(np.argsort(row))

ax = []
count = 0
idx = 0
while count < columns*rows:
    try:
        image = mpimg.imread("data/ml-1m/posterImage/"+str(poster_dict[ranking[randomIdx, idx]])+'.jpg')
    except:
        print(str(poster_dict[idx])+'.jpg not exist')
        idx += 1
        continue
    
    ax.append( fig.add_subplot(rows, columns, count+1))
    ax[-1].set_title(poster_dict[ranking[randomIdx, idx]])
    ax[-1].axis('off')
    plt.imshow(image)
    idx += 1    
    count+=1

plt.show()
