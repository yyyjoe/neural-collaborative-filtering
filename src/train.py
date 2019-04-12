import pandas as pd
import numpy as np
from data import SampleGenerator
import config_factory
from argparse import ArgumentParser
import os
from os import path
parser = ArgumentParser()
parser.add_argument('--model', type=str, default="gmf",choices=["gmf","vgmf","mlp","neumf","vneumf"], dest="model")
args = parser.parse_args()

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

config,engine = config_factory.get_config(args.model)

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
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))


# DataLoader for training
sample_generator = SampleGenerator(ml1m_rating)
evaluate_data = sample_generator.evaluate_data

# Specify the exact model
performance=[]
total_loss=[]
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    #engine.scheduler.step()
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    loss,save_dir = engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    if(epoch%10==0 or epoch==config['num_epoch']-1):
        engine.save(config['alias'], epoch, hit_ratio, ndcg)
    performance.append([epoch,hit_ratio,ndcg])
    total_loss.append([epoch,loss])
np.save(save_dir + "/" + args.model +"_loss.npy",np.array(total_loss))
np.save(save_dir + "/" + args.model +"_accuracy.npy",np.array(performance))
