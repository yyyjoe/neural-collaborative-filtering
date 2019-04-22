from gmf import GMFEngine
from vgmf import VGMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from vneumf import VNeuMFEngine
import numpy as np
import os
from os import path
gmf_config = {'alias': '',
              'model': 'gmf',
              'dropout': 0,
              'num_epoch': 200,
              'batch_size': 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 4,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

vgmf_config = {'alias': '',
              'model': 'vgmf',
              'dropout': 0,
              'num_epoch': 200,
              'batch_size': 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 4,
              'latent_dim_v': 4,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/gmf/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir':'/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}


mlp_config = {'alias': '',
              'model': 'mlp',
              'dropout': 0.1,
              'num_epoch': 200,
              'batch_size': 1024,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 1,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/gmf/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir':'/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': '',
                'model': 'neumf',
                'dropout': 0.1,
                'num_epoch': 200,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 4,
                'latent_dim_mlp': 4,
                'latent_dim': 4,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0,
                'use_cuda': True,
                'device_id': 2,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/gmf/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/mlp/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir':'/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }


vneumf_config = {'alias': '',
                'num_epoch': 200,
                'dropout': 0.1,
                'model': 'vneumf',
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 4,
                'latent_dim_mlp': 4,
                'latent_dim_v': 4,
                'latent_dim': 4,
                'num_negative': 4,
                'layers': [8,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'layers_v': [8,32,16,8],
                'l2_regularization': 0,
                'use_cuda': True,
                'device_id': 3,
                'pretrain': False,
                'pretrain_neumf': 'checkpoints/gmf/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'model_dir':'/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

engine_map = {
    'gmf': GMFEngine,
    'mlp': MLPEngine,
    'neumf': NeuMFEngine,
    'vneumf':VNeuMFEngine,
    'vgmf':VGMFEngine,
}

config_map = {
    'gmf': gmf_config,
    'mlp': mlp_config,
    'neumf': neumf_config,
    'vneumf': vneumf_config,
    'vgmf': vgmf_config,
}


def get_config(name):
    if name not in engine_map:
        raise ValueError('Name of dataset unknown %s' % name)
    if not os.path.exists('checkpoints/'+ name):
        os.mkdir('checkpoints/'+ name)
    print("Load model " + name)
    return config_map[name],engine_map[name](config_map[name])
