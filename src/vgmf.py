import torch
from gmf import GMF
from mlp import MLP
from engine import Engine
from utils import use_cuda, resume_checkpoint

class VGMF(torch.nn.Module):
    def __init__(self, config):
        super(VGMF, self).__init__()

        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim']
        self.latent_dim_v = config['latent_dim_v']

        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)


        self.embedding_user_v  = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_v)
        
        self.fc_embedding = torch.nn.ModuleList()
        self.fc_embedding.append(torch.nn.Linear(in_features=2048, out_features=128))
        self.fc_embedding.append(torch.nn.ReLU())
        self.fc_embedding.append(torch.nn.Linear(in_features=128, out_features=self.latent_dim_v))

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim_mf, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices, poster_embeddings):
        ###
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        user_embedding_v  = self.embedding_user_v(user_indices)
        for idx, _ in enumerate(range(len(self.fc_embedding))):
            poster_embeddings = self.fc_embedding[idx](poster_embeddings)
        item_embedding_v  = poster_embeddings


        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)
        v_vector = torch.mul(user_embedding_v, item_embedding_v)

        ###
        vector = mf_vector + v_vector
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained GMF model"""
        config = self.config
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])
        self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data



class VGMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = VGMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(VGMFEngine, self).__init__(config)
        print(self.model)

        if config['pretrain']:
            self.model.load_pretrain_weights()

