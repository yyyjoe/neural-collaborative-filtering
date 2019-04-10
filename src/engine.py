import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK
from datetime import datetime
import os
from os import path

class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.now = datetime.now()
        self.date_time = self.now.strftime("%m%d_%H%M%S")

        self.save_dir='checkpoints/' + config['model'] +'/drop' + str(config['dropout']) + '_dim' + str(config['latent_dim']) + '_neg' + str(config['num_negative']) + '_lr' + str(config['adam_lr'])
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.crit = torch.nn.BCELoss()
        self.scheduler = lr_scheduler.MultiStepLR(self.opt, milestones=[100,200], gamma=0.1)

    def train_single_batch(self, users, items, ratings, poster_embeddings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings, poster_embeddings = users.cuda(), items.cuda(), ratings.cuda(), poster_embeddings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items,poster_embeddings)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0.
        count=0
        for batch_id, batch in enumerate(train_loader):
            count+=1
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating, poster_embedding = batch[0], batch[1], batch[2], batch[3]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating,poster_embedding)
            if(batch_id%300==0):
                print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)
        return total_loss/count,self.save_dir

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            test_embeddings, negative_embeddings = evaluate_data[4], evaluate_data[5]
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()
                test_embeddings = test_embeddings.cuda()
                negative_embeddings = negative_embeddings.cuda()

            test_scores = self.model(test_users, test_items, test_embeddings)
            negative_scores = self.model(negative_users, negative_items, negative_embeddings)
            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()

            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
        return hit_ratio, ndcg

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        
        model_dir = self.save_dir + self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
