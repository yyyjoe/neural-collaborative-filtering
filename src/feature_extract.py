import torch
import numpy as np
import os.path as osp
from PIL import Image
from torchvision import transforms
import pandas as pd
from gmf import GMF
from mlp import MLP
from engine import Engine
from utils import use_cuda, resume_checkpoint
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
class NeuMF(torch.nn.Module):
    def __init__(self):
        # Poster Feature
        super(NeuMF, self).__init__()
        resnet50 = models.resnet152(pretrained=True)
        modules = list(resnet50.children())[:-1]
        resnet50 =torch.nn.Sequential(*modules)
        for p in resnet50.parameters():
            p.requires_grad = False
        self.resnet50=resnet50

    def forward(self, img):
        item_embedding_v = torch.squeeze(self.resnet50(img))

        return item_embedding_v
    
class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self,ratings,poster_dict):
        self.poster_dict=poster_dict
        self.ratings=ratings
        self.img_path="./data/posterImages/"
        self.transform_train = transforms.Compose([
                                                    transforms.Resize((224, 224)),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                                    ]
                                                )

    def instance_a_train_loader(self):
        """instance train loader for one training epoch"""
        train=[]
        for item_id,movie_id in self.poster_dict.items():
            img_path = self.img_path + str(movie_id) + ".jpg"
            train.append(img_path)
            
        trainloader = DataLoader(
            ImageDataset(train, transform=self.transform_train),
            batch_size=1, shuffle=False)

        return trainloader

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = read_image(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


# Load Data
ml1m_dir = 'data/ml-1m/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
# Reindex
print(ml1m_rating)
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')

tmp=zip(ml1m_rating["itemId"],ml1m_rating["mid"])
poster_dict={}
for row in tmp:
    poster_dict[row[0]]=row[1]

ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]

sample_generator = SampleGenerator(ml1m_rating,poster_dict)
train_loader = sample_generator.instance_a_train_loader()

model = NeuMF()
model.eval()
model.cuda()

embedding=[]
for batch_id, batch in enumerate(train_loader):
    imgs = batch.cuda()
    embedding_pred = model(imgs)
    embedding.append(embedding_pred.cpu().numpy())
np.save('./data/ml1m_embeddings.npy',np.array(embedding))