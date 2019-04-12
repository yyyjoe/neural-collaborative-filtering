# Using Movie Posters in Movie Recommender System
Movie posters signal the style and content of the films. By including the visual information, the movie recommend system can better filter preferences according to user's choices. A deep learning recommend system, Neural collaborative filtering(NCF), obtain a great performance by using neural networks to learn user-item embedding features. This repo extend NCF model as visual NCF by learning the user-poster embedding features.

The authors of NCF published [a nice implementation](https://githubcom/hexiangnan/neural_collaborative_filtering) written in tensorflow(keras), and [LaceyChen17](https://githubcom/LaceyChen17/neural-collaborative-filtering) provides the implementation written in **pytorch**. This repo implement visual NCF model based on [LaceyChen17's code](https://githubcom/LaceyChen17/neural-collaborative-filtering).



## Dataset
[The Movielens 1M Datasets](https://grouplens.org/datasets/movielens/1m/) is used to test the repo.

## Movie Poster Embeddings
We scrape the movie poster images from IMDB website and use Resnet152 (pretrained on ImageNet) to obtain our movie poster embedding features. The correspondence movieId to imdbId is in the file links.csv. (or you can just download the embeddings with  `download.py`)

## How to run
- Clone this repository:
```bash
$ git clone --recursive https://github.com/yyyjoe/neural-collaborative-filtering.git
```

- Cd to src directory:
```bash
$ cd src
```

- Download movie poster embedding features:
```bash
$ python download.py
```

- Train the model (checkpoints saved in `./checkpoints/MODEL_NAME/`)
```bash
# python train.py --model MODEL_NAME(gmf, vgmf, mlp, neumf, vneumf)
$ python train.py --model gmf
```

- You can modify the training setting in `config_factory.py`

## Performance
The hyper params are not tuned. Better performance can be achieved with careful tuning, especially for the MLP model. Pretraining the user embedding & item embedding might be helpful to improve the performance of the MLP model. 

Experiments' results with `num_negative_samples = 4` and `dim_latent_factor=8`  are shown as follows

![GMF V.S. MLP](./res/figure/factor8neg4.png)

Note that the MLP model was trained from scratch but the authors suggest that the performance might be boosted by pretrain the embedding layer with GMF model.

![NeuMF pretrain V.S no pretrain](./res/figure/neumf_factor8neg4.png)

The pretrained version converges much faster.

### L2 regularization for GMF model
Large l2 regularization might lead to the bug of  `HR=0.0 NDCG=0.0`

### L2 regularization for MLP model
a bit l2 regulzrization seems to improve the performance of the MLP model

![L2 for MLP](./res/figure/mlp_l2_reg.png)

### MLP with pretrained user/item embedding
Pre-training the MLP model with user/item embedding from the trained GMF gives better result.

MLP network size = [16, 64, 32, 16, 8]

![Pretrain for MLP](./res/figure/mlp_pretrain_hr.png)
![Pretrain for MLP](./res/figure/mlp_pretrain_ndcg.png)

### Implicit feedback without pretrain
Ratings are set to 1 (interacted) or 0 (uninteracted). Train from scratch.
![binarize](./res/figure/binarize.png) 

### Pytorch Versions
The repo works under torch 1.0. You can find the old versions working under torch 0.2 and 0.4 in **tags**.

