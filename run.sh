# Cora
python main.py --backbone gcn --dataset cora --lr 0.01 --num_layers 4 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --method ours --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 3 --runs 5 --data_dir ../data/

# Citeseer
python main.py --backbone gcn --dataset citeseer --lr 0.005 --num_layers 4 \
    --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 \
    --method ours --ours_layers 1 --use_graph --graph_weight 0.7 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 3 --runs 5

# Pubmed
python main.py --backbone gcn --dataset pubmed --lr 0.005 --num_layers 4 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
     --rand_split_class --valid_num 500 --test_num 1000 \
     --method ours --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01  \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 1 --runs 5


# Actor
python main.py --backbone gcn --dataset film --lr 0.1 --num_layers 8 \
    --hidden_channels 64 --weight_decay 0.0005 --dropout 0.6   \
    --method difformer --use_graph --graph_weight 0.5 --num_heads 1 --ours_use_residual --ours_use_act \
    --alpha 0.5 --ours_dropout 0.6 --ours_weight_decay 0.0005 --device 3 --runs 10 --epochs 500  --data_dir ../data/

# Deezer
python main.py --backbone gcn --rand_split --dataset deezer-europe --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --method ours --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
    --alpha 0.5  --device 2 --runs 5

# Squirrel
python main.py --backbone gcn  --dataset squirrel --lr 0.001 --num_layers 8 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.3   \
    --method difformer --ours_layers 1 --use_graph --ours_use_act --ours_use_residual --num_heads 1 \
    --alpha 0.5  --device 3 --runs 10 --data_dir ../data/

# Chameleon
python main.py --backbone gcn --dataset chameleon --lr 0.001 --num_layers 2 \
    --hidden_channels 64 --ours_layers 1 --weight_decay 0.001 --dropout 0.6  \
    --method ours --use_graph --num_heads 1 --ours_use_residual \
    --alpha 0.5  --device 1 --runs 10 --epochs 200

####### SGFormer #######
# HPU CORA
# Cora
PT_HPU_LAZY_MODE=0 python main.py --backbone gcn --dataset cora --lr 0.01 --num_layers 4 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --method ours --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 3 --runs 5 --data_dir ../data/

# HPU Citeseer
PT_HPU_LAZY_MODE=0 python main.py --backbone gcn --dataset citeseer --lr 0.005 --num_layers 4 \
    --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 \
    --method ours --ours_layers 1 --use_graph --graph_weight 0.7 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 3 --runs 5
    
# HPU Pubmed
PT_HPU_LAZY_MODE=0 python main.py --backbone gcn --dataset pubmed --lr 0.005 --num_layers 4 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
     --rand_split_class --valid_num 500 --test_num 1000 \
     --method ours --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01  \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 1 --runs 5

# OGBN-Arxiv
PT_HPU_LAZY_MODE=0 python main=batch.py --method sgformer  --dataset ogbn-arxiv --metric acc --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
    --gnn_num_layers 3  --gnn_dropout 0.5 --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
    --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0. --trans_use_residual --trans_use_weight --trans_use_bn \
    --seed 123 --runs 5 --epochs 9 --eval_step 9 --device 0


# HPU Nodeformer CORA
PT_HPU_LAZY_MODE=0 python main.py --backbone gcn --dataset cora --lr 0.01 --num_layers 4 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --method nodeformer --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 3 --runs 5 --data_dir ../data/

# HPU Nodeformer Citeseer
PT_HPU_LAZY_MODE=0 python main.py --backbone gcn --dataset citeseer --lr 0.005 --num_layers 4 \
    --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 \
    --method nodeformer --ours_layers 1 --use_graph --graph_weight 0.7 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 3 --runs 5

# HPU Nodeformer Pubmed
PT_HPU_LAZY_MODE=0 python main.py --backbone gcn --dataset pubmed --lr 0.005 --num_layers 4 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
     --rand_split_class --valid_num 500 --test_num 1000 \
     --method nodeformer --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01  \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 1 --runs 5