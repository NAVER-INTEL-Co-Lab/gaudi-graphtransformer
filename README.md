# Gaudi-graphtransformer
HPU version of the graph transformer architectures

## Introduction
This repository is related to the template for graph transformer by using Intel Gaudi-v2 devices.
Specifically, our goal is to provide the following contents:

- Implementation of Graph Transformer models which is compatible to Intel Gaudi-v2.
- Developing sparse matrix multiplication kernels in TPC-C levels which helps efficient computation

The main difference between original deep learning structures and graph neural network is sparsity of the dataset.
Since, the graph datasets are composed with high sparsity. To resolve this issues, many GNN frameworks (such as PyG and DGL) providing spmm operations.
Unfortunately, current version of Intel Gaudi-v2 is not supporting spase matrix multiplication [Intel Forum](https://forum.habana.ai/t/questions-regarding-the-architecture-about-habana-gaudi/355/6)

We adapt and modify compatibility based on [SGFormer official](https://github.com/qitianwu/SGFormer) codes.


## Implemented models

- SGFormer (NeurIPS 2023) 
- GraphGPS
- Cobformer
- Nodeformer

## How to run this repository

```shell
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v /home/irteamsu:/root vault.habana.ai/gaudi-docker/1.17.1/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest
pip install -r requirements_docker.txt
```

Please refer to run.sh commands to run each models to HPU

For the large datasets such as ogbn-arxiv, ogbn-proteins, we conducted subgraph sampling training due to the memory issues.

Since current version of the codes were implemented with dense matrix multiplication version, ogbn-arxiv need 100GB for the full-graph training.

## Functionality

We will support spmm kernels in TPC-C kernel levels.

## General Issues

```python
 #model = torch.compile(model, backend = "hpu_backend")
 device = "hpu"
 model = model.to(device)
```

Current version of the code occurs error when we use torch.compile() with backend = "hpu_backend". 
It seems to be related to not work with dynamic shapes of the tensors when we move to HPU.

For the speedup, it needs to be resolved in the future version of the codes.

