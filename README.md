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

## Functionality

We will support spmm kernels in TPC-C kernel levels.

