# VecSetSearch
Vector Set Search Algorithm



## Dataset

**Marco_embedding**: 

Download datasets from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/ytianbc_connect_ust_hk/EuaO6KmFlR5JpeWUUKLH6ccBEpFudB8yEJYBGnPRpX-G3g?e=zyZvsf).
Pls see https://microsoft.github.io/msmarco/Datasets for more details about datasets - MS MARCO (MicroSoft MAchine Reading COmprehension).
We follow https://github.com/ThirdAIResearch/Dessert to generate `doclensxx.npy` and `encodingx_float16.npy` (For end-to-end testing, we can skip these two intermediate files and directly make modifications on [colBERT/PLAID](https://github.com/stanford-futuredata/ColBERT)).
BTW, https://github.com/CosimoRulli/emvb is a good optimization based on PLAID.

The dataset contains a total of 8.8 million passages. Their embeddings are stored across 354 separate `.npy` files, with each embedding having a dimension of 128. Since the number of embeddings per passage varies, we use `doclens` files to store the number of embeddings for each passage. This allows us to correctly split the corresponding `encoding` files.

For example, in `doclens1.npy`, there are 25,000 integers, such as `[63, 52, 63, ...]`. This means `encoding1_float16.npy` contains embeddings for 25,000 passages. The first 63 rows (63 * 128) correspond to passage 1, the next 52 rows correspond to passage 2, and so on.

The traing query dataset `queries_embeddings.npy`'s format is 101093 * 32 * 128, which means there are 10k queries in total, and each query consists of 32 128-d vectors. `reshaped_queries_embeddings.npy`'s format is 101093 * (32 * 128).



`similarities_labels_xxx.npy` files are ground truth for our method. We compute the distance (sum of MaxSim) between each query and passage pair. For example, `similarities_labels_10000.npy` stores the similaries beween all queries and first 10000 passages. Here we provide the results for the first 1M passages for testing.


## Experiments
### HNSW Demo
'''
cd hnswlib/
mkdir build
cd build
cmake ..
make
./example_vecset_search
'''