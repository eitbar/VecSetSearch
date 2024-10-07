# VecSetSearch
Vector Set Search Algorithm



## Dataset

**Marco_embedding**: 

`doclensxx.npy` and `encodingx_float16.npy`
The dataset contains a total of 8.8 million passages. Their embeddings are stored across 245 separate `.npy` files, with each embedding having a dimension of 128. Since the number of embeddings per passage varies, we use `doclens` files to store the number of embeddings for each passage. This allows us to correctly split the corresponding `encoding` files.

For example, in `doclens1.npy`, there are 25,000 integers, such as `[63, 52, 63, ...]`. This means `encoding1_float16.npy` contains embeddings for 25,000 passages. The first 63 rows (63 * 128) correspond to passage 1, the next 52 rows correspond to passage 2, and so on.

The traing query dataset `queries_embeddings.npy`'s format is 101093 * 32 * 128, which means there are 10k queries in total, and each query consists of 32 128-d vectors.


`similarities_labels_xxx.npy` files are ground truth for our method. We compute the distance (sum of MaxSim) between each query and passage pair. For example, `similarities_labels_10000.npy` stores the similaries beween all queries and first 10000 passaages.
