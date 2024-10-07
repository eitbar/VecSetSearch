import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
选择了10000个查询和100000个文档为例
'''


# 指定文件范围
start = 10000
end = 100000
step = 10000

# 生成随机的行索引
num_rows = 10000
first_file = np.load(f'/ssddata/ytianbc/test/marco_embeddings/similarities_labels_{start}.npy')
indices = np.random.choice(first_file.shape[0], num_rows, replace=False)

# 读取并拼接所有的.npy文件
data = []
for i in tqdm(range(start, end + 1, step)):
    file = f'/ssddata/ytianbc/test/marco_embeddings/similarities_labels_{i}.npy'
    arr = np.load(file)
    arr = arr[indices]  # 选择随机的行
    data.append(arr)

# 按列拼接
data = np.concatenate(data, axis=1)
print(data.shape)


n, m = data.shape  # dimensions of the matrix

# Generate a binary mask with the same dimensions as data
# Here we assume that approximately 20% of the entries are observed
mask = np.random.choice([0, 1], size=(n, m), p=[0.8, 0.2])

# Set the first column to 1
mask[:, 0] = 1

# 100 代表 之后代码里的niters = 100
np.save('mask_gene.npy', mask)
np.save('data_gene.npy', data)
np.save('indices_gene.npy', indices)

# data = np.load('data_100.npy')
# mask = np.load('mask_100.npy')

def als(X, mask, rank, niters, lambda_, tol=1e-3):
    """
    Alternating Least Squares algorithm for matrix factorization
    X: matrix to factorize
    mask: binary mask of observed entries
    rank: rank of the factorization
    niters: number of iterations
    lambda_: regularization parameter
    tol: tolerance for convergence
    """
    n, m = X.shape
    A = np.random.rand(n, rank)
    B = np.random.rand(m, rank)
    mse_prev = np.inf
    mse_values = []
    print("----------- Begin Iteration -----------")
    for _ in tqdm(range(niters)):
        # 进度条
        if _ % 100 == 0:
            print(f"Iteration {_} / {niters}")
        target = X + (1 - mask) * (np.dot(A, B.T))
        A = np.linalg.solve(np.dot(B.T, B) + lambda_ * np.eye(rank), np.dot(target, B).T).T
        target = X + (1 - mask) * (np.dot(A, B.T))
        B = np.linalg.solve(np.dot(A.T, A) + lambda_ * np.eye(rank), np.dot(target.T, A).T).T

        mse_curr = np.mean((mask * (X - np.dot(A, B.T)))**2)
        mse_values.append(mse_curr)
        if np.abs(mse_prev - mse_curr) < tol:
            break
        mse_prev = mse_curr
    print(f"mse_values : {mse_values}")

    # plt.plot(mse_values)
    # plt.xlabel('Iteration')
    # plt.ylabel('MSE')
    # plt.show()
    
    # 绘制MSE随迭代次数的变化
    plt.figure(figsize=(8, 4))
    plt.plot(mse_values, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('MSE vs. Iteration')
    plt.grid(True)
    plt.show()
    plt.savefig('mse_vs_iteration_gene.png')
    plt.close()

    return A, B, mse_values

# X = np.array([[0.5, 2, 4], [1, 3, 5]])
# mask = np.array([[1, 0, 1], [1, 1, 1]])
masked_data = data * mask
A, B, mse_values = als(masked_data, mask, rank = 200, niters= 100, lambda_= 0.2)

# Save the factorized matrices
np.save('A_gene.npy', A)
np.save('B_gene.npy', B)

# A = np.load('A_100.npy')
# B = np.load('B_100.npy')
# data = np.load('data_100.npy')
# mask = np.load('mask_100.npy')

predictions =  masked_data + (1-mask)*np.dot(A, B.T)

test_mse_curr = np.mean(((data - predictions))**2)
print(f"Test MSE: {test_mse_curr}")


k = 100  # number of top elements to select


gt_top_k_indices = np.argpartition(data, -k, axis=1)[:, -k:]
gt_top_k_values = np.take_along_axis(data, gt_top_k_indices, axis=1)

# recall for seen queries
k_values = list(range(100, 2100, 100))
recall_values = []

for k_cand in tqdm(k_values):
    pre_top_k_indices = np.argpartition(predictions, -k_cand, axis=1)[:, -k_cand:]
    recall_list = []
    for i in range(gt_top_k_indices.shape[0]):
        recall = np.isin(pre_top_k_indices[i], gt_top_k_indices[i]).sum() / k
        recall_list.append(recall)
    recall_at_k = np.mean(recall_list)
    recall_values.append(recall_at_k)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_values, recall_values, marker='o')
plt.title('Recall@k vs k_cand')
plt.xlabel('k_cand')
plt.ylabel('Recall@k')
plt.grid(True)
plt.show()
plt.savefig('recall_vs_k_cand_seen.png')

'''
lambda_= 0.2
Test MSE: 0.49526037604898226
recall@100: 0.476904  # k_candidates = 100
recall@100: 0.8589899999999999 # k_candidates = 1000


'''