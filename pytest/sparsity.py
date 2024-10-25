import numpy as np

def check_sparsity(file_path, threshold=0.7):
    # 加载 .npy 文件
    array = np.load(file_path)
    print(array.shape)
    # 计算数组中的元素总数和零元素的数量
    total_elements = array.size
    zero_elements = np.count_nonzero(array == 0)

    # 计算稀疏性比例
    sparsity_ratio = zero_elements / total_elements

    # 打印稀疏性信息
    print(f"Total elements: {total_elements}")
    print(f"Zero elements: {zero_elements}")
    print(f"Sparsity ratio: {sparsity_ratio:.2f}")

    # 判断是否稀疏，阈值默认为 0.7，即70%是零元素则视为稀疏
    if sparsity_ratio >= threshold:
        print("The array is sparse.")
    else:
        print("The array is not sparse.")

if __name__ == "__main__":
    # 你可以在这里指定你的 .npy 文件路径
    file_path = "../data/doclens104.npy"
    
    # 调用函数并检查稀疏性
    check_sparsity(file_path)
