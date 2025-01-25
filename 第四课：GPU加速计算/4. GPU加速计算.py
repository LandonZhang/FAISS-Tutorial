# 检查电脑中的GPU配置
import faiss
import numpy as np
import time

# 固定随机种子
np.random.seed(42)

# 创建全局数据与查询向量
dim = 256
data = np.random.rand(100000, dim)
query = np.random.rand(2, dim)

# 使用上一章学的IndexIVFPQ来检测GPU的运算效率
def test():
    # 初始化CPU类index
    quantizer = faiss.IndexFlatL2(dim)
    index_cpu = faiss.IndexIVFPQ(quantizer, dim, 500, 8, 8) # 2^8 = 256

    # 转化至GPU类index
    src = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(src, 0, index_cpu)

    # 创建完整索引并添加数据
    index_gpu.train(data)
    assert index_gpu.is_trained
    index_gpu.add(data)
    start_time_test = time.time()
    index_gpu.nprobe = 1
    D, I = index_gpu.search(query, k=2)

    # 返回结果
    print(f"查询所需时间为:{time.time()-start_time_test:.4f}s")
    print(f"最短距离为：{D}")
    print(f"最短距离索引为：{I}", end='\n' + '*'*9 + '\n')

    # 将index转化为 CPU 用于存储到本地，否则会报错
    index_cpu = faiss.index_gpu_to_cpu(index_gpu)
    faiss.write_index(index_cpu, "GPU_to_CPU.faiss")


# 允许主程序
if __name__ == "__main__":
    total_start_time = time.time()
    test()
    print(f"程序总运行时间是:{time.time() - total_start_time}")