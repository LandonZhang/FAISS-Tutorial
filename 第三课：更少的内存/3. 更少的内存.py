import numpy as np
import faiss
import time

# 固定随机种子
np.random.seed(42)

# 创建全局数据与查询向量
dim = 256
data = np.random.rand(100000, dim)
query = np.random.rand(2, dim)

def time_calculator(test_function):
    def wrapper():
        print('*' * 13)
        print("程序运行开始！")
        start_time = time.time()
        test_function()
        print(f"程序运行时间是:{time.time() - start_time}")
        print('*' * 13, end='\n' * 2)
    return wrapper

@time_calculator
def test01():
    index = faiss.IndexFlatL2(dim)
    index.add(data)
    start_time_test = time.time()
    D, I = index.search(query, k=2)
    print(f"查询所需时间为:{time.time()-start_time_test:.4f}s")
    print(f"最短距离为：{D}")
    print(f"最短距离索引为：{I}")
    faiss.write_index(index, "IndexFlatL2.faiss")

@time_calculator
def test02():
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, 500)
    index.train(data)
    index.add(data)
    start_time_test = time.time()
    index.nprobe = 10  # 增加nprobe值
    D, I = index.search(query, k=2)
    print(f"查询所需时间为:{time.time()-start_time_test:.4f}s")
    print(f"最短距离为：{D}")
    print(f"最短距离索引为：{I}")
    faiss.write_index(index, "IndexIVFFlat.faiss")

@time_calculator
def test03():
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, 500, 8, 8)
    index.train(data)
    index.add(data)
    start_time_test = time.time()
    index.nprobe = 10  # 增加nprobe值
    D, I = index.search(query, k=2)
    print(f"查询所需时间为:{time.time()-start_time_test:.4f}s")
    print(f"最短距离为：{D}")
    print(f"最短距离索引为：{I}")
    faiss.write_index(index, "IndexIVFPQ.faiss")

if __name__ == "__main__":
    print("IndexFlatL2:")
    test01()
    print("IndexIVFFlat:")
    test02()
    print("IndexIVFPQ:")
    test03()
