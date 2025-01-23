import numpy as np
import faiss
import time


# 使用暴力索引完成数据库的建立与检索
def index_flat_ip(data, dim, query):
    index = faiss.IndexFlatIP(dim)
    index.add(data)
    start_time = time.time()
    D, I = index.search(query, k=2)
    print(f"程序运行的时间是:{(time.time() - start_time):.4f}s")
    print(D)
    print(I)
    return D, I


# 使用IndexIVFFlat的方式创建索引并查询
def index_ivf_flat(data, dim, query, nlist=1000, nprobe=1):
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    assert not index.is_trained  # 由于存在聚类模型，所以需要进行训练
    index.train(data)  # 传入数据，找到规定数量的聚类中心
    assert index.is_trained
    index.add(data)
    index.nprobe = nprobe
    start_time = time.time()
    D, I = index.search(query, k=2)
    print(f"程序运行的时间是:{(time.time() - start_time):.4f}s")
    print(D)
    print(I)
    return D, I


if __name__ == "__main__":
    # 创建原始数据
    data = np.random.rand(100_000, 256)
    dim = 256
    # 创建查询
    query = np.random.rand(2, 256)
    # 执行查询
    D_flat, I_flat = index_flat_ip(data, dim, query)

    # 改变nprobe的值，查看两种方法什么时候结果一致
    nprobes = [1, 2, 3, 4, 5]

    for nprobe in nprobes:
        D_ivf, I_ivf = index_ivf_flat(data, dim, query, nprobe=nprobe)
        if np.array_equal(D_flat, D_ivf):
            print(f"nprobe={nprobe}时，两种方法结果一致")
        else:
            print(f"nprobe={nprobe}时，两种方法结果不一致")