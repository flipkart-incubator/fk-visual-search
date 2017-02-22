import os
import glob
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import json
import operator
import time
import numpy as np
import lmdb
import sys
import math

__author__ = 'ananya.h'


def construct_fv_map_from_lmdbs(crops_index_path, catalog_index_path):
    crops_env = lmdb.open(crops_index_path)
    catalog_env = lmdb.open(catalog_index_path)
    num_crops = crops_env.stat()["entries"]
    num_catalog = catalog_env.stat()["entries"]
    needed_ids = list()
    fv_size = 4097
    data_map = np.zeros((num_crops + num_catalog, fv_size), dtype=np.float32)
    index_id_map = {}
    n = 0

    print "Loading crops"
    with crops_env.begin() as txn:
        for k, fv_string in txn.cursor():
            index_id_map[n] = k
            needed_ids.append(k)
            data_map[n, :] = np.append(np.fromstring(fv_string, dtype=np.float32), 0)
            n += 1
            if n % 10000 == 0:
                print("Finished reading fvs for " + str(n))

    print "Loading catalog"
    with catalog_env.begin() as txn:
        for k, fv_string in txn.cursor():
            index_id_map[n] = k
            data_map[n, :] = np.append(np.fromstring(fv_string, dtype=np.float32), 0)
            n += 1
            if n % 10000 == 0:
                print("Finished reading fvs for " + str(n))
    return index_id_map, needed_ids, num_crops, data_map.flatten()


def compute_nn(index_map, data, fv_size, num_crops, output_db, needed_ids=None, k=1000, device_id=0):
    env = lmdb.open(output_db, map_size=500 * 1024 * 1024 * 1024)
    dev = cuda.Device(device_id)
    ctx = dev.make_context()
    for i in range(100):  # 100 is preconfigured. Just a magic number
        try:
            pycuda.driver.Context.pop()
        except:
            break
    pycuda.driver.Context.push(ctx)
    reverse_index = {v: k for k, v in index_map.items()}
    if not needed_ids:
        needed_ids = index_map.values()
    needed_indices = {item_id: reverse_index[item_id] for item_id in needed_ids if item_id in reverse_index}
    results = {}
    n = len(data) / fv_size
    row = np.zeros(n, dtype=np.float32)
    data_gpu = cuda.mem_alloc(data.nbytes)
    row_gpu = cuda.mem_alloc(row.nbytes)
    cuda.memcpy_htod(data_gpu, data)
    cuda.memcpy_htod(row_gpu, row)
    print("Preallocation is completed")

    mod = SourceModule("""
    __global__ void compute_l2_dist(float *data, float *output, int a, int count)
    {
        int dim = """ + str(fv_size) + """;
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if(id < count)
        {
            float dist = 0;
            float total_dist = 0;
            int i;
            for(i=0; i<dim; i++)
            {
                dist = (data[a * dim + i] - data[id * dim + i]);
                total_dist += dist * dist;
            }
            output[id] = total_dist;
        }
    }""")

    # func = mod.get_function("compute_chi_square_dist")
    func = mod.get_function("compute_l2_dist")
    func.prepare("PPii")
    print("Compilation is complete")
    THREADS_PER_BLOCK = 32
    NUMBER_OF_BLOCKS = int(math.ceil(float(n) / float(THREADS_PER_BLOCK)))

    count = 0
    start = time.time()
    for item_id, index in needed_indices.iteritems():
        func.prepared_call((NUMBER_OF_BLOCKS, 1), (THREADS_PER_BLOCK, 1, 1), data_gpu, row_gpu, index, n)
        cuda.memcpy_dtoh(row, row_gpu)
        pruned_row = row[num_crops:]
        minIndices = np.argpartition(pruned_row, k)
        nns = zip([index_map[num_crops + x] for x in minIndices[:k]], pruned_row[minIndices[:k]])
        nns = sorted(nns, key=operator.itemgetter(1))
        results[item_id] = [(x[0], str(x[1])) for x in nns]
        count += 1
        if count % 1000 == 0:
            with env.begin(write=True) as txn:
                for i in results:
                    txn.put(i, json.dumps(results[i]))
            results = {}
            end = time.time()
            print("Finished Processing " + str(count) + " in " + str(end - start))
            start = end
    with env.begin(write=True) as txn:
        for i in results:
            txn.put(i, json.dumps(results[i]))


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 3:
        print("Usage: python generate_knn_cuda.py <crops_index_path> <catalog_index_path> <output_db>")
        sys.exit(1)
    crops_index_path = args[0]
    catalog_index_path = args[1]
    output_db = args[2]
    index_id_map, needed_ids, num_crops, data = construct_fv_map_from_lmdbs(crops_index_path, catalog_index_path)
    fv_size = 4097
    compute_nn(index_id_map, data, fv_size, num_crops, output_db, needed_ids=needed_ids, k=1000)
