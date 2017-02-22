import json
import os
import sys
import lmdb

__author__ = 'ananya.h'


def compute_recall(vertical, path_to_nn_lmdb, k_s=[1, 3, 5, 10, 20]):
    env = lmdb.open(path_to_nn_lmdb)
    base_dir = "/data/street2shop"
    meta_dir = os.path.join(base_dir, "meta", "json")
    retrieval_path = os.path.join(meta_dir, "retrieval_"+vertical+".json")
    test_data = os.path.join(meta_dir, "test_pairs_"+vertical+".json")
    image_dir = os.path.join(base_dir, "structured_images", vertical)
    query_dir = os.path.join(base_dir, "structured_images", vertical+"_query")

    with open(retrieval_path) as jsonFile:
        data = json.load(jsonFile)
    photo_to_product_map = {}
    product_to_photo_map = {}
    for info in data:
        photo_to_product_map[info["photo"]] = info["product"]
    for photo in photo_to_product_map:
        product = photo_to_product_map[photo]
        if product not in product_to_photo_map:
            product_to_photo_map[product]  = set()
        product_to_photo_map[product].add(photo)
    with open(test_data) as jsonFile:
        test_pairs = json.load(jsonFile)
    missing_photo, missing_product, valid_count = 0, 0, 0
    recall_dict = {}
    for k in k_s:
        recall_dict[k] = [0, 0]
    with env.begin() as txn:
        for pair in test_pairs:
            photo = pair["photo"]
            product = pair["product"]
            if not os.path.exists(os.path.join(query_dir, str(photo)+".jpg")):
                missing_photo+=1
                continue
            prod_available = True
            for p in product_to_photo_map[product]:
                if not os.path.exists(os.path.join(image_dir, str(p)+".jpg")):
                    prod_available = False
                    break
            if not prod_available:
                missing_product+=1
                continue
            result = txn.get(str(photo))
            valid_count+=1
            nn = json.loads(txn.get(str(photo)))
            product_nn = []
            for item in nn:
                p = int(item[0])
                prod = photo_to_product_map[p]
                if prod not in product_nn:
                    product_nn.append(prod)
            assert len(product_nn) > k_s[-1]
            for k in k_s:
                if product in product_nn[:k]:
                    recall_dict[k][0]+=1
                recall_dict[k][1]+=1
    print("Missing query %d Missing product set %d Total %d"%(missing_photo, missing_product, valid_count))
    for k in k_s:
        print("Recall at %d is %0.3f "%(k, recall_dict[k][0]*1.0/recall_dict[k][1]))
    return recall_dict



if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        print("Usage is python compute_recall.py <vertical> <path_to_nn_lmdb>")
        sys.exit(1)
    compute_recall(args[0], args[1])