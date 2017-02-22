import glob
import json
import os

__author__ = 'ananya.h'


base_dir = "/data/street2shop"
meta_dir = os.path.join(base_dir, "meta", "json")
image_dir = os.path.join(base_dir, "images")
structured_dir = os.path.join(base_dir, "structured_images")
all_pair_file_paths = glob.glob(meta_dir + "/retrieval_*.json")

for path in all_pair_file_paths:
    vertical = path.split("_")[-1].split(".")[0]
    query_dir = os.path.join(structured_dir, vertical+"_query")
    if not os.path.exists(query_dir):
        os.mkdir(query_dir)
    catalog_dir = os.path.join(structured_dir, vertical)
    if not os.path.exists(catalog_dir):
        os.mkdir(catalog_dir)
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    product_ids = set()
    for item in data:
        product_ids.add(item["photo"])
    print("Symlinking catalog ids for %s", vertical)
    for product_id in product_ids:
        img_path = os.path.join(image_dir, str(product_id)+".jpg")
        dst_path = os.path.join(catalog_dir, str(product_id)+".jpg")
        if os.path.exists(img_path):
            os.symlink(img_path, dst_path)
    query_ids = set()
    for partition in ["train", "test"]:
        partition_file = partition+"_pairs_"+vertical+".json"
        with open(os.path.join(meta_dir, partition_file)) as jsonFile:
            pairs = json.load(jsonFile)
        for pair in pairs:
            query_ids.add(pair["photo"])
    print("Symlinking query ids for %s", vertical)
    for query_id in query_ids:
        img_path = os.path.join(image_dir, str(query_id)+".jpg")
        dst_path = os.path.join(query_dir, str(query_id)+".jpg")
        if os.path.exists(img_path):
            os.symlink(img_path, dst_path)