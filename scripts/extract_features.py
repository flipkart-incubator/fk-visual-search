import glob
import os
import sys
from scripts.indexer import Indexer

__author__ = 'ananya.h'



if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        print("Requires parameters vertical")
        sys.exit(1)
    vertical = args[0]
    base_dir = "/data/snapshots/civr_wivr_91203"
    lmdb_dir = "/data/street2shop/lmdbs/"
    base_image_dir = "/data/street2shop/structured_images/"
    config = {}
    config["layer"] = "linear_embedding_q_norm"
    config["input_layer"] = "data_q"
    config["fv_db_path"] = os.path.join(lmdb_dir, vertical+"_civr_wivr_91203_543750")
    config["path_to_deploy_file"] = os.path.join(base_dir, "vgg16_triplet_shallow_deploy.prototxt")
    config["path_to_model_file"] = os.path.join(base_dir, "_iter_543750.caffemodel.h5")
    imdir = os.path.join(base_image_dir, vertical+"_256")
    if not os.path.exists(imdir):
        imdir = os.path.join(base_image_dir, vertical)
    image_paths = glob.glob(imdir+"/*.jpg")
    indexer = Indexer(config, image_paths)
    indexer.index(20)