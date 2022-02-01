import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    random.seed(2021)
    
    # Get dataset
    ds_size = len(glob.glob(data_dir + '/preprocessed_data/*.tfrecord'))
    
    # create the directry
    for _dir in ["train", "val", "test"]:
        os.makedirs(data_dir+_dir, exist_ok=True)
        
    train_size = int(ds_size * (train_size/100))
    test_size = int(ds_size * (test_size/100))
    val_size = int(ds_size * (val_size/100))
    
    dic = {'train':train_size,
           'test':test_size,
           'val':val_size}

    for split_type,size in dic.items() :
        dataset = os.listdir(data_dir + '/preprocessed_data/')
        
        batch_selected = random.sample(dataset, size)
        for batch in batch_selected:
            shutil.move(data_dir + '/preprocessed_data/'+batch,data_dir + f'/{split_type}/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)