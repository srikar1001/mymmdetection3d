import argparse
from os import path as osp
from tools.data_converter import waymo_converter as waymo
from tools.data_converter.create_gt_database import create_groundtruth_database
from tools.data_converter import kitti_converter as kitti

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

def create_trainvaltestsplitfile(dataset_dir):
    trainingdir = os.path.join(dataset_dir, 'training', 'image_2')
    ImageSetdir = os.path.join(dataset_dir, 'ImageSets')
    if not os.path.exists(ImageSetdir):
        os.makedirs(ImageSetdir)

    images = os.listdir(trainingdir)
    # totalimages=len([img for img in images])
    # print("Total images:", totalimages)
    dataset = []
    for img in images:
        dataset.append(img[:-4])#remove .png
    print("Total images:", len(dataset))
    df = pd.DataFrame(dataset, columns=['index'], dtype=np.int32)
    X_train, X_val = train_test_split(df, train_size=0.8, test_size=0.2, random_state=42)
    print("Train size:", X_train.shape)
    print("Val size:", X_val.shape)
    write_to_file(os.path.join(ImageSetdir, 'trainval.txt'), df.sort_values('index')['index'])
    write_to_file(os.path.join(ImageSetdir, 'train.txt'), X_train.sort_values('index')['index'])
    write_to_file(os.path.join(ImageSetdir, 'val.txt'), X_val.sort_values('index')['index'])

    testdir = os.path.join(dataset_dir, 'test', 'image_2')
    testimages = os.listdir(testdir)
    # totalimages=len([img for img in images])
    # print("Total images:", totalimages)
    testdataset = []
    for img in testimages:
        testdataset.append(img[:-4])#remove .png
    dftest = pd.DataFrame(testdataset, columns=['index'], dtype=np.int32)
    print("Test size:", dftest.shape)
    write_to_file(os.path.join(ImageSetdir, 'test.txt'), dftest.sort_values('index')['index'])


def write_to_file(path, data): 
    file = open(path, 'w') 
    for idx in data: 
        #print(idx)
        file.write(str(idx).zfill(6))
        file.write('\n')

    file.close()
    print('Done in ' + path)

def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int): Number of input consecutive frames. Default: 5 \
            Here we store pose information of these frames for later use.
    """
    from tools.data_converter import waymo_converter as waymo

    #splits = ['training', 'validation', 'testing']
    splits = ['training', 'validation', 'test']
    for i, split in enumerate(splits):
        #load_dir = osp.join(root_path, 'waymo_format', split)
        load_dir = osp.join(root_path, split)
        if split == 'validation':
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split == 'test'))
        converter.convert()
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(out_dir, info_prefix, max_sweeps=max_sweeps)
    create_groundtruth_database(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False)

def createwaymo_info(root_path,
                    info_prefix,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int): Number of input consecutive frames. Default: 5 \
            Here we store pose information of these frames for later use.
    """
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    # Create ImageSets, train test split
    create_trainvaltestsplitfile(out_dir)

    kitti.create_waymo_info_file(out_dir, info_prefix, max_sweeps=max_sweeps)
    create_groundtruth_database(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False)

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='waymo', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='/data/cmpe249-f20/Waymo',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='/data/cmpe249-f20',
    required='False',
    help='name of info pkl')
parser.add_argument(
        '--createsplitfile_only',
        action='store_true',
        help='create train val split files')
parser.add_argument(
        '--createinfo_only',
        action='store_true',
        help='create info files')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=2, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    print("Dataset:", args.dataset)
    if args.createsplitfile_only:
        create_trainvaltestsplitfile(args.out_dir)
    else:
        createwaymo_info(args.root_path, args.extra_tag, args.out_dir, args.workers)
        
    # if args.dataset == 'waymo':
    #     if args.createinfo_only:
    #         else:
    #         waymo_data_prep(
    #             root_path=args.root_path,
    #             info_prefix=args.extra_tag,
    #             version=args.version,
    #             out_dir=args.out_dir,
    #             workers=args.workers,
    #             max_sweeps=args.max_sweeps)