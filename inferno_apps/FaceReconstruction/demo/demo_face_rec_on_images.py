"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

from inferno_apps.FaceReconstruction.utils.load import load_model
from inferno.datasets.ImageTestDataset import TestData
import inferno
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from inferno_apps.FaceReconstruction.utils.output import save_obj, save_images, save_codes
from inferno_apps.FaceReconstruction.utils.test import test
from inferno.utils.other import get_path_to_assets


def main():
    parser = argparse.ArgumentParser()
    # add the input folder arg 
    parser.add_argument('--input_folder', type=str, default= "/home/tien/inferno/assets/" +"data/EMOCA_test_example_data/images/affectnet_test_examples")
    parser.add_argument('--output_folder', type=str, default="image_output", help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMICA-CVT_flame2020_notexture', help='Name of the model to use.')
    # parser.add_argument('--model_name', type=str, default='EMICA_flame2020_notexture', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str, default="/home/tien/inferno/assets/" + "FaceReconstruction/models")
    parser.add_argument('--save_images', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=False, help="If true, output meshes will be saved")
    
    args = parser.parse_args()


    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    path_to_models = args.path_to_models
    input_folder = args.input_folder
    output_folder = args.output_folder
    model_name = args.model_name

    # 1) Load the model
    face_rec_model, conf = load_model(path_to_models, model_name)
    face_rec_model.cuda()
    face_rec_model.eval()

    # 2) Create a dataset
    dataset = TestData(input_folder, face_detector="fan", max_detection=20)

    ## 4) Run the model on the data
    for i in auto.tqdm( range(len(dataset))):
        batch = dataset[i]
        vals = test(face_rec_model, batch)
        visdict = face_rec_model.visualize_batch(batch, i, None, in_batch_idx=None)
        # name = f"{i:02d}"
        current_bs = batch["image"].shape[0]

        for j in range(current_bs):
            name =  batch["image_name"][j]

            sample_output_folder = Path(output_folder) / name
            sample_output_folder.mkdir(parents=True, exist_ok=True)

            if args.save_mesh:
                save_obj(face_rec_model, str(sample_output_folder / "mesh_coarse.obj"), vals, j)
            if args.save_codes:
                save_codes(Path(output_folder), name, vals, i=j)
            if args.save_images:
                save_images(output_folder, name, visdict, with_detection=True, i=j)
 

    print("Done")


if __name__ == '__main__':
    main()
