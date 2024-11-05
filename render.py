import argparse
from glob import glob
import cv2
import numpy as np
import torch
from inferno_apps.FaceReconstruction.utils.load import load_model
from inferno.datasets.ImageTestDataset import TestData
import json
from tqdm import tqdm
import os
import torchvision
from skimage.io import imsave
import matplotlib.pyplot as plt


path_to_model = "/home/tien/inferno/assets/" + "FaceReconstruction/models"
model_name = "EMICA-CVT_flame2020_notexture"
face_rec_model, conf = load_model(path_to_model, model_name)
face_rec_model.cuda()
face_rec_model.eval()

from pathlib import Path
from einops import rearrange, reduce, repeat
from skimage.io import imsave
import inferno.utils.DecaUtils as util

import matplotlib.pyplot as plt

# to reconstruct with the output of diffusion we need a few steps

def load_the_defauld_face(path_to_dir):
    cam = torch.load(os.path.join(path_to_dir, "cam.pt"))
    globalpose = torch.load(os.path.join(path_to_dir, "globalpose.pt"))
    lightcode = torch.load(os.path.join(path_to_dir, "lightcode.pt"))
    texcode = torch.load(os.path.join(path_to_dir, "texcode.pt"))
    shapecode = torch.load(os.path.join(path_to_dir, "shapecode.pt"))
    
    return (cam, globalpose, lightcode, texcode, shapecode, )

def read_the_output_tensor(path_to_tensor):
    if isinstance(path_to_tensor, str):
        return torch.load(path_to_tensor, map_location="cpu")
    elif isinstance(path_to_tensor, torch.Tensor):
        return path_to_tensor

def build_the_batch(face_shape_path, prediction_tensor):
    
    expression_and_jawpose = read_the_output_tensor(prediction_tensor)
    
    expression = expression_and_jawpose[:, :, 3:103]
    expression = reduce(expression, "b c d -> b d", "mean")
    jaw_pose = expression_and_jawpose[:, :, :3]
    jaw_pose = reduce(jaw_pose, "b c d -> b d", "mean")
    
    defaul_face = load_the_defauld_face(face_shape_path)
    cam, globalpose, lightcode, texcode, shapecode = map(lambda x: torch.mean(x, dim = 0), defaul_face)
    cam, globalpose, lightcode, texcode, shapecode = map(lambda x: repeat(x, "d -> c d",c=jaw_pose.shape[0]), (cam, globalpose, lightcode, texcode, shapecode))
    
    image = torch.rand(jaw_pose.shape[0], 3, 224, 224)
    
    batch = {
        "image": image,
        "cam": cam,
        "globalpose": globalpose,
        "lightcode": lightcode,
        "texcode": texcode,
        "shapecode": shapecode,
        "expcode": expression,
        "jawpose": jaw_pose
    }
    
    for key in batch:
        batch[key] = batch[key].cuda()
        
    return batch

def save_obj(face_rec_model, filename, opdict, i=0):
    # dense_template_path = Path(inferno.__file__).parents[1] / 'assets' / "DECA" / "data" / 'texture_data_256.npy'
    vertices = opdict['verts'][i].detach().cpu().numpy()
    faces = face_rec_model.renderer.render.faces[0].detach().cpu().numpy()
    uvcoords = face_rec_model.renderer.render.raw_uvcoords[0].detach().cpu().numpy()
    uvfaces = face_rec_model.renderer.render.uvfaces[0].detach().cpu().numpy()
    # save coarse mesh, with texture and normal map
    util.write_obj(filename, vertices, faces,
                #    texture=texture,
                   uvcoords=uvcoords,
                   uvfaces=uvfaces,
                #    normal_map=normal_map
                   )
    
# def split_to_smaller_batch_fitting_mem(batch, size):
#     for i in range(0, len(batch), size):
#         yield {key: batch[key][i:i+size] for key in batch}

import time
def decode_latent_to_image(face_shape_path, prediction_tensor_path, model, output_folder, name):
    big_batch = build_the_batch(face_shape_path, prediction_tensor_path)
    # print(batch["expcode"].shape)
    start = time.time()
    
    for i in range(0, big_batch["expcode"].shape[0], 16):
        
        batch = {key: big_batch[key][i:i+16] for key in big_batch}
    
        batch = model.decode_latents(batch, training = False, validation = False, ring_size = 1)
        
        visdict = face_rec_model.visualize_batch(batch, 0, None, in_batch_idx=None)
        # render_time = time.time() - start
        # print(f"Render time: {render_time}")
        # print(f"Render time per image: {render_time / batch['expcode'].shape[0]}")
        # print(f"fpd: {1 / (render_time / batch['expcode'].shape[0])}")
        
        current_bs = batch["expcode"].shape[0]
        imgs = []
        for j in range(current_bs):
            # name =  batch["image_name"][j]

            # sample_output_folder = Path(output_folder) / name
            # sample_output_folder.mkdir(parents=True, exist_ok=True)

            # save_obj(face_rec_model, str(sample_output_folder / "mesh_coarse.obj"), vals, j)

            # save_images(output_folder, name, visdict, with_detection=True, i=j)
            
            # matplot show image
            # plt.imshow(visdict['shape_image'][j])
            # plt.show()
            
            img = visdict['shape_image'][j]

            # update vertical line
            update()
            
            fig_temp.canvas.draw()
            img_plot_temp = np.array(fig_temp.canvas.renderer.buffer_rgba())
            img_plot_temp = cv2.cvtColor(img_plot_temp, cv2.COLOR_RGBA2RGB)
            img_plot_temp = cv2.resize(img_plot_temp, (500, 500))
            img = cv2.resize(img, (500, 500))
            
            img = np.concatenate([img, img_plot_temp], axis = 1)
            
            # img = rearrange(img, "h w c -> c h w")
            # imgs.append(img)
            imsave(os.path.join(output_dir, basename.split(".")[0], f"{i + j}.jpg"), img)
    return imgs

if __name__ == "__main__":

    # output_dir = "/media/tien/SSD-NOT-OS/pain_intermediate_data/output/try_1/sample"
    defaut_face_path = "/home/tien/inferno/default_face"
    
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--defaut_face_path", type=str, required=False, default=defaut_face_path)
    parser.add_argument("--output_dir", type=str, required=True)
    
    arg = parser.parse_args()
    
    input_path = arg.input_path
    defaut_face_path = arg.defaut_face_path
    output_dir = arg.output_dir
    
    # input_path = "/media/tien/SSD-NOT-OS/pain_intermediate_data/output/try_warmup/sample/54720_1.pt"
    # defaut_face_path = "/home/tien/inferno/default_face"
    # output_path = "out"
    
    ctrl_path = input_path[:-3] + "_ctrl.pt"
    
    ctrl = torch.load(ctrl_path, map_location = "cpu")
    
    temp = ctrl[2][0].cpu()
    x = torch.arange(0, temp.shape[0])
    
    fig_temp = plt.figure()
    plt.plot(x, temp)

    current_time = 0
    
    vertical_line = plt.axvline(x = 0, color = 'b', label = 'axvline - full height')
    
    def update():
        global current_time
        vertical_line.set_xdata(current_time)
        current_time += 1
        # return vertical_line,
    
    # temp = reduce(temp, "b -> b d", "mean")

    rendered_images = []

    import os

    # input tensor is "{name}.pt"
    # for input_tensor in os.listdir(input_dir):
    
    basename = os.path.basename(input_path)
        
    os.makedirs(os.path.join(output_dir, basename.split(".")[0]), exist_ok=True)
    
    decode_latent_to_image(defaut_face_path, input_path, face_rec_model, output_dir, basename)

    # for idx, img in enumerate(rendered_images):
    #     # torchvision.utils.save_image(img, os.path.join(output_path, basename.split(".")[0], f"{idx}.jpg"))
    #     imsave(os.path.join(output_dir, basename.split(".")[0], f"{idx}.jpg"), img)
        # rendered_images = []

    os.system(f"ffmpeg -r 25 -i  {output_dir}/{basename.split('.')[0]}/%d.jpg  -vcodec mpeg4 -b:v 10M -y {output_dir}/{basename.split('.')[0]}.mp4")
    
    # plot the image to a grid using torchvison
    # grid = torchvision.utils.make_grid(torch.stack(rendered_images), nrow=10, normalize=True, scale_each=True)

    # save the grid
    # torchvision.utils.save_image(grid, "sample.png")

# DONE: parameterize this to be callable from the command line with argparse from reconstruction evaluation of diffusion module