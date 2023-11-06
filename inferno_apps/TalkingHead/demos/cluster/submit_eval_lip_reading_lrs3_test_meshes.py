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

from inferno.utils.condor import execute_on_cluster
from pathlib import Path
import inferno_apps.TalkingHead.evaluation.evaluate_lip_reading_on_lrs3_meshes as script
import datetime
from omegaconf import OmegaConf
import time as t
import random
from omegaconf import DictConfig, OmegaConf, open_dict 
import sys
import shutil

# submit_ = False
submit_ = True

# if submit_:
#     config_path = Path(__file__).parent / "submission_settings.yaml"
#     if not config_path.exists():
#         cfg = DictConfig({})
#         cfg.cluster_repo_path = "todo"
#         cfg.submission_dir_local_mount = "todo"
#         cfg.submission_dir_cluster_side = "todo"
#         cfg.python_bin = "todo"
#         cfg.username = "todo"
#         OmegaConf.save(config=cfg, f=config_path)
        
#     user_config = OmegaConf.load(config_path)
#     for key, value in user_config.items():
#         if value == 'todo': 
#             print("Please fill in the settings.yaml file")
#             sys.exit(0)


def submit(resume_folder, subset, 
           bid=10, 
           max_price=None,
           ):
    cluster_repo_path = "/home/rdanecek/workspace/repos/inferno"
    submission_dir_local_mount = "/is/cluster/work/rdanecek/talking_head_eval/submission_lrs3_lipread"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/talking_head_eval/submission_lrs3_lipread"

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(time)) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(script.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[2].name / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    python_bin = 'python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = 30 * 1024
    gpu_mem_requirement_mb_max = 40000
    # gpu_mem_requirement_mb = None
    cpus = 8 #cfg.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = 1
    num_jobs = 1
    max_time_h = 36
    job_name = "train_talking_head"
    cuda_capability_requirement = 7
    mem_gb = 20

    args = f" {resume_folder} {str(subset)} "

    #    env="work38",
    #    env="work38_clone",
    # env = "/is/cluster/fast/rdanecek/envs/work38_fast" 
    env = "/is/cluster/fast/rdanecek/envs/work38_fast_clone"
    # if env is an absolute path to the conda environment
    if Path(env).exists() and Path(env).resolve() == Path(env):
        python_bin = str(Path(env) / "bin/python")
        assert Path(python_bin).exists(), f"Python binary {python_bin} does not exist"

    execute_on_cluster(str(cluster_script_path),
                       args,
                       str(submission_folder_local),
                       str(submission_folder_cluster),
                       str(cluster_repo_path),
                       python_bin=python_bin,
                       username=username,
                       gpu_mem_requirement_mb=gpu_mem_requirement_mb,
                       gpu_mem_requirement_mb_max=gpu_mem_requirement_mb_max,
                       cpus=cpus,
                       mem_gb=mem_gb,
                       gpus=gpus,
                       num_jobs=num_jobs,
                       bid=bid,
                       max_time_h=max_time_h,
                       max_price=max_price,
                       job_name=job_name,
                       cuda_capability_requirement=cuda_capability_requirement,
                       env=env, 
                       )
    t.sleep(1)



def run_talking_head_eval():    
    resume_folders = []

  

    # bid = 2000
    # bid = 150
    bid = 28
    # bid = 100
    # max_price = 250
    # max_price = 200
    max_price = 500

    subfolders = ['sub1', 'sub2', 'sub3']

    resume_folders = []
    resume_folders += ["CT_lrs3_test"]
    resume_folders += ["FF_lrs3_test"]
    resume_folders += ["VOCA_lrs3_test"]


    for subfolder in subfolders:

        for resume_folder in resume_folders:
            if submit_:
                submit(resume_folder, subfolder, bid=bid, max_price=max_price)
            else: 
                script.evaluate_model(resume_folder, subfolder)



if __name__ == "__main__":
    run_talking_head_eval()
