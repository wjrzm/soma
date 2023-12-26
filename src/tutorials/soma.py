import os.path as osp
import numpy as np
from glob import glob


import os.path as osp
from src.soma.train.soma_trainer import create_soma_data_id
from src.soma.run_soma.paper_plots.mosh_soma_dataset import gen_stagei_mocap_fnames
from src.soma.tools.run_soma_multiple import run_soma_on_multiple_settings

soma_expr_id = 'V48_02_SOMA'

soma_data_settings = [(5, 3, 0.0, 1.0), ] # upto 5 occlusions, upto 3 ghost points, 0.0% real data, 100. % synthetic data
soma_work_base_dir = '/mnt/d/Code/SOMA'
support_base_dir = osp.join(soma_work_base_dir, 'support_files')
soma_marker_layout_fname = osp.join(support_base_dir, 'marker_layouts/SOMA/soma_subject1/clap_001.c3d')

num_gpus = 1 # number of gpus for training
num_cpus = 4


soma_data_ids = [create_soma_data_id(*soma_data_setting) for soma_data_setting in soma_data_settings]
print(soma_data_ids)
mocap_base_dir = osp.join(support_base_dir, 'evaluation_mocaps/original')
soma_mocap_target_ds_name = 'SOMA_unlabeled_mpc'

run_soma_on_multiple_settings(
        soma_expr_ids=[soma_expr_id],
        soma_mocap_target_ds_names=[
            'SOMA_unlabeled_mpc',
        ],
        soma_data_ids=soma_data_ids,
        soma_cfg={
            'soma.batch_size': 512,
            'dirs.support_base_dir': support_base_dir,
            'mocap.unit': 'mm',
            'save_c3d': True,
            'keep_nan_points': True,  # required for labeling evaluation
            'remove_zero_trajectories': False  # required for labeling evaluation
        },
        mocap_base_dir=mocap_base_dir,
        run_tasks=['soma'],

        mocap_ext='.c3d',
        soma_work_base_dir = soma_work_base_dir,
        
        parallel_cfg = {
            # 'max_num_jobs': 1, # comment to run on all mocaps
            'randomly_run_jobs': True,
        },
    )