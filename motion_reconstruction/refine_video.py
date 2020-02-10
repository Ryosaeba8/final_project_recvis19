%%writefile /content/motion_reconstruction/refine_video.py
"""
Driver to fine-tune detection using time + openpose 2D keypoints.

For preprocessing, run run_openpose.py which will compute bbox trajectories.

Assumes there's only one person in the video FTM.
Also assumes that the person is visible for a contiguous duration.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2
from absl import flags
import numpy as np
from os.path import exists, join, basename, dirname
from os import makedirs
import tempfile
import shutil
from os import system
from glob import glob
import deepdish as dd
from imageio import imwrite
import sys
import pickle
sys.path.insert(0, '/content/motion_reconstruction/')
sys.path.insert(0, '/content/motion_reconstruction/src')
sys.path.insert(0,'/content/smpl')
import tensorflow as tf
from smpl_webuser.serialization import load_model
from src.config import get_config
from src.util.video import read_data, collect_frames
from src.util.renderer import SMPLRenderer, draw_openpose_skeleton, render_original
from src.refiner import Refiner
# from jason.bvh_core import write2bvh


# Defaults:
kVidDir = '/content/SfV_data/original_video'
# Where the smoothed results will be stored.
kOutDir = '/content/SfV_data/out_videos'
# Holds h5 for each video, which stores OP outputs, after trajectory assignment.
kOpDir = '/content/SfV_data/params'


kMaxLength = 1000
kVisThr = 0.2
RENDONLY = False

# set if you only want to render specific renders.
flags.DEFINE_string('render_only', '', 'If not empty and are either {mesh, mesh_only}, only renders that result.')
flags.DEFINE_string('vid_dir', kVidDir, 'directory with videso')
flags.DEFINE_string('out_dir', kOutDir, 'directory to output results')
flags.DEFINE_string('op_dir', kOpDir,
                    'directory where openpose output is')


model = None
sess = None


def run_video(frames, per_frame_people, config, out_mov_path):
    """
    1. Extract all frames, preprocess it
    2. Send it to refiner, get 3D pose back.

    Render results.
    """
    proc_imgs, proc_kps, proc_params, start_fr, end_fr = collect_frames(
        frames, per_frame_people, config.img_size, vis_thresh=kVisThr)

    num_frames = len(proc_imgs)

    proc_imgs = np.vstack(proc_imgs)

    out_res_path = out_mov_path

    if not exists(out_res_path) or config.viz:
        # Run HMR + refinement.
        tf.reset_default_graph()
        model = Refiner(config, num_frames)
        scale_factors = [np.mean(pp['scale'])for pp in proc_params]
        offsets = np.vstack([pp['start_pt'] for pp in proc_params])
        results = model.predict(proc_imgs, proc_kps, scale_factors, offsets)

        # Pack proc_param into result.
        results['proc_params'] = proc_params

        import pandas as pd
        joints3d = results['joints3d']
        joints_names = ['Ankle.R_x', 'Ankle.R_y', 'Ankle.R_z',
                          'Knee.R_x', 'Knee.R_y', 'Knee.R_z',
                          'Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                          'Hip.L_x', 'Hip.L_y', 'Hip.L_z',
                          'Knee.L_x', 'Knee.L_y', 'Knee.L_z', 
                          'Ankle.L_x', 'Ankle.L_y', 'Ankle.L_z',
                          'Wrist.R_x', 'Wrist.R_y', 'Wrist.R_z', 
                          'Elbow.R_x', 'Elbow.R_y', 'Elbow.R_z', 
                          'Shoulder.R_x', 'Shoulder.R_y', 'Shoulder.R_z', 
                          'Shoulder.L_x', 'Shoulder.L_y', 'Shoulder.L_z',
                          'Elbow.L_x', 'Elbow.L_y', 'Elbow.L_z',
                          'Wrist.L_x', 'Wrist.L_y', 'Wrist.L_z', 
                          'Neck_x', 'Neck_y', 'Neck_z', 
                          'Head_x', 'Head_y', 'Head_z', 
                          'Nose_x', 'Nose_y', 'Nose_z', 
                          'Eye.L_x', 'Eye.L_y', 'Eye.L_z', 
                          'Eye.R_x', 'Eye.R_y', 'Eye.R_z', 
                          'Ear.L_x', 'Ear.L_y', 'Ear.L_z', 
                          'Ear.R_x', 'Ear.R_y', 'Ear.R_z']
        # Pack results:
        result_dict = {}
        used_frames = frames[start_fr:end_fr + 1]
        for i, (frame, proc_param) in enumerate(zip(used_frames, proc_params)):
            bbox = proc_param['bbox']
            op_kp = proc_param['op_kp']

            # Recover verts from SMPL params.
            theta = results['theta'][i]
            pose = theta[3:3+72]
            shape = theta[3+72:]
            smpl.trans[:] = 0.
            smpl.betas[:] = shape
            smpl.pose[:] = pose
            verts = smpl.r
            result_here = {
                'theta': np.expand_dims(theta, 0),
                'joints': np.expand_dims(results['joints'][i], 0),
                'cams': results['cams'][i],
                'joints3d': results['joints3d'][i],
                'verts': verts,
                'op_kp': op_kp,
                'proc_param': proc_param
            }
            result_dict[i] = [result_here]

        # Save results & write bvh.
        outfile = open(out_res_path,'wb')
        pickle.dump(result_dict, outfile, protocol=2)
        outfile.close()
    return None

def main(config):
    np.random.seed(5)
    video_paths = sorted(glob(config.vid_dir +  "/*"))
    pred_dir = kOutDir
    #import pdb; pdb.set_trace()
    for i, vid_path in enumerate(video_paths):
        vid_path_ = vid_path + '/frames/'
        #out_mov_path = join(pred_dir, basename(vid_path).replace('.mp4', '.pkl'))
        out_mov_path = join(pred_dir, basename(vid_path) + '.pkl')
        if not exists(out_mov_path) or config.viz:
            print('working on %s' % basename(vid_path))
            frames, per_frame_people, valid = read_data(vid_path, config.op_dir, max_length=kMaxLength)
            if valid:
                run_video(frames, per_frame_people, config, out_mov_path)

    print('Finished writing to %s' % pred_dir)


if __name__ == '__main__':
    config = get_config()

    if len(config.render_only) > 0:
        RENDONLY = True
        REND_TYPE = config.render_only
        rend_types = ['mesh', 'mesh_only', 'op', 'op_only']
        if not np.any(np.array([REND_TYPE == rend_t for rend_t in rend_types])):
            print('Unknown rend type %s!' % REND_TYPE)
            import pdb; pdb.set_trace()

    if not config.load_path:
        raise Exception('Must specify a model to use to predict!')
    if 'model.ckpt' not in config.load_path:
        raise Exception('Must specify a model checkpoint!')

    if not exists(config.out_dir):
        makedirs(config.out_dir)

    # For visualization.
    renderer = SMPLRenderer(img_size=config.img_size, flength=1000.,
                            face_path=config.smpl_face_path)

    smpl = load_model(config.smpl_model_path)

    main(config)
