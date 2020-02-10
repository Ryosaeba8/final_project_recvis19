%%writefile /content/motion_reconstruction/ref_2.py
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
from imageio import imwrite
import sys
import pickle
sys.path.insert(0, '/content/motion_reconstruction/')
sys.path.insert(0, '/content/motion_reconstruction/src')
sys.path.insert(0,'/content/smpl')
from smpl_webuser.serialization import load_model
from src.config import get_config
from src.util.video import read_data, collect_frames
from src.util.renderer import SMPLRenderer, draw_openpose_skeleton, render_original
from src.refiner import Refiner
# from jason.bvh_core import write2bvh


# Defaults:
kVidDir = '/content/SfV_data/original_video'
# Where the smoothed results will be stored.
kOutDir = '/content/SfV_data/'
# Holds h5 for each video, which stores OP outputs, after trajectory assignment.
kOpDir = kOutDir


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
    print(out_res_path)
    infile = open(out_res_path, 'rb')
    result_dict = pickle.load(infile)
    infile.close()    
    temp_dir = tempfile.mkdtemp(dir='/content/')
    print('writing to %s' % temp_dir)

    used_frames = frames[start_fr:end_fr + 1]
    for i, (frame, proc_param) in enumerate(zip(used_frames, proc_params)):
        if i % 10 == 0:
            print('%d/%d' % (i, len(used_frames)))

        result_here = result_dict[i][0]

        # Render each frame.
        if RENDONLY and 'only' in REND_TYPE:
            rend_frame = np.ones_like(frame)
            skel_frame = np.ones_like(frame) * 255
            op_frame = np.ones_like(frame) * 255
        else:
            rend_frame = frame.copy()
            skel_frame = frame.copy()
            op_frame = frame.copy()

        op_frame = cv2.putText(
            op_frame.copy(),
            'OpenPose Output', (10, 50),
            0,
            1,
            (0, 0, 0),
            thickness=3)
        other_vp = np.ones_like(frame)
        other_vp2 = np.ones_like(frame)

        op_kp = result_here['op_kp']
        bbox = result_here['proc_param']['bbox']
        op_frame = draw_openpose_skeleton(op_frame, op_kp)

        if not RENDONLY or (RENDONLY and 'op' not in REND_TYPE):
            rend_frame, skel_frame, other_vp, other_vp2 = render_original(
                rend_frame,
                skel_frame,
                proc_param,
                result_here,
                other_vp,
                other_vp2,
                bbox,
                renderer)
            row1 = np.hstack((frame, skel_frame, np.ones_like(op_frame) * 255))
            row2 = np.hstack((rend_frame, other_vp2[:, :, :3], op_frame))
            final_rend_img = np.vstack((row2, row1)).astype(np.uint8)

        if RENDONLY:
            if 'mesh' in REND_TYPE:
                final_rend_img = rend_frame.astype(np.uint8)
            elif 'op' in REND_TYPE:
                final_rend_img = op_frame.astype(np.uint8)
            else:
                final_rend_img = skel_frame.astype(np.uint8)

        import matplotlib.pyplot as plt
        #plt.ion()
        #plt.figure(1)
        #plt.clf()
        #plt.imshow(final_rend_img)
        #plt.title('%d/%d' % (i, len(used_frames)))
        #plt.pause(1e-3)

        out_name = join(temp_dir, 'frame%03d.png' % i)
        imwrite(out_name, final_rend_img)

    # Write video.
    cmd = 'ffmpeg -y -threads 16  -i %s/frame%%03d.png -profile:v baseline -level 3.0 -c:v libx264 -pix_fmt yuv420p -an -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" %s' % (
        temp_dir, out_mov_path.split('.')[0] + '.mp4')
    system(cmd)
    #shutil.rmtree(temp_dir)

def main(config):
    np.random.seed(5)
    video_paths = sorted(glob(join(config.vid_dir, "")))
    pred_dir = kOutDir

    for i, vid_path in enumerate(video_paths):
        #out_mov_path = join(pred_dir, basename(vid_path).replace('.mp4', '.pkl'))
        out_mov_path = join(pred_dir, basename(vid_path) + '.pkl')
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

    if not exists(config.out_dir):
        makedirs(config.out_dir)

    # For visualization.
    renderer = SMPLRenderer(img_size=config.img_size, flength=1000.,
                            face_path=config.smpl_face_path)

    main(config)

