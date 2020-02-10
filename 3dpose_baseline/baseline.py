## Baseline algorithm

!git clone https://github.com/ArashHosseini/3d-pose-baseline.git
!cd 3d-pose-baseline/ && mkdir data && cd data && wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip && unzip h36m.zip && rm h36m.zip
!cp drive/'My Drive'/experiments.tar.gz /content
!tar -zxvf experiments.tar.gz && mv experiments 3d-pose-baseline/
!unzip drive/'My Drive'/SfV_data.zip -d /content/
!unzip drive/'My Drive'/handtool_videos_minimal.zip  -d /content/

### 3D pose baseline
import os
import glob

files = glob.glob('/content/SfV_data/params/*')
files = sorted([f for f in files if'pkl' not in f])[:1]
for file_ in files :
  print(file_)
  cmd = 'cd 3d-pose-baseline && python src/openpose_3dpose_sandbox.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 200 --load 4874200 --pose_estimation_json ' + file_ +' --write_gif --gif_fps 24'
  res = os.system(cmd)
  if res > 0:
    print('PROBLEEEMMMM')
    pass
