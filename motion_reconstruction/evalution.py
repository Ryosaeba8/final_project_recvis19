'''
This script is dedicated to RecVis18 student project - Topic D and Topic M.
For any questions please contact: zongmian.li@inria.fr
------
Evaluating mean per joint position error (MPJPE) on the Handtools dataset.

To do the job, please copy the following functions to your project:
- getJ3dPosFromXML
- procrustes
- compute_euclidean_dist

You can evaluate the predicted 3D pose predPos (njoints x 3) in this way:
GTPos = getJ3dPosFromXML(XMLPath, nameDict=None) # nameDict is the joint labels defined in the xml, no need to change it here
R, t, s, predPos = procrustes(predPos, GTPos) # solving Procrustes problem using SVD
mpjpe = compute_euclidean_dist(predPos, GTPos) # report the mean distance error
'''

import numpy as np
import numpy.linalg as LA
import xml.etree.ElementTree as ET
#from smpl_webuser.serialization import load_model
#from smpl_webuser.lbs import global_rigid_transformation

def procrustes(A, B):
    '''
    Solves the orthogonal Procrustes problem given a set of 3D points A (3 x N)
    and a set of target 3D points B (3 x N). Namely, it computes a group of
    R(otation), t(ranslation) and s(cale) that aligns A with B.
    '''
    # input check
    transposed = False
    if A.shape[0]!=3:
        A = A.T
        B = B.T
        transposed = True
    N = A.shape[1]
    assert(B.shape==(3,N))
    # compute mean
    a_bar = A.mean(axis=1, keepdims=True)
    b_bar = B.mean(axis=1, keepdims=True)
    # calculate rotation
    A_c = A - a_bar
    B_c = B - b_bar
    M = A_c.dot(B_c.T)
    U, Sigma, Vh = LA.svd(M)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1,-1] = LA.det(V)*LA.det(U)
    R = V.dot(Z.dot(U.T))
    # compute scale
    s = np.trace(R.dot(M)) / np.trace(A_c.T.dot(A_c))
    # compute translation
    t = b_bar - s*(R.dot(a_bar))
    # compute A after alignment
    A_hat = s*(R.dot(A)) + t
    if transposed:
        A_hat = A_hat.T
    return (R, t, s, A_hat)

def compute_euclidean_dist(S1,S2,debug=False):
    #print S1.shape, S2.shape
    assert(S2.shape == S1.shape)
    assert(S1.shape[1] == 3) # S1 and S2 should be of shape njoints x 3
    distance = 0.
    for i in range(S1.shape[0]):
        distance += LA.norm(S1[i]-S2[i])
        if debug:
            print ('joint #{0}, distance {1}'.format(i,LA.norm(S1[i]-S2[i])))
    distance /= S1.shape[0]
    return distance

def getJ3dPosFromXML(XMLPath, nameDict=None):
    if nameDict is None:
        nameDict = {'R_Ankle':0,
                    'R_Knee':1,
                    'R_Hip':2,
                    'L_Hip':3,
                    'L_Knee':4,
                    'L_Ankle':5,
                    'R_Wrist':6,
                    'R_Elbow':7,
                    'R_Shoulder':8,
                    'L_Shoulder':9,
                    'L_Elbow':10,
                    'L_Wrist':11}
    annotation = ET.parse(XMLPath).getroot()
    keypoints = annotation.find('keypoints')
    GTPos = np.zeros((12,3))
    for keypoint in keypoints.findall('keypoint'):
        name = keypoint.get('name')
        x = float(keypoint.get('x'))
        y = float(keypoint.get('y'))
        # pay attention: convert to right hand coordinate frame by multiplying -1
        z = -1.*float(keypoint.get('z'))
        if name in nameDict.keys():
            GTPos[nameDict[name]] = np.array([x,y,z])
    return GTPos
import glob
import pickle, json
import pandas as pd
path_pred = 'SfV_data/out_videos'
path_truth = 'ground_truth_3d_poses'
path_other = '3d-pose-baseline/maya/3d_'
joints_index =['3', '2', '1', '4', '5', '6', '17', '16', '15', '12', '13', '14']
files = sorted(glob.glob(path_pred + '/*.pkl' ))
df = pd.DataFrame()
for file in files :
    name = file.split('/')[-1][:-4]
    infile = open(file, 'rb')
    result_dict = pickle.load(infile)
    infile.close()    
    XMLPath = path_truth + '/' +  name + '/info/'
    frames = glob.glob(XMLPath + '*') 
    mean = []
    mean_2 = []
    file_other = path_other + name + '.json'
    try :
      infile = open(file_other, 'rb')
      result_other = json.load(infile)
      infile.close() 
    except :
      pass
    for frame in frames :
      ind = int(frame.split('_0.xml')[0][-3:])
      _3d_poses = np.zeros((len(joints_index), 3))
      for indi, i in enumerate(joints_index) :
        _3d_poses[indi] = result_other[str(indi)][str(i)]['translate']      
      predPos = result_dict[ind][0]['joints3d'][0:12]
      GTPos = getJ3dPosFromXML(frame, nameDict=None)
      R, t, s, predPos = procrustes(predPos, GTPos)
      mean.append(compute_euclidean_dist(predPos, GTPos)) #+ '_frame_' + str(ind)
      R, t, s, predPos = procrustes(_3d_poses, GTPos)
      mean_2.append(compute_euclidean_dist(predPos, GTPos))
    df = df.append({'frame' :  name , 'joints error hmr' : np.mean(mean), 'joints error baseline' : np.mean(mean_2)}, ignore_index = True)
