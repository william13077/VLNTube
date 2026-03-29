import numpy as np
import cv2
import os
import json
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from pathfinding.core.grid import Grid
import copy
from vlntube.path_finder import simplify_path_with_collision_check
from vlntube.path_utils import sample_walkable_point_in_polygon, get_path, visualize_and_save_result, densify, smooth_path_spline, smooth_path_average, smooth_path_conditional, simplify_path, find_representative_points
import datetime
import pdb
# --- Helper functions ---
import random
import natsort
random.seed(1024)

# --- 1. Initialization and configuration ---
parser = argparse.ArgumentParser(description='Stage 1: Sample walkable points in each room.')
parser.add_argument('--dataroot', type=str, default='/mnt/6t/dataset/vlnverse/',
                    help='Root directory containing scene folders')
parser.add_argument('--metaroot', type=str, default='/data/lsh/scene_summary/metadata/',
                    help='Root directory containing scene metadata (freemap, room_region, etc.)')
args = parser.parse_args()

SAMPLE_NUM = 5
dataroot = args.dataroot
metaroot = args.metaroot

temp_dir = os.listdir(dataroot)
dir = natsort.natsorted([i for i in temp_dir if os.path.isdir(os.path.join(dataroot,i))])


for scene_id in dir:
    if scene_id not in ['kujiale_0003']:
        continue
    # 2nd version: changed waypoint generation, and the freemap is different
    vis_dir = os.path.join(dataroot,scene_id,'sampled_points_publish')
    if os.path.exists(os.path.join(vis_dir,'sampled_points.json')):
        continue
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # --- 2. Load data ---
    occupancy_path = os.path.join(metaroot,scene_id,'freemap.npy')
    annotation_path = os.path.join(dataroot,scene_id,'room_in_images.json')
    ANNOTATION_PATH = os.path.join(metaroot, scene_id, 'room_region.json')
    occ = np.load(occupancy_path)

    rgb = cv2.imread(os.path.join(dataroot,scene_id,'occupancy.png'))
    try:
        with open(annotation_path,'r') as f: scene_anno = json.load(f)
    except:
        scene_anno = {}
    try:
        with open(ANNOTATION_PATH, 'r') as f: scene_anno_2 = json.load(f)
    except:
        scene_anno_2 = {}

    # --- 3. Prepare walkable map ---
    # Copy map data: mark obstacles as 0, walkable areas as 1
    matrix = copy.copy(occ[1:,1:])
    matrix[matrix==2]=0
    # Dilate obstacles to create safety margins for path planning
    matrix=cv2.dilate(1-matrix,np.ones([4, 4]), iterations=2) # 2nd version
    matrix = 1 - matrix
    grid = Grid(matrix = matrix.tolist())

    # Get start and end regions
    iters = max(len(scene_anno), len(scene_anno_2.keys()))
    for idx in range(iters):
        try:
            name = scene_anno[idx]['room_type']
        except:
            name = list(scene_anno_2.keys())[idx]
        all_points =[]
        try:
            polygon_coords = [ [int(p[0]),int(p[1])] for p in scene_anno[idx]['polygon']]
        except:
            polygon_coords = [(p[1],p[0])  for p in scene_anno_2[name] ]

        # Approximate Monte Carlo sampling
        for i in range(2000):
            start_point = sample_walkable_point_in_polygon(matrix, polygon_coords) #[x,y]
            all_points.append(start_point)

        if None not in all_points:
            spoints,_,_ = find_representative_points(all_points,SAMPLE_NUM)
            spoints = np.unique(spoints,axis=0)
        else:
            spoints = np.array([])
        scene_anno[idx].update({'sampled_points':spoints.tolist()})
        visualize_and_save_result(matrix,polygon_coords,spoints,os.path.join(vis_dir,f'{name}_{idx}.png'))


    with open(os.path.join(vis_dir,'sampled_points.json'), 'w', encoding='utf-8') as json_file:
        print(f'Sampling json for {scene_id}')
        json.dump(scene_anno, json_file, indent=4, ensure_ascii=False)
