import tqdm
import random
import os

from aistlib.dataset import AISTDataset
from aistlib.dataset_split_setting import get_split
from aistlib.processing import calculate_metrics

motion_dir = '/home/rui/local/data/AIST++/smpl_posenet_2stage_30fps/'
music_dir = '/home/rui/local/data/AIST++/music/'

dataset = AISTDataset(motion_dir, music_dir)
split = get_split(
    dataset.video_names, task='generation', subset='val', is_paired=False)
video_names = split['video_names']
music_names = split['music_names']

music_paths, joints_list, bpm_list = [], [], []
for video_name, music_name in tqdm.tqdm(zip(video_names, music_names)):
    data = dataset.get_item(video_name, only_motion=True)
    music_path = os.path.join(dataset.music_dir, f'{music_name}.wav')
    music_paths.append(music_path)
    joints_list.append(data['motion_data']['smpl_joints'])
    bpm_list.append(data['music_tempo'])

# Under 60FPS. so `tol=12` means 0.2s.
print ('--------- tol=0.2s --------------')
calculate_metrics(
    music_paths, joints_list, bpm_list, tol=6, verbose=True)
