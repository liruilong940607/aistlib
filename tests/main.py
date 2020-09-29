import aistlib.dataset

motion_dir = '/home/rui/local/data/AIST++/smpl_posenet_2stage_30fps/'
music_dir = '/home/rui/local/data/AIST++/music/'

dataset = aistlib.dataset.AISTDataset(motion_dir, music_dir)
motion_data, music_data = dataset.get_item(dataset.video_names[0], verbose=True)
# aistlib.dataset.AISTDataset.plot_music(music_data)
aistlib.dataset.AISTDataset.plot_smpl_poses(
    motion_data['smpl_poses'], motion_data['smpl_trans'])