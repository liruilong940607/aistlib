import os
import matplotlib.pyplot as plt

from . import loader
from . import dataset_split_setting
from . import processing
from . import plot_motion
from . import fk

_DIR = os.path.dirname(__file__)

class AISTDataset(object):
    """A dataset class for loading, processing and plotting AIST++"""
    def __init__(self, motion_dir, music_dir):
        self.motion_dir = motion_dir
        self.music_dir = music_dir
        # video names
        with open(os.path.join(_DIR, 'video_list.txt'), 'r') as f:
            self.video_names = [l.strip() for l in f.readlines()]
        # music names
        with open(os.path.join(_DIR, 'music_list.txt'), 'r') as f:
            self.music_names = [l.strip() for l in f.readlines()] 
        # ignore video names: those videos have problematic 3D keypoints.
        with open(os.path.join(_DIR, 'ignore_list.txt'), 'r') as f:
            ignore_video_names = [l.strip() for l in f.readlines()]
            self.video_names = [name for name in self.video_names 
                                if name not in ignore_video_names]
        self.video_names = sorted(self.video_names)
        self.fk_engine = fk.SMPLForwardKinematics()
        
    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        return self.get_item(video_name)

    def get_item(self, video_name, only_motion=False, verbose=False):
        music_name = loader.get_music_name(video_name)
        music_tempo = loader.get_tempo(music_name)

        motion_path = os.path.join(self.motion_dir, f'{video_name}.pkl')
        music_path = os.path.join(self.music_dir, f'{music_name}.wav')
        
        motion_data = loader.load_pkl(
            motion_path, 
            keys=['smpl_poses', 'smpl_scaling', 'smpl_trans'])
        motion_data['smpl_trans'] /= motion_data['smpl_scaling'] 
        del motion_data['smpl_scaling']

        motion_data['smpl_joints'] = self.fk_engine.from_aa(
            motion_data['smpl_poses'], motion_data['smpl_trans'])
        
        for k, v in motion_data.items(): 
            # interpolate 30 FPS motion data into 60 FPS
            motion_data[k] = processing.interp_motion(v)

        if only_motion: # save time
            music_data = None
        else:
            music_data = processing.music_features_all(
                music_path, tempo=music_tempo)
        
        if verbose:
            print (f'---- {video_name} ----')
            for k, v in motion_data.items():
                print (f'[motion] {k}: {v.shape}')
            for k, v in music_data.items():
                print (f'[music] {k}: {v.shape}')
        
        return {
            'video_name': video_name,
            'music_name': music_name,
            'motion_path': motion_path,
            'music_path': music_path,
            'motion_data': motion_data,
            'music_data': music_data,
            'music_tempo': music_tempo,
        }

    @classmethod
    def plot_music(cls, music_data, save_path=None):
        """Visualize muisc feature data."""
        figsize = (19.2, 10.8)
        nrows = len(music_data)
        fig, ax = plt.subplots(nrows=nrows, sharex=True, figsize=figsize)
        ax = [ax] if nrows == 1 else ax
        for i, (key, value) in enumerate(music_data.items()):
            ax[i].plot(value)
            ax[i].set_title(key)
        if save_path is None:
            plt.show()  # interactive
        else:
            fig.savefig(save_path)
        plt.close()


    @classmethod
    def plot_joints(cls, joints, save_path=None):
        """Visualize 3D joints."""
        out_dir, fname = None, None
        if save_path:
            out_dir = os.path.dirname(save_path)
            fname = os.path.basename(save_path).split('.')[0]
        plot_motion.animate_matplotlib(
            positions=[joints],
            colors=['r'],
            titles=[''],
            fig_title='smpl_joints',
            fps=60,
            figsize=(5.0, 5.0),
            out_dir=out_dir,
            fname=fname,
        )
    
    @classmethod
    def plot_smpl_poses(cls, smpl_poses, smpl_trans=None, save_path=None):
        """Visualize SMPL joint angles, along with global translation."""
        fk_engine = fk.SMPLForwardKinematics()
        joints = fk_engine.from_aa(smpl_poses, smpl_trans)
        cls.plot_joints(joints, save_path)

