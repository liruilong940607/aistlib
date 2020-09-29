import os
import shutil
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
from matplotlib import pyplot as plt, animation as animation

from . import ffmpeg


SMPL_NR_JOINTS = 24
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def animate_matplotlib(
    positions, colors, titles, fig_title, parents=SMPL_PARENTS, audio_path=None,
    change_color_frames=None, color_after_change=None, overlay=False,
    fps=30, out_dir=None, to_video=True, fname=None, keep_frames=True,
    figsize=(10.8, 10.8)):
    """
    Visualize motion given 3D positions. Can visualize several motions side by side. If
    the sequence lengths don't match, all animations are displayed until the shortest
    sequence length.
    Args:
        positions: a list of np arrays in shape (seq_length, n_joints, 3) giving the 3D
            positions per joint and frame
        colors: list of color for each entry in `positions`
        titles: list of titles for each entry in `positions`
        fig_title: title for the entire figure
        parents: skeleton structure
        fps: frames per second
        change_color_frames: frame ids that the color of the plot is changed (for each
            entry in `positions`)
        color_after_change: what color to apply after `change_color_after_frame`
        overlay: if true, all entries in `positions` are plotted into the same subplot
        out_dir: output directory where the frames and video is stored. Don't pass for
            interactive visualization.
        to_video: whether to convert frames into video clip or not.
        fname: video file name.
        keep_frames: Whether to keep video frames or not.
    """
    seq_length = np.amin([pos.shape[0] for pos in positions])
    n_joints = positions[0].shape[1]
    pos = positions

    # create figure with as many subplots as we have skeletons
    fig = plt.figure(figsize=figsize)
    plt.clf()
    n_axes = 1 if overlay else len(pos)
    axes = [fig.add_subplot(1, n_axes, i + 1, projection='3d') for i in range(n_axes)]
    fig.suptitle(fig_title)

    # create point object for every bone in every skeleton
    all_lines = []
    # available_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    for i, joints in enumerate(pos):
        idx = 0 if overlay else i
        ax = axes[idx]

        lines_j = [
            ax.plot(joints[0:1, n, 0], joints[0:1, n, 1], joints[0:1, n, 2], '-o',
                    markersize=2.0, color=colors[i])[0] for n in range(1, n_joints)]
        all_lines.append(lines_j)

        ax.set_title(titles[i])

    # dirty hack to get equal axes behaviour
    min_val = np.amin(pos[0], axis=(0, 1))
    max_val = np.amax(pos[0], axis=(0, 1))
    max_range = (max_val - min_val).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max_val[0] + min_val[0])
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max_val[1] + min_val[1])
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max_val[2] + min_val[2])

    for ax in axes:
        ax.set_aspect('auto')
        # ax.axis('off')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        ax.view_init(elev=90, azim=-90)

    def on_move(event):
        # find which axis triggered the event
        source_ax = None
        for i in range(len(axes)):
            if event.inaxes == axes[i]:
                source_ax = i
                break

        # transfer rotation and zoom to all other axes
        if source_ax is None:
            return

        for i in range(len(axes)):
            if i != source_ax:
                axes[i].view_init(elev=axes[source_ax].elev, azim=axes[source_ax].azim)
                axes[i].set_xlim3d(axes[source_ax].get_xlim3d())
                axes[i].set_ylim3d(axes[source_ax].get_ylim3d())
                axes[i].set_zlim3d(axes[source_ax].get_zlim3d())
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig_text = fig.text(0.05, 0.05, '')

    def update_frame(num, positions, lines):
        for l in range(len(positions)):
            k = 0
            pos = positions[l]
            points_j = lines[l]
            for i in range(1, len(parents)):
                a = pos[num, i]
                b = pos[num, parents[i]]
                p = np.vstack([b, a])
                points_j[k].set_data(p[:, :2].T)
                points_j[k].set_3d_properties(p[:, 2].T)
                if (change_color_frames is not None and \
                    change_color_frames[l] is not None and \
                    num in change_color_frames[l]):
                    points_j[k].set_color(color_after_change)
                else:
                    points_j[k].set_color(colors[l])
                k += 1

        time_passed = '{:>.2f} seconds passed'.format(1 / fps * num)
        fig_text.set_text(time_passed)

    # create the animation object, for animation to work reference to this object must be kept
    fargs = (pos, all_lines)
    line_ani = animation.FuncAnimation(
        fig, update_frame, seq_length, fargs=fargs, interval=1000 / fps)

    if out_dir is None:
        plt.show()  # interactive
    else:
        video_dir = os.path.join(out_dir, "videos")
        save_to = os.path.join(out_dir, "frames", fname + "_skeleton")

        if not os.path.exists(save_to):
            os.makedirs(save_to)

        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        # Save frames to disk.
        for j in range(0, seq_length):
            update_frame(j, *fargs)
            fig.savefig(
                os.path.join(save_to, 'frame_{:0>4}.{}'.format(j, "jpg")), dip=1000)

        # Create a video clip.
        if to_video:
            ffmpeg.save_to_movie(
                os.path.join(video_dir, fname + "_skeleton.mp4"),
                os.path.join(save_to, 'frame_%04d.jpg'))

        if audio_path:
            ffmpeg.attach_audio_to_movie(
                os.path.join(video_dir, fname + "_skeleton.mp4"),
                audio_path,
                os.path.join(video_dir, fname + "_skeleton_audio.mp4")
            )

        # Delete frames if they are not required to store.
        if not keep_frames:
            shutil.rmtree(save_to)

    plt.close()


# if __name__ == '__main__':
#     import argparse
#     import os
#     import aist_plusplus_lib as aist

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', type=str,
#                         default='/home/rui/local/data/AIST/raw/',
#                         help='AIST video store directory.')
#     parser.add_argument('--video_name', type=str,
#                         default='gBR_sBM_c01_d04_mBR0_ch01',
#                         help='AIST video name.')
#     parser.add_argument('--pred_file', type=str,
#                         default=None,
#                         help='Prediction from ST-Transformer. (.npy)')
#     args = parser.parse_args()

#     # available_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    
#     if args.pred_file is None:
#         video_name = args.video_name
#         music_path = os.path.join(args.data_dir, f'{video_name}.wav')
#         motion_path = os.path.join(args.data_dir, f'{video_name}.pkl')
#         keypoints = aist.load(motion_path)['smpl_joints']
#     else:
#         video_name = os.path.basename(args.pred_file)[:-4]        
#         music_path = os.path.join(args.data_dir, f'{video_name}.wav')
#         keypoints = np.load(args.pred_file)

#     animate_matplotlib(
#         positions=[keypoints],
#         colors=['r'],
#         titles=[''],
#         audio_path=music_path,
#         color_after_change='g',
#         fig_title='smpl_joints',
#         out_dir='./data/vis_plot/',
#         fname=video_name,
#     )
