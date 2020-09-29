from collections import OrderedDict
import pickle
import os

import librosa
import numpy as np
import scipy
import scipy.signal as scisignal
import tqdm

FPS = 60
HOP_LENGTH = 512
SR = FPS * HOP_LENGTH
EPS = 1e-6
CACHE_DIR = '/home/rui/local/data/AIST++/music_features/'
os.makedirs(CACHE_DIR, exist_ok=True)

# ===========================================================
# Music Processing Fuctions
# ===========================================================
def music_features_all(path, tempo=120.0, concat=False):
    cache_path = os.path.join(
        CACHE_DIR, os.path.basename(path).replace('.wav', '.pkl'))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            features = pickle.load(f)
    else:
        data = music_load(path)
        envelope = music_envelope(data=data)

        # tempogram = music_tempogram(envelope=envelope)
        mfcc = music_mfcc(data=data)
        chroma = music_chroma(data=data)
        _, peak_onehot = music_peak_onehot(envelope=envelope)
        _, beat_onehot, _ = music_beat_onehot(envelope=envelope, start_bpm=tempo)

        features = OrderedDict({
            'envelope': envelope[:, None],
            # 'tempogram': tempogram,
            'mfcc': mfcc,
            'chroma': chroma,
            'peak_onehot': peak_onehot,
            'beat_onehot': beat_onehot,
        })
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    if concat:
        return np.concatenate([v for k, v in features.items()], axis=1)
    else:
        return features


def music_load(path):
    """Load raw music data."""
    data, _ = librosa.load(path, sr=SR)
    return data


def music_envelope(path=None, data=None):
    """Calculate raw music envelope."""
    assert (path is not None) or (data is not None)
    if data is None:
        data = music_load(path)
    envelope = librosa.onset.onset_strength(data, sr=SR)
    return envelope # (seq_len,)


def music_tempogram(path=None, envelope=None, win_length=384):
    """Calculate music feature: tempogram."""
    assert (path is not None) or (envelope is not None)
    if envelope is None:
        envelope = music_envelope(path)
    tempogram = librosa.feature.tempogram(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH, 
        win_length=win_length)
    return tempogram.T  # (seq_len, 384)


def music_mfcc(path=None, data=None, m_mfcc=20):
    """Calculate music feature: mfcc."""
    assert (path is not None) or (data is not None)
    if data is None:
        data = music_load(path)
    mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=m_mfcc)
    return mfcc.T  # (seq_len, 20)


def music_chroma(path=None, data=None, n_chroma=12):
    """Calculate music feature: chroma."""
    assert (path is not None) or (data is not None)
    if data is None:
        data = music_load(path)
    chroma = librosa.feature.chroma_cens(
        data, sr=SR, hop_length=HOP_LENGTH, n_chroma=n_chroma)
    return chroma.T  # (seq_len, 12)


def music_peak_onehot(path=None, envelope=None):
    """Calculate music onset peaks.
    
    Return:
        - envelope: float array with shape of (seq_len,)
        - peak_onehot: one-hot array with shape of (seq_len,)
    """
    assert (path is not None) or (envelope is not None)
    if envelope is None:
        envelope = music_envelope(path=path)
    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1
    return envelope, peak_onehot


def music_beat_onehot(
    path=None, envelope=None, start_bpm=120.0, tightness=100):
    """Calculate music beats.
    
    Return:
        - envelope: float array with shape of (seq_len,)
        - beat_onehot: one-hot array with shape of (seq_len,)
    """
    assert (path is not None) or (envelope is not None)
    if envelope is None:
        envelope = music_envelope(path=path)
    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
        start_bpm=start_bpm, tightness=tightness)
    beat_onehot = np.zeros_like(envelope, dtype=bool)
    beat_onehot[beat_idxs] = 1
    return envelope, beat_onehot, tempo


# ===========================================================
# Motion Processing Fuctions
# ===========================================================
def interp_motion(joints):
    """Interpolate 30FPS motion into 60FPS."""
    seq_len = joints.shape[0]
    x = np.arange(0, seq_len)
    fit = scipy.interpolate.interp1d(x, joints, axis=0, kind='cubic')
    joints = fit(np.linspace(0, seq_len-1, 2*seq_len))
    return joints


def motion_peak_onehot(joints):
    """Calculate motion beats.
    Kwargs:
        joints: [nframes, njoints, 3]
    Returns:
        - envelope: motion energy.
        - peak_onhot: motion beats.
        - peak_energy: motion beats energy.
    """
    # Calculate velocity.
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10) # 10 for 60FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    # Second-derivative of the velocity shows the energy of the beats
    derivative = np.zeros_like(envelope, dtype=np.float32)
    derivative[2:] = envelope[2:] - envelope[1:-1]
    derivative2 = np.zeros_like(envelope, dtype=np.float32)
    derivative2[3:] = derivative[3:] - derivative[2:-1]
    peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)

    # optimize peaks
    # peak_onehot[peak_energy<0.5] = 0
    return envelope, peak_onehot, peak_energy


# ===========================================================
# Metrics Processing Fuctions
# ===========================================================
def select_aligned(music_beats, motion_beats, tol=6):
    """ Select aligned beats between music and motion.

    For each motion beat, we try to find an one-to-one mapping in the music beats.
    Kwargs:
        music_beats: onehot vector
        motion_beats: onehot vector
        tol: tolerant number of frames [i-tol, i+tol]
    Returns:
        music_beats_aligned: aligned idxs list
        motion_beats_aligned: aligned idxs list
    """
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]

    music_beats_aligned = []
    motion_beats_aligned = []
    accu_inds = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        dists[accu_inds] = np.Inf
        ind = np.argmin(dists)

        if dists[ind] > tol:
            continue
        else:
            music_beats_aligned.append(music_beat_idxs[ind])
            motion_beats_aligned.append(motion_beat_idx)
            accu_inds.append(ind)

    music_beats_aligned = np.array(music_beats_aligned)
    motion_beats_aligned = np.array(motion_beats_aligned)
    return music_beats_aligned, motion_beats_aligned


def alignment_score(music_beats, motion_beats, sigma=3):
    """Calculate alignment score between music and motion."""
    if motion_beats.sum() == 0:
        return 0.0

    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]

    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        ind = np.argmin(dists)
        score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        score_all.append(score)
    return sum(score_all) / len(score_all) 


def calculate_metrics(
    music_paths, joints_list, bpm_list, tol=12, sigma=3, start_id=0, verbose=False):
    """Calculate metrics for (motion, music) pair"""
    assert len(music_paths) == len(joints_list)
    
    metrics = {
        'beat_coverage': [],
        'beat_hit': [],
        'beat_alignment': [],
        'beat_energy': [],
        'motion_energy': [],
    }

    for music_path, joints, bpm in tqdm.tqdm(zip(music_paths, joints_list, bpm_list)):
        bpm = 120 if bpm is None else bpm

        # extract beats
        # music_envelope, music_beats, tempo = music_beat_onehot(music_path, start_bpm=bpm)
        # music_envelope, music_beats = music_peak_onehot(music_path)
        music_features = music_features_all(music_path, tempo=bpm)
        music_envelope, music_beats = music_features['envelope'], music_features['beat_onehot']
        motion_envelope, motion_beats, motion_beats_energy = motion_peak_onehot(joints)
        
        end_id = min(motion_envelope.shape[0], music_envelope.shape[0])

        music_envelope = music_envelope[start_id:end_id]
        music_beats = music_beats[start_id:end_id]
        motion_envelope = motion_envelope[start_id:end_id]
        motion_beats = motion_beats[start_id:end_id]
        motion_beats_energy = motion_beats_energy[start_id:end_id]
        
        # alignment
        music_beats_aligned, motion_beats_aligned = select_aligned(
            music_beats, motion_beats, tol=tol)
        
        # metrics
        metrics['beat_coverage'].append(
            motion_beats.sum() / (music_beats.sum() + EPS))        
        metrics['beat_hit'].append(
            len(motion_beats_aligned) / (motion_beats.sum() + EPS))
        metrics['beat_alignment'].append(
            alignment_score(music_beats, motion_beats, sigma=sigma))
        metrics['beat_energy'].append(
            motion_beats_energy[motion_beats].mean() if motion_beats.sum() > 0 else 0.0)
        metrics['motion_energy'].append(
            motion_envelope.mean())
        
    for k, v in metrics.items():
        metrics[k] = sum(v) / len(v)
        if verbose:
            print (f'{k}: {metrics[k]:.3f}')
    return metrics











