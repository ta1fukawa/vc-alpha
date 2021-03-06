import csv
import os
import warnings
import sys
import multiprocessing

import librosa
import numpy as np
import pyworld
import scipy.interpolate

warnings.filterwarnings('ignore')

assert len(sys.argv) == 3

target_length = int(sys.argv[1])  # 32
min_sp_length = int(sys.argv[2])  # 16
min_nfile = 999

src = 'resource/jvs_ver1/fixed/jvs%(person)03d/VOICEACTRESS100_%(voice)03d.wav'
lab = 'resource/jvs_ver1/fixed/jvs%(person)03d/VOICEACTRESS100_%(voice)03d.lab'
dst_dir = 'resource/jvs_ver1/data_%(target_len)d_%(filter_len)d' % {'target_len': target_length, 'filter_len': min_sp_length}
dst = dst_dir + '/jvs%(person)03d/VOICEACTRESS100_%(idx)03d_%(deform_type)s.npz'

def process(person):
    print('[Processing] person:', person + 1)
    
    variable = { key: list() for key in ['f0', 'sp', 'ap'] }
    stretch  = { key: list() for key in ['f0', 'sp', 'ap'] }
    label    = { key: list() for key in ['label'] }

    for voice in range(100):

        specific = { 'person': person + 1, 'voice': voice + 1 }

        try:
            wave, sr = librosa.load(src % specific, sr=24000, dtype=np.float64, mono=True)
        except:
            continue

        with open(lab % specific, 'r', newline='', encoding='utf-8') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            labels = [row for row in tsv_reader]

        # 音響特徴量の抽出
        _f0, t = pyworld.dio(wave, sr) # 基本周波数の抽出
        # _f0, t = pyworld.harvest(wave, sr)
        f0 = pyworld.stonemask(wave, _f0, t, sr) # 洗練させるらしい f0 (n, )
        sp = pyworld.cheaptrick(wave, f0, t, sr) # スペクトル包絡の抽出 spectrogram (n, f)
        ap = pyworld.d4c(wave, f0, t, sr) # 非周期性指標の抽出 aperiodicity (n, f)

        for start_sec, end_sec, phoneme in labels:

            if phoneme in ['silB', 'silE', 'sp']:
                continue

            separation_rate = 200
            hop_length      = sr // separation_rate

            start_frame = int(float(start_sec) * separation_rate)
            end_frame   = int(float(end_sec) * separation_rate)

            try:
                start_frame += np.where(f0[start_frame:end_frame])[0][0]
                end_frame   -= np.where(f0[start_frame:end_frame][::-1])[0][0]
            except:
                continue
            if start_frame + min_sp_length + 1 >= end_frame:
                continue

            strech_rate = (end_frame - start_frame) / target_length

            before_t = np.linspace(0, target_length - 1, end_frame - start_frame)
            after_t  = np.arange(target_length)

            variable['f0'].append(f0[start_frame:end_frame].astype(np.float32))
            variable['sp'].append(sp[start_frame:end_frame].astype(np.float32))
            variable['ap'].append(ap[start_frame:end_frame].astype(np.float32))

            # 長さをストレッチして固定長化
            stretch_f0 = scipy.interpolate.interp1d(before_t, f0[start_frame:end_frame], kind='linear')(after_t)
            stretch_sp = librosa.phase_vocoder(sp[start_frame:end_frame].T, strech_rate, hop_length=hop_length).T
            stretch_ap = librosa.phase_vocoder(ap[start_frame:end_frame].T, strech_rate, hop_length=hop_length).T

            stretch['f0'].append(stretch_f0.astype(np.float32))
            stretch['sp'].append(stretch_sp.astype(np.float32))
            stretch['ap'].append(stretch_ap.astype(np.float32))

            label['label'].append(phoneme)
    
    os.makedirs(os.path.split(dst)[0] % { 'person': person + 1 }, exist_ok=True)

    nfile = len(label['label']) // 32
    for idx in range(nfile):

        specific = { 'person': person + 1, 'idx': idx + 1 }
            
        np.savez_compressed(dst % { **specific, 'deform_type': 'variable' }, **{key: value[idx * 32:(idx + 1) * 32] for key, value in variable.items()}, **label)
        np.savez_compressed(dst % { **specific, 'deform_type': 'stretch' }, **{key: value[idx * 32:(idx + 1) * 32] for key, value in stretch.items()}, **label)
        os.symlink(os.path.split(dst)[1] % { **specific, 'deform_type': 'variable' }, dst % { **specific, 'deform_type': 'padding' })

    return nfile

pool_obj = multiprocessing.Pool()

nfile_list = pool_obj.map(process, range(100))
print(np.min(nfile_list))
