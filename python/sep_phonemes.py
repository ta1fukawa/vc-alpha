import csv
import os

import librosa
import numpy as np
import pyworld
import scipy.interpolate

src = 'resource/jvs_ver1_fixed/jvs%(person)03d/VOICEACTRESS100_%(voice)03d.wav'
lab = 'resource/jvs_ver1_fixed/jvs%(person)03d/VOICEACTRESS100_%(voice)03d.lab'
dst = 'resource/jvs_ver1_phonmes/%(err_type)s/jvs%(person)03d/%(fix_type)s_%(file_idx)03d.npz'

for person in range(100):

    print('[Processing] person:', person + 1)

    class Data():
        def __init__(self):
            self.variable_f0 = list()
            self.variable_sp = list()
            self.variable_ap = list()

            self.stretch_f0 = list()
            self.stretch_sp = list()
            self.stretch_ap = list()

            self.label = list()
            
    noerr = Data()
    pserr = Data()
    vcerr = Data()

    for voice in range(100):

        person_exclusion_list = [8, 16, 17, 21, 23, 29, 35, 37, 42, 46, 47, 50, 54, 58, 59, 73, 88, 97]
        voice_exclusion_list  = [5, 18, 24, 40, 42, 44, 46, 55, 56, 59, 60, 61, 63, 65, 71, 73, 75, 81, 84, 85, 87, 93, 94, 98]

        if person in person_exclusion_list and voice in voice_exclusion_list:
            continue

        specific = { 'person': person + 1, 'voice': voice + 1 }

        wave, sr = librosa.load(src % specific, sr=24000, dtype=np.float64, mono=True)

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
            target_length   = 32

            start_frame = int(float(start_sec) * separation_rate)
            end_frame   = int(float(end_sec) * separation_rate)
            strech_rate = (end_frame - start_frame) / target_length

            before_t  = np.linspace(0, target_length - 1, end_frame - start_frame)
            after_t   = np.arange(target_length)

            # 長さをストレッチして固定長化
            stretch_f0 = scipy.interpolate.interp1d(before_t, f0[start_frame:end_frame], kind='linear')(after_t)
            stretch_sp = librosa.phase_vocoder(sp[start_frame:end_frame].T, strech_rate, hop_length=hop_length).T[:target_length]
            stretch_ap = librosa.phase_vocoder(ap[start_frame:end_frame].T, strech_rate, hop_length=hop_length).T[:target_length]

            if person in person_exclusion_list:
                data = pserr
            elif voice in voice_exclusion_list:
                data = vcerr
            else:
                data = noerr

            data.variable_f0.append(f0[start_frame:end_frame].astype(np.float32))
            data.variable_sp.append(sp[start_frame:end_frame].astype(np.float32))
            data.variable_ap.append(ap[start_frame:end_frame].astype(np.float32))

            data.stretch_f0.append(stretch_f0.astype(np.float32))
            data.stretch_sp.append(stretch_sp.astype(np.float32))
            data.stretch_ap.append(stretch_ap.astype(np.float32))

            data.label.append(phoneme)
    
    for err_type in ['pserr', 'vcerr', 'noerr']:

        batch_size = 32
        if err_type == 'pserr':
            data = pserr
        elif err_type == 'vcerr':
            data = vcerr
        else:
            data = noerr

        os.makedirs(os.path.split(dst)[0] % { 'err_type': err_type, 'person': person + 1 }, exist_ok=True)

        for file_idx in range(len(data.label) // batch_size):
            specific = { 'err_type': err_type, 'person': person + 1, 'file_idx': file_idx + 1 }
            start_frame = file_idx * batch_size
            end_frame   = (file_idx + 1) * batch_size

            variable = {
                'f0': data.variable_f0[start_frame:end_frame],
                'sp': data.variable_sp[start_frame:end_frame],
                'ap': data.variable_ap[start_frame:end_frame],
                'label': data.label[start_frame:end_frame],
            }

            np.savez_compressed(dst % (specific | { 'fix_type': 'variable' }), **variable)

            stretch = {
                'f0': data.stretch_f0[start_frame:end_frame],
                'sp': data.stretch_sp[start_frame:end_frame],
                'ap': data.stretch_ap[start_frame:end_frame],
                'label': data.label[start_frame:end_frame],
            }

            np.savez_compressed(dst % (specific | { 'fix_type': 'strech' }), **stretch)