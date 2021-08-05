import os
import pathlib
import shutil
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path.append('python/julius4seg')

from julius4seg.sp_inserter import ModelType
from sample.run_segment import run_segment

src      = 'resource/jvs_ver1/jvs%(person)03d/parallel100/wav24kHz16bit/VOICEACTRESS100_%(voice)03d.wav'
yomi     = 'resource/jvs_hiho/voiceactoress100_spaced_julius.txt'
yomi_tmp = '/tmp/VOICEACTRESS100_%(voice)03d.txt'

hmm      = 'resource/dictation-kit-4.5/model/phone_m/jnas-mono-16mix-gid.binhmm'

dst      = 'resource/jvs_ver1_fixed/jvs%(person)03d/VOICEACTRESS100_%(voice)03d.wav'
wave_tmp = '/tmp/jvs%(person)03d_VOICEACTRESS100_%(voice)03d.wav'
lab      = 'resource/jvs_ver1_fixed/jvs%(person)03d/VOICEACTRESS100_%(voice)03d.lab'

with open(yomi, 'r') as f:
    yomi_list = f.readlines()

for voice in range(100):
    specific = { 'voice': voice + 1 }
    with open(yomi_tmp % specific, 'w') as f:
        f.write(yomi_list[voice])

for person in range(100):
    
    print('processing:', person + 1)
    os.makedirs(os.path.split(dst)[0] % {'person': person + 1}, exist_ok=True)

    for voice in range(100):

        if person in [8, 16, 17, 21, 23, 29, 35, 37, 42, 46, 47, 50, 54, 58, 59, 73, 88, 97] \
            and voice in [5, 18, 24, 40, 42, 44, 46, 55, 56, 59, 60, 61, 63, 65, 71, 73, 75, 81, 84, 85, 87, 93, 94, 98]:
            continue

        specific1 = { 'person': person + 1, 'voice': voice + 1 }
        specific2 = { 'person': person + 1, 'voice': voice + 2 }

        if person == 57 and voice in [21]:
            # 重複
            pass

        elif person == 57 and voice in [13]:
            # 結合
            wave24k, sr = librosa.load(src % specific1, sr=24000, mono=True)
            sf.write(dst % specific1, wave24k[:int(sr * 3.53)], sr, subtype='PCM_16')
            sf.write(dst % specific2, wave24k[int(sr * 3.94):], sr, subtype='PCM_16')
            
            wave16k1 = librosa.resample(wave24k[:int(sr * 3.53)], sr, 16000)
            wave16k2 = librosa.resample(wave24k[int(sr * 3.94):], sr, 16000)
            sf.write(wave_tmp % specific1, wave16k1, 16000, subtype='PCM_16')
            sf.write(wave_tmp % specific2, wave16k2, 16000, subtype='PCM_16')

        elif person == 57 and 13 < voice <= 20:
            # ずれ
            shutil.copy(src % specific1, dst % specific2)
            wave16k, sr = librosa.load(dst % specific2, sr=16000, mono=True)
            sf.write(wave_tmp % specific2, wave16k, sr, subtype='PCM_16')

        else:
            shutil.copy(src % specific1, dst % specific1)
            wave16k, sr = librosa.load(dst % specific1, sr=16000, mono=True)
            sf.write(wave_tmp % specific1, wave16k, sr, subtype='PCM_16')

        julius4seg_args = {
            'wav_file': pathlib.Path(wave_tmp % specific1),
            'input_yomi_file': pathlib.Path(yomi_tmp % specific1),
            'output_seg_file': pathlib.Path(lab % specific1),
            'input_yomi_type': 'katakana',
            'like_openjtalk': False,
            'input_text_file': None,
            'output_text_file': None,
            'hmm_model': hmm,
            'model_type': ModelType.gmm,
            'padding_second': 0,
            'options': None
        }

        try:
            run_segment(**julius4seg_args, only_2nd_path=False)
        except:
            run_segment(**julius4seg_args, only_2nd_path=True)

        os.remove(wave_tmp % specific1)

for voice in range(100):
    specific = { 'voice': voice + 1 }
    os.remove(yomi_tmp % specific)
