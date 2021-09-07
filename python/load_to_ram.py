import os
import shutil
import numpy as np

src = 'resource/jvs_ver1_phonemes_v5/jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz'
dst = '/var/tmp/ram/jvs_ver1_phonemes_v5/jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz'

voice_train_size = 20
voice_check_size = 6
known_person_list   = [89, 90, 39, 2, 27, 68, 24, 99, 40, 34, 61, 71, 26, 63, 9, 92]
unknown_person_list = [94, 64, 0, 13, 77, 7, 16, 17, 57, 65, 84, 15, 46, 48, 70, 56]

def load_to_ram(person_list, voice_list):
    for person in person_list:
        
        print('[Processing] person:', person + 1)
        
        os.makedirs(os.path.split(dst)[0] % {'person': person + 1}, exist_ok=True)

        for voice in voice_list:

            specific = { 'person': person + 1, 'voice': voice + 1 }

            shutil.copyfile(src % { **specific, 'deform_type': 'stretch' }, dst % { **specific, 'deform_type': 'stretch' })
            shutil.copyfile(src % { **specific, 'deform_type': 'variable' }, dst % { **specific, 'deform_type': 'variable' })
            os.symlink(os.path.split(dst)[1] % { **specific, 'deform_type': 'variable' }, dst % { **specific, 'deform_type': 'padding' })

load_to_ram(known_person_list,   range(voice_train_size))
load_to_ram(unknown_person_list, range(voice_train_size, voice_train_size + voice_check_size))
