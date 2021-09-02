import os
import shutil
import numpy as np

src = 'resource/jvs_ver1_phonemes/jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz'
dst = '/var/tmp/ram/jvs_ver1_phonemes/jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz'

person_known_size = 16
person_unknown_size = 16
voice_train_size = 64
voice_check_size = 8

for person in range(person_known_size + person_unknown_size):
    
    print('[Processing] person:', person + 1)
    
    os.makedirs(os.path.split(dst)[0] % {'person': person + 1}, exist_ok=True)

    for voice in range(voice_train_size + voice_check_size):

        specific = { 'person': person + 1, 'voice': voice + 1 }

        shutil.copyfile(src % { **specific, 'deform_type': 'stretch' }, dst % { **specific, 'deform_type': 'stretch' })
        shutil.copyfile(src % { **specific, 'deform_type': 'variable' }, dst % { **specific, 'deform_type': 'variable' })
        os.symlink(os.path.split(dst)[1] % { **specific, 'deform_type': 'variable' }, dst % { **specific, 'deform_type': 'padding' })
