import os
import shutil
import numpy as np

src = 'resource/jvs_ver1_phonemes/jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz'
dst = '/var/tmp/ram/jvs_ver1_phonemes/jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz'

person_known_size = 16
person_unknown_size = 16
voice_train_size = 64
voice_check_size = 8

person_no_list = np.array([8, 16, 17, 21, 23, 29, 35, 37, 42, 46, 47, 50, 54, 58, 59, 73, 88, 97])
voice_no_list  = np.array([5, 18, 24, 40, 42, 44, 46, 55, 56, 59, 60, 61, 63, 65, 71, 73, 75, 81, 84, 85, 87, 93, 94, 98])

def calc_file_idx(no_list, base_size):
    last_append_size = 0
    while True:
        append_size = np.count_nonzero(no_list < base_size + last_append_size)
        if append_size == last_append_size:
            break
        last_append_size = append_size
    return base_size + last_append_size

person_known_idx   = calc_file_idx(person_no_list, person_known_size)
person_unknown_idx = calc_file_idx(person_no_list, person_known_size + person_unknown_size)
voice_train_idx    = calc_file_idx(voice_no_list, voice_train_size)
voice_check_idx    = calc_file_idx(voice_no_list, voice_train_size + voice_check_size)

known_person_list   = list(filter(lambda x:x not in person_no_list, np.arange(person_known_idx)))
# unknown_person_list = list(filter(lambda x:x not in person_no_list, np.arange(person_known_idx, person_unknown_idx)))
voice_list          = list(filter(lambda x:x not in voice_no_list, np.arange(voice_check_idx)))

for person in known_person_list:
    
    print('[Processing] person:', person + 1)
    
    os.makedirs(os.path.split(dst)[0] % {'person': person + 1}, exist_ok=True)

    for voice in voice_list:

        specific = { 'person': person + 1, 'voice': voice + 1 }

        shutil.copyfile(src % { **specific, 'deform_type': 'stretch' }, dst % { **specific, 'deform_type': 'stretch' })
        shutil.copyfile(src % { **specific, 'deform_type': 'variable' }, dst % { **specific, 'deform_type': 'variable' })
        os.symlink(os.path.split(dst)[1] % { **specific, 'deform_type': 'variable' }, dst % { **specific, 'deform_type': 'padding' })
