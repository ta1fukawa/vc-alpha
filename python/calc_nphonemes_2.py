import numpy as np

src = 'resource/jvs_ver1_phonemes_2/jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz'
dst = 'resource/jvs_ver1_nphonemes_2_%(condition)s.txt'

def calc_nphonemes(person, voice_list):
    nphonemes_list = list()
    for voice in voice_list:
        specific = { 'person': person + 1, 'voice': voice + 1, 'deform_type': 'variable' }
        data = np.load(src % specific, allow_pickle=True)
        nphonemes = len(data['f0'])
        nphonemes_list.append(str(nphonemes))
    return nphonemes_list

complete_list  = np.arange(100)
person_no_list = [8, 16, 17, 21, 23, 29, 35, 37, 42, 46, 47, 50, 54, 58, 59, 73, 88, 97]
person_ok_list = list(filter(lambda x:x not in person_no_list, complete_list))
voice_no_list  = [5, 18, 24, 40, 42, 44, 46, 55, 56, 59, 60, 61, 63, 65, 71, 73, 75, 81, 84, 85, 87, 93, 94, 98]
voice_ok_list  = list(filter(lambda x:x not in voice_no_list, complete_list))

ok_nphonemes = calc_nphonemes(0, complete_list)
no_nphonemes = calc_nphonemes(person_no_list[0], voice_ok_list)

with open(dst % { 'condition': 'ok' }, 'w') as f:
    f.write('\n'.join(ok_nphonemes))

with open(dst % { 'condition': 'no' }, 'w') as f:
    f.write('\n'.join(no_nphonemes))
