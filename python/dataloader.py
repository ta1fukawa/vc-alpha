import numpy as np
import torch
import librosa

class DataLoader(torch.utils.data.Dataset):
    '''
    スペクトログラムに変換済みの音声データセットを読み込む。
    PyTorchでの学習用。
    可変長データ、ストレッチ済みデータ、ゼロパディングに対応。

    Parameters
    ----------
    person_list : list of int
        ファイル名のperson番号-1の番号リスト。
    voice_list : list of int
        ファイル名のvoice番号-1の番号リスト。
    batch_size : tuple of int
        (person_length, voice_length)のタプル。
        ただしdeform_type=='stretch'の場合は(1, 1)。
    nphonemes_path : str
        音素長ファイルのパス。
    dataset_path : str
        データセットのパス。
        例："%(deform_type)/%(person)s/%(voice).wav"
    deform_type : str
        変形の種類
    phonemes_length : int
        deform_type=='padding'の場合の整形後のサイズ。
    '''

    def __init__(self, person_list, voice_list, batch_size, nphonemes_path, dataset_path, deform_type, phonemes_length=None, mel=False, seed=None):
        self.person_list = person_list
        self.voice_list  = voice_list
        self.batch_size  = batch_size

        with open(nphonemes_path % { 'condition': 'ok' }, 'r') as f:
            self.nphonemes_list = np.array(list(map(int, f.read().split('\n'))))[voice_list]

        self.dataset_path    = dataset_path
        self.deform_type     = deform_type
        self.phonemes_length = phonemes_length
        self.mel_basis       = librosa.filters.mel(sr=24000, n_fft=1024) if mel else None

        self.person_nbatches  = len(person_list) // batch_size[0]
        self.phoneme_nbatches = np.sum(self.nphonemes_list) // batch_size[1]

        if seed is not None:
            self.reset_shuffle(seed)
        else:
            self.shuffle = np.arange(len(self))

    def __len__(self):
        return self.person_nbatches * self.phoneme_nbatches

    def reset_shuffle(self, seed=0):
        np.random.seed(seed)
        self.shuffle = np.random.permutation(len(self))

    def __getitem__(self, batch_idx):
        '''
        Returns
        -------
        data : [batch_length, phoenemes_length, nfft // 2]
        true : [batch_length]
        '''

        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError()

        batch_idx = self.shuffle[batch_idx]

        # インデックスをperson方向とphoneme方向に分解
        phoneme_batch_idx, person_batch_idx = divmod(batch_idx, self.person_nbatches)

        # 読み込むべきファイルのpersonの範囲を求める
        person_start_idx = person_batch_idx * self.batch_size[0]
        person_end_idx   = (person_batch_idx + 1) * self.batch_size[0]

        # 読み込むべきファイルのvoiceの範囲を求める
        nphonemes_accumulation_list = np.cumsum(self.nphonemes_list)
        voice_start_idx = np.where(phoneme_batch_idx * self.batch_size[1] < nphonemes_accumulation_list)[0][0]
        voice_end_idx   = np.where((phoneme_batch_idx + 1) * self.batch_size[1] <= nphonemes_accumulation_list)[0][0] + 1
        nphonemes_accumulation_list = np.insert(nphonemes_accumulation_list, 0, 0)
        
        # 複数のvoiceを結合したデータから取り出すべき範囲を求める
        voice_start_phoneme = phoneme_batch_idx * self.batch_size[1] - nphonemes_accumulation_list[voice_start_idx]
        voice_end_phoneme   = voice_start_phoneme + self.batch_size[1]

        data = list()
        for person_idx in range(person_start_idx, person_end_idx):
            
            person_data = list()
            for voice_idx in range(voice_start_idx, voice_end_idx):
                specific = {
                    'person'     : self.person_list[person_idx] + 1,
                    'voice'      : self.voice_list[voice_idx] + 1,
                    'deform_type': self.deform_type,
                }
                pack = np.load(self.dataset_path % specific, allow_pickle=True)
                
                if self.mel_basis is not None:
                    if self.deform_type == 'stretch':
                        sp = np.array([np.dot(x, self.mel_basis.T) for x in pack['sp']])
                    elif self.deform_type == 'variable':
                        sp = np.array([np.dot(x, self.mel_basis.T) for x in pack['sp']])
                    elif self.deform_type == 'padding':
                        sp = np.array([self._zero_padding(np.dot(x, self.mel_basis.T)[:self.phonemes_length], self.phonemes_length) for x in pack['sp']])
                else:
                    if self.deform_type == 'stretch':
                        sp = pack['sp'][:, :, 1:]
                    elif self.deform_type == 'variable':
                        sp = np.array([x[:, 1:] for x in pack['sp']])
                    elif self.deform_type == 'padding':
                        sp = np.array([self._zero_padding(x[:self.phonemes_length, 1:], self.phonemes_length) for x in pack['sp']])

                person_data.extend(sp)
            data.extend(person_data[voice_start_phoneme:voice_end_phoneme])
        data = np.array(data)
        
        label  = np.concatenate([[person_idx] * self.batch_size[1] for person_idx in range(person_start_idx, person_end_idx)])

        data  = torch.from_numpy(data).float().to('cuda')
        label = torch.from_numpy(label).long().to('cuda')
        return data, label

    @staticmethod
    def _zero_padding(x, target_length):
        y_pad = target_length - len(x)
        return np.pad(x, ((0, y_pad), (0, 0)), mode='constant') if y_pad > 0 else x

def test():
    nphonemes_path = 'resource/jvs_ver1_nphonemes_%(condition)s.txt'
    dataset_path   = 'resource/jvs_ver1_phonemes/jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz'

    complete_list  = np.arange(100)
    person_no_list = [8, 16, 17, 21, 23, 29, 35, 37, 42, 46, 47, 50, 54, 58, 59, 73, 88, 97]
    person_ok_list = list(filter(lambda x:x not in person_no_list, complete_list))
    voice_no_list  = [5, 18, 24, 40, 42, 44, 46, 55, 56, 59, 60, 61, 63, 65, 71, 73, 75, 81, 84, 85, 87, 93, 94, 98]
    voice_ok_list  = list(filter(lambda x:x not in voice_no_list, complete_list))

    loader = DataLoader(complete_list, voice_ok_list, (2, 32), nphonemes_path, dataset_path, 'padding', 32)
    for data in loader:
        print(data[0].shape, data[1].shape)