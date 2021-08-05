# vc-alpha

声質変換の研究用コード。

## データセットの準備

### 以下をダウンロードしておく

- https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus -> resource/jvs_ver1
- https://github.com/Hiroshiba/jvs_hiho -> resource/jvs_hiho
- https://github.com/Hiroshiba/julius4seg -> python/julius4seg

### init_dataset.pyを実行

実行すると、以下の処理が行われる。

#### 修復可能なデータ

**ファイル名**

jvs_ver1/jvs058/parallel100/wav24kHz16bit/内のファイルについて、以下のようにファイル名を修正する。

- VOICEACTRESS100_015.wav -> VOICEACTRESS100_016.wav
- VOICEACTRESS100_016.wav -> VOICEACTRESS100_017.wav
- VOICEACTRESS100_017.wav -> VOICEACTRESS100_018.wav
- VOICEACTRESS100_018.wav -> VOICEACTRESS100_019.wav
- VOICEACTRESS100_019.wav -> VOICEACTRESS100_020.wav
- VOICEACTRESS100_020.wav -> VOICEACTRESS100_021.wav
- VOICEACTRESS100_021.wav -> VOICEACTRESS100_022.wav

**音声データ**

jvs_ver1/jvs058/parallel100/wav24kHz16bit/VOICEACTRESS100_014.wavについて、音声ファイルを読み込んで以下のように分離する。

- 先頭～03.53秒 -> VOICEACTRESS100_014.wav
- 03.94秒～末尾 -> VOICEACTRESS100_014.wav

#### 修復不可能なデータ

欠損・誤り等があるデータは下記の通りである。
このデータに含まれる話者18人とコーパス24文について、その直積（共通項）に相当するデータを取り除く。

**ファイルの欠損**

- jvs_ver1/jvs030/parallel100/wav24kHz16bit/VOICEACTRESS100_045.wav
- jvs_ver1/jvs074/parallel100/wav24kHz16bit/VOICEACTRESS100_094.wav
- jvs_ver1/jvs089/parallel100/wav24kHz16bit/VOICEACTRESS100_019.wav

**データの誤り等**

- jvs_ver1/jvs009/parallel100/wav24kHz16bit/VOICEACTRESS100_086.wav
- jvs_ver1/jvs009/parallel100/wav24kHz16bit/VOICEACTRESS100_095.wav
- jvs_ver1/jvs017/parallel100/wav24kHz16bit/VOICEACTRESS100_082.wav
- jvs_ver1/jvs018/parallel100/wav24kHz16bit/VOICEACTRESS100_072.wav
- jvs_ver1/jvs022/parallel100/wav24kHz16bit/VOICEACTRESS100_047.wav
- jvs_ver1/jvs024/parallel100/wav24kHz16bit/VOICEACTRESS100_088.wav
- jvs_ver1/jvs036/parallel100/wav24kHz16bit/VOICEACTRESS100_057.wav
- jvs_ver1/jvs038/parallel100/wav24kHz16bit/VOICEACTRESS100_006.wav
- jvs_ver1/jvs038/parallel100/wav24kHz16bit/VOICEACTRESS100_041.wav
- jvs_ver1/jvs043/parallel100/wav24kHz16bit/VOICEACTRESS100_085.wav
- jvs_ver1/jvs047/parallel100/wav24kHz16bit/VOICEACTRESS100_085.wav
- jvs_ver1/jvs048/parallel100/wav24kHz16bit/VOICEACTRESS100_043.wav
- jvs_ver1/jvs048/parallel100/wav24kHz16bit/VOICEACTRESS100_076.wav
- jvs_ver1/jvs051/parallel100/wav24kHz16bit/VOICEACTRESS100_025.wav
- jvs_ver1/jvs055/parallel100/wav24kHz16bit/VOICEACTRESS100_056.wav
- jvs_ver1/jvs055/parallel100/wav24kHz16bit/VOICEACTRESS100_076.wav
- jvs_ver1/jvs055/parallel100/wav24kHz16bit/VOICEACTRESS100_099.wav
- jvs_ver1/jvs059/parallel100/wav24kHz16bit/VOICEACTRESS100_061.wav
- jvs_ver1/jvs059/parallel100/wav24kHz16bit/VOICEACTRESS100_064.wav
- jvs_ver1/jvs059/parallel100/wav24kHz16bit/VOICEACTRESS100_066.wav
- jvs_ver1/jvs059/parallel100/wav24kHz16bit/VOICEACTRESS100_074.wav
- jvs_ver1/jvs060/parallel100/wav24kHz16bit/VOICEACTRESS100_082.wav
- jvs_ver1/jvs074/parallel100/wav24kHz16bit/VOICEACTRESS100_062.wav
- jvs_ver1/jvs098/parallel100/wav24kHz16bit/VOICEACTRESS100_060.wav
- jvs_ver1/jvs098/parallel100/wav24kHz16bit/VOICEACTRESS100_099.wav
