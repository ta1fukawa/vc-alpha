# vc-alpha

声質変換の研究用コード。
個人用なのでコードは美しくない。

1. 以下をダウンロードし、ファイルを解凍しておく

- https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus -> resource/jvs_ver1
- https://github.com/Hiroshiba/jvs_hiho -> resource/jvs_hiho
- https://github.com/Hiroshiba/julius4seg -> python/julius4seg
- https://osdn.net/projects/julius/downloads/71011/dictation-kit-4.5.zip/ -> resource/dictation-kit-4.5

2. juliusをインストールしておく

インストール方法は[こちら](https://github.com/julius-speech/julius)を参照。

3. init_dataset.pyを実行する

```Bash
python3 python/init_dataset.py
```

一部のファイルを修正を行っている。
修正できないファイルについては、そのリストに含まれる話者18人とコーパス24文の直積（共通項）をすべて取り除く。
（詳細は[こちら](dataset.md)）

4. sep_phonemes.pyを実行する

```Bash
python3 python/sep_phonemes.py
```

5. calc_nphonemes.pyを実行する

```Bash
python3 python/alc_nphonemes.py
```

6. main.pyを実行する

```Bash
python3 python/main.py
```

引数で色々なパターンを実験できるので、`-h`をつけてヘルプを確認してね。
