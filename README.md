# vc-alpha

声質変換の研究用コード。

1. 以下をダウンロードし、ファイルを解凍しておく

- https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus -> resource/jvs_ver1
- https://github.com/Hiroshiba/jvs_hiho -> resource/jvs_hiho
- https://github.com/Hiroshiba/julius4seg -> python/julius4seg

2. init_dataset.pyを実行する

一部のファイルを修正を行う。
修正できないファイルについては、そのリストに含まれる話者18人とコーパス24文の直積（共通項）をすべて取り除く。
（詳細は[こちら](dataset.md)）
