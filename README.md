# GradeSystemForEssay

# データについて
## エッセイ（学習者が書いた作文）は以下

- A1, A2, B1の3段階

`/home/lr/hayashi/github/GradeSystemForEssay/cefrj`

以下の各ディレクトリの説明

- `/original/ ` 学習者が書いた作文（日本語はJPに置換）

- `/correct/ ` プロが添削した作文

## textbook

- A1, A2, B1, B2, C1, C2の6段階

`/home/lr/hayashi/github/GradeSystemForEssay/textbook`

以下の各ディレクトリの説明

- `/raw_xml/` textbookそれぞれ（全部で96冊分？）のxmlデータ

- `/raw/` `/raw_xml/`内のデータをパースしてテキスト抽出したデータ

- `/10_instance/[raw, raw_xml]/` textbookを10文ずつ分割して，作成したもの



# プログラムの動かし方

0. (仮想環境かなにかでpip install -r requirements.txt)

1. 学習時

`python run.py -m train -o test.pkl`

=> run.py と同じディレクトリにtest.pklが生成される

※ 注意：今は全データを学習に回しているのでほんとはtrain/dev/testに回したほうが良いと思う


2. テスト時

`python run.py -m test -i hoge.dat`

=> run.pyと同じディレクトリにtest.pklが置いてあると動くはず，test.datは適当なエッセイなりのテキストファイル
