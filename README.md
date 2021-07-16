# GradeSystemForEssay

# データについて
## エッセイ（学習者が書いた作文）は以下にある(github上に置くのはまずいので）

- A1, A2, B1の三段階

`/home/lr/hayashi/github/GradeSystemForEssay/cefrj`

以下の各ディレクトリの説明

- `/original/ ` 学習者が書いた作文（日本語はJPに置換）

- `/correct/ ` プロが添削した作文

## textbook

- aaaa

# プログラムの動かし方

0. (仮想環境かなにかでpip install -r requirements.txt)

1. 学習時

`python run.py -m train -o test.pkl`

=> run.py と同じディレクトリにtest.pklが生成される

※ 注意：今は全データを学習に回しているのでほんとはtrain/dev/testに回したほうが良いと思う


2. テスト時

`python run.py -m test -i hoge.dat`

=> run.pyと同じディレクトリにtest.pklが置いてあると動くはず，test.datは適当なエッセイなりのテキストファイル
