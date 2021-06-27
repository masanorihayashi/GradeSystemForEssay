# GradeSystemForEssay

# データについて
エッセイ（学習者が書いた作文）は以下にある(github上に置くのはまずいので）

- A1, A2, B1の三段階（もしテキストブックのほうが必要であれば教えて頂ければ）

`/home/lr/hayashi/github/GradeSystemForEssay/cefrj`

以下の各ディレクトリの説明

- `/original/ ` 学習者が書いた作文（日本語はJPに置換）

- `/correct/ ` プロが添削した作文

- 当面それ以外は不要と思われる（もし何か追加で必要なら教えて下さい）

# プログラムの動かし方

0. (仮想環境かなにかでpip install -r requirements.txt)

1. 学習時

`python run.py -m train -o test.pkl`

=> run.py と同じディレクトリにhoge.pklが生成される

※ 注意：今は全データを学習に回しているのでほんとはtrain/dev/testに回したほうが良いと思う


2. テスト時

`python run.py -m test -i hoge.dat`

=> run.pyと同じディレクトリにtest.pklが置いてあると動くはず，test.datは適当なエッセイなりのテキストファイル
