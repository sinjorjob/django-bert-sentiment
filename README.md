# Django app for emotion analysis using Bert

###  本リポジトリのプログラムは下記の書籍「つくりながら学ぶ！PyTorchによる発展ディープラーニング」第8章「自然言語による感情分析（BERT)を参考に作成したものです。

https://github.com/YutaroOgawa/pytorch_advanced/blob/master/LICENSE  

上記書籍では、IMDbのデータセットを用いた英語文章のネガポジ分類になっていますが、このリポジトリでは日本語文章でのネガポジ分類モデルになっています。  
12個のAttentionの平均をとってどの単語を重視して判定したかも可視化できるようにしています。


学習データにはTISが無料で公開している機械学習で感情解析を行うためのデータセット「chABSA-dataset」を用いています。  

https://github.com/chakki-works/chABSA-dataset

chABSA-datasetのデータの本文がネガティブか、ポジティブ化を自動判定するDjangoのWEBアプリケーションのコードも付属しています。     なお、全データセットのうち、1970個を訓練データ数、843個をテストデータとしてモデルを構築しています。

BERTは京都大学が公開しているpytorch-pretrained-BERTモデルを利用しています。

http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB

**デモ動画**

![BERT](https://user-images.githubusercontent.com/34405452/67568657-c298d980-f767-11e9-8d3f-09230667772d.gif)


# フォルダ構成  

- notebook  
　　BERTモデル作成に至るまでの各種コード（JupyterNotebook形式）
- source  
    Djangoアプリケーションのソースコード
  

```
{'[PAD]': 0,'[UNK]': 1,'[CLS]': 2,'[SEP]': 3,'[MASK]': 4,'の': 5,'、': 6,　　・・・省略・・・
```


# Djangoアプリ構築手順


### 1.1 リポジトリをClone

### 1.2 各種ライブラリー導入  

最低限必要なモジュールの導入
```
conda create -n bert_model python=3.6
conda install pytorch=0.4 torchvision -c pytorch
conda install pytorch=0.4 torchvision cudatoolkit -c pytorch
pip install pyknp
pip install django
pip install django-bootstrap4
pip install django-widgets-improved
pip install torchtext
pip install mojimoji
pip install attrdict

```
juman++の環境構築は以下の記事を参考に実施してください。  
https://sinyblog.com/deaplearning/juman/

### 1.3 パスの変更

config.py内の各種パスを各環境に合わせて変更してください。

**※本アプリはWindowsのUbuntu環境で構築しています。**    
　Windows環境でも動くと思いますが、パス等は必要に応じて変更が必要です。

### 1.4 text.pklの作成

Bertモデル用のVocabデータを以下の手順で作成する必要があります。

```
python manage.py shell
from app1.utils import *
from app1.config import *
TEXT = create_vocab(PKL_FILE)
```
※bert\app1\data\text.pklが生成されます。

### 1.5 アプリ起動

```
cd bert 
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

アクセスＵＲＬ  
http://127.0.0.1:8000/demo/sentiment


###各種ファイルの概要
```
- bert_fine_tuning_chABSA.pth（学習済みモデルファイル）
　　→各自JupyterNotebookを参考にファインチューニングしたモデルファイルを配置してください。
- train.tsv（学習用データ)
- test.tsv(検証用データ)
- train_dumy.tsv（text.pkl生成用のダミーデータファイル)
- test_dumy.tsv（text.pkl生成用のダミーデータファイル)
- bert_config.json(Bertモデル用のパラメータファイル） 
- vocab.txt（BERT学習
```

bert_config.jsonとvocab.txtは以下からダウンロードできます。
※京都大学が公開しているBERT日本語Pretrainedモデル

http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB

