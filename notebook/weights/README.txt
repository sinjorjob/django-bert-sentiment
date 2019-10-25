* BERT日本語Pretrainedモデル

近年提案されたBERTが様々なタスクで精度向上を達成しています。BERTの[[公式サイト:https://github.com/google-research/bert]]では英語pretrainedモデルや多言語pretrainedモデルが公開されており、そのモデルを使って対象タスク(例: 評判分析)でfinetuningすることによってそのタスクを高精度に解くことができます。

多言語pretrainedモデルには日本語も含まれていますので日本語のタスクに多言語pretrainedモデルを利用することも可能ですが、基本単位がほぼ文字となっていることは適切ではないと考えます。そこで、入力テキストを形態素解析し、形態素をsubwordに分割したものを基本単位とし、日本語テキストのみ(Wikipediaを利用)でpretrainingしました。

** ダウンロード
- [[Japanese_L-12_H-768_A-12_E-30_BPE.zip:http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JapaneseBertPretrainedModel/Japanese_L-12_H-768_A-12_E-30_BPE.zip&name=Japanese_L-12_H-768_A-12_E-30_BPE.zip]] (1.6G)

公式で配布されているpretrainedモデルと同様のファイル形式になっており、
- TensorFlow checkpoint (bert_model.ckpt.meta, bert_model.ckpt.index, bert_model.ckpt.data-00000-of-00001)
- 語彙リストファイル (vocab.txt)
- configファイル (bert_config.json)

が含まれています。また、pytorch版BERT ([[pytorch-pretrained-BERT:https://github.com/huggingface/pytorch-pretrained-BERT]])用に変換したモデル (pytorch_model.bin)も同梱しています。

** 詳細
以下に日本語pretrainedモデルの詳細を示します。
- 入力テキスト: 日本語Wikipedia全て (約1,800万文)
- 入力テキストに[[Juman++:https://github.com/ku-nlp/jumanpp]]で形態素解析を行い、さらに[[BPE:https://github.com/rsennrich/subword-nmt]]を適用しsubwordに分割
- BERT_{BASE}と同じ設定 (12-layer, 768-hidden, 12-heads)
- 30 epoch (1GPUで1epochに約1日かかるのでpretrainingに約30日)
- 語彙数: 32,000 (形態素、subwordを含む)
- max_seq_length: 128

BERTの[[公式スクリプト:https://github.com/google-research/bert]] (run_classifier.pyなど)を用いてfinetuningする方法は公式で配布されているpretrainedモデルと同様で、以下のようにオプションを指定してください。

 export BERT_BASE_DIR=/path/to/Japanese_L-12_H-768_A-12_E-30_BPE

 python run_classifier.py \
  ...
   --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
     --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --do_lower_case False

入力テキストはJuman++で形態素解析し、形態素単位に分割してください。(BPEは適用する必要はありません。finetuning時に語彙リストを参照しながら自動的にsubwordに分割されます。)

注意: --do_lower_case False オプションをつけてください。これをつけないと、濁点が落ちてしまいます。また、tokenization.pyの以下の行をコメントアウトしてください。これを行わないと漢字が全て一文字単位になってしまいます。
 # text = self._tokenize_chinese_chars(text)

pytorch-pretrained-BERTでfinetuningする場合 (examples/run_classifier.py)、

 python run_classifier.py \
  ..
   --bert_model $BERT_BASE_DIR

のようにしてモデルを指定し、--do_lower_case オプションをつけないでください(つけるとTrueになります)。また、公式スクリプトと同様、pytorch_pretrained_bert/tokenization.pyに対して、上記のコメントアウトをしてください。


なお、形態素解析を行わず、文に対してSentencepieceを用いてpretrainingしたものが https://github.com/yoheikikuta/bert-japanese (日本語Wikipediaで学習)やhttps://www.hottolink.co.jp/blog/20190311-2 (日本語twitterデータで学習)で公開されています。

** 参考文献
柴田 知秀, 河原 大輔, 黒橋 禎夫: BERTによる日本語構文解析の精度向上, 言語処理学会 第25回年次大会,  pp.205-208,  名古屋,  (2019.3). ([[pdf:http://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/F2-4.pdf]], [[slide:https://speakerdeck.com/tomohideshibata/nlp2019-bert-parsing-shibata]])
