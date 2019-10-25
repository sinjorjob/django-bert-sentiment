import re
import string
import mojimoji
import pickle
import torch
import torchtext
from app1.config import *
from app1.bert import BertTokenizer, get_config, BertModel, BertForCHABSA, load_vocab

# 単語分割用のTokenizerを用意
tokenizer_bert = BertTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)


def pickle_load(path):
    with open(path, 'rb') as f:
        TEXT = pickle.load(f)
    return TEXT

def pickle_dump(TEXT, path):
    with open(path, 'wb') as f:
        pickle.dump(TEXT, f)


def preprocessing_text(text):
    # 半角・全角の統一
    text = mojimoji.han_to_zen(text) 
    # 改行、半角スペース、全角スペースを削除
    text = re.sub('\r', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('　', '', text)
    text = re.sub(' ', '', text)
    # 数字文字の一律「0」化
    text = re.sub(r'[0-9 ０-９]+', '0', text)  # 数字

    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    return text


# 前処理と単語分割をまとめた関数を定義
# 単語分割の関数を渡すので、tokenizer_bertではなく、tokenizer_bert.tokenizeを渡す点に注意
def tokenizer_with_preprocessing(text, tokenizer=tokenizer_bert.tokenize):
    text = preprocessing_text(text)
    ret = tokenizer(text)  # tokenizer_bert
    return ret


def create_tensor(text, max_length,TEXT):
    #入力文章をTorch Teonsor型にのINDEXデータに変換
    token_ids = torch.ones((max_length)).to(torch.int64)
    ids_list = list(map(lambda x: TEXT.vocab.stoi[x] , text))
    print(ids_list)
    for i, index in enumerate(ids_list):
        token_ids[i] = index
    return token_ids


def create_vocab_text():
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                            lower=False, include_lengths=True, batch_first=True, fix_length=max_length, init_token="[CLS]", eos_token="[SEP]", pad_token='[PAD]', unk_token='[UNK]')
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path=DATA_PATH, train='train_dumy.tsv',
        test='test_dumy.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])
    vocab_bert, ids_to_tokens_bert = load_vocab(vocab_file=VOCAB_FILE)
    TEXT.build_vocab(train_val_ds, min_freq=1)
    TEXT.vocab.stoi = vocab_bert
    pickle_dump(TEXT, PKL_FILE)

    return TEXT
 

def conver_to_model_format(input_seq, TEXT):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_text = tokenizer_with_preprocessing(input_seq) #入力文章を前処理しTensorに変換
    input_text.insert(0, '[CLS]')
    input_text.append('[SEP]')
    text = create_tensor(input_text, max_length ,TEXT)
    text = text.unsqueeze_(0)   #  torch.Size([256])  > torch.Size([1, 256])
    input = text.to(device) # GPUが使えるならGPUにデータを送る
    return input

def build_bert_model():
    TEXT = pickle_load(PKL_FILE)   #vocabデータのロード
    config = get_config(file_path=BERT_CONFIG)
    net_bert = BertModel(config) # BERTモデルを作成します
    net_trained = BertForCHABSA(net_bert)
    net_trained.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_trained.eval()   # モデルを検証モードに
    net_trained.to(device)

    return net_trained, TEXT


def highlight(word, attn):
    "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"

    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)


def mk_html(input, preds, normlized_weights, TEXT):
    "HTMLデータを作成する"

    # indexの結果を抽出
    index = 0
    sentence = input.squeeze_(0) # 文章  #  torch.Size([1, 256])  > torch.Size([256]) 
    pred = preds[0]  # 予測


    # 予測結果を文字に置き換え
    if pred == 0:
        pred_str = "Negative"
        html = '推論ラベル：<font color=red>{}</font><br><hr>'.format(pred_str)
    else:
        pred_str = "Positive"
        html = '推論ラベル：<font color=blue>{}</font><br><hr>'.format(pred_str)

    # Self-Attentionの重みを可視化。Multi-Headが12個なので、12種類のアテンションが存在

    for i in range(12):

        # indexのAttentionを抽出と規格化
        # 0単語目[CLS]の、i番目のMulti-Head Attentionを取り出す
        # indexはミニバッチの何個目のデータかをしめす
        attens = normlized_weights[index, i, 0, :]
        attens /= attens.max()

        #html += '[BERTのAttentionを可視化_' + str(i+1) + ']<br>'
        for word, attn in zip(sentence, attens):

            # 単語が[SEP]の場合は文章が終わりなのでbreak
            if tokenizer_bert.convert_ids_to_tokens([word.numpy().tolist()])[0] == "[SEP]":
                break

            # 関数highlightで色をつける、関数tokenizer_bert.convert_ids_to_tokensでIDを単語に戻す
            #html += highlight(tokenizer_bert.convert_ids_to_tokens(
            #    [word.numpy().tolist()])[0], attn)
        #html += "<br><br>"

    # 12種類のAttentionの平均を求める。最大値で規格化
    all_attens = attens*0  # all_attensという変数を作成する
    for i in range(12):
        attens += normlized_weights[index, i, 0, :]
    attens /= attens.max()

    html += '[BERTのAttentionを可視化_ALL]<br>'
    for word, attn in zip(sentence, attens):

        # 単語が[SEP]の場合は文章が終わりなのでbreak
        if tokenizer_bert.convert_ids_to_tokens([word.numpy().tolist()])[0] == "[SEP]":
            break

        # 関数highlightで色をつける、関数tokenizer_bert.convert_ids_to_tokensでIDを単語に戻す
        html += highlight(tokenizer_bert.convert_ids_to_tokens(
            [word.numpy().tolist()])[0], attn)
    html += "<br><br>"

    return html


