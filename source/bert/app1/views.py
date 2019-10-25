from django.shortcuts import render
from django.views import generic
from .forms import InputForm
from app1.utils import build_bert_model, conver_to_model_format, mk_html
from app1.config import *
from IPython.display import HTML

import torch


class InputView(generic.FormView):
    form_class = InputForm
    template_name = 'app1/demo.html'

    def form_valid(self, form):
        net_trained, TEXT = build_bert_model()  #BERTモデルのBuildとVocabデータの取得       
        input_seq = self.request.POST["messages"] #画面からの入力文章を取得
        input = conver_to_model_format(input_seq, TEXT) #入力文章をモデルの入力フォーマットに変換
        #学習済みBERTモデルでネガポジ予測（outputsとAttention情報を取得）
        outputs, attention_probs = net_trained(input, token_type_ids=None, attention_mask=None,
                                       output_all_encoded_layers=False, attention_show_flg=True)
        _, preds = torch.max(outputs, 1)  # 予測結果(0 or 1)を取得
        html_output = mk_html(input, preds, attention_probs, TEXT)  # HTMLコードを作成

        context = {
            'input_seq': input_seq,
            'html_output': html_output,
        }
        return render(self.request, 'app1/demo.html', context)

    def form_invalid(self, form):
        return render(self.request, 'app1/demo.html', {'form': form})

