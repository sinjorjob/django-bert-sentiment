from django.conf import settings
import os

DATA_PATH = os.path.join(settings.BASE_DIR, 'app1/data')
PKL_FILE = os.path.join(DATA_PATH, 'text.pkl')
BERT_CONFIG = os.path.join(DATA_PATH, 'bert_config.json')
MODEL_FILE = os.path.join(DATA_PATH, 'bert_fine_tuning_chABSA.pth')
VOCAB_FILE = os.path.join(DATA_PATH, 'vocab.txt')
max_length = 256

