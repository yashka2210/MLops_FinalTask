import torch
from transformers import RobertaForSequenceClassification, RobertaConfig, get_linear_schedule_with_warmup, set_seed
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
from tree_sitter import Language, Parser
from model import Model
from test import test
from train import TextDataset
from predict import predict
from data_preprocessing import file_inner, obfuscate
import pytest

base = 'microsoft/unixcoder-base'  # 'microsoft/unixcoder-base' 'microsoft/codebert-base'
model_id = "saved_models"  # '/kaggle/input/linevul-adapter' '/kaggle/input/unixcoder-adapter'

tokenizer = AutoTokenizer.from_pretrained(base)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
set_seed(n_gpu)

config = RobertaConfig.from_pretrained(base)
config.num_labels = 1
model = RobertaForSequenceClassification.from_pretrained(base, config=config, ignore_mismatched_sizes=True).to(device)
model = Model(model, config, tokenizer)
model.to(device)

config = PeftConfig.from_pretrained(model_id)
model = PeftModel.from_pretrained(model, model_id)

# Настраиваем парсер для C#
parser = Parser()
CSHARP_LANGUAGE = Language('build/my-languages.so', 'c_sharp')
parser.set_language(CSHARP_LANGUAGE)

def test_metrics():
    test_data_file = "data/test.csv"
    test_dataset = TextDataset(tokenizer, test_data_file=test_data_file, file_type='test')
    result = test(model, tokenizer, test_dataset, device)
    assert (result['test_f1'] > 0.8)

def test_file_with_vuls():
    string_data = file_inner("data/file_with_vuls.cs")
    string_data = obfuscate(parser, string_data)
    result = predict(model, tokenizer, string_data, device, do_linelevel_preds=True)
    assert sum(result['vulnerable']) > 0

def test_without_vuls():
    string_data = file_inner("data/file_without_vuls.cs")
    string_data = obfuscate(parser, string_data)
    result = predict(model, tokenizer, string_data, device, do_linelevel_preds=True)
    assert sum(result['vulnerable']) == 0

