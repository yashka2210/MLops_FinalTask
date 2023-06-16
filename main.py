import streamlit as st
import os
import pandas as pd
from io import StringIO
import re
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, RobertaConfig, get_linear_schedule_with_warmup, set_seed
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from scipy.special import softmax

import string
import tree_sitter
from tree_sitter import Language, Parser
import codecs

from model import Model
from predict import predict
from data_preprocessing import obfuscate

#Настраиваем пул наших языков
Language.build_library(
  # Store the library in the `build` directory
  'build/my-languages.so',

  # Include one or more languages
  [
    'tree-sitter-c-sharp'
  ]
)

#Настраиваем парсер для C#
parser = Parser()
CSHARP_LANGUAGE = Language('build/my-languages.so', 'c_sharp')
parser.set_language(CSHARP_LANGUAGE)



#Подготовка модели для предикта
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

#Вывод на экран
st.title('Определение уязвимостей в коде C#')

st.markdown('Проверьте, **уязвим** ли ваш код.')
st.markdown('Загрузить ваш файл с кодом :point_down:')

uploaded_file = st.file_uploader("Выберите файл")
if uploaded_file is not None:
    with st.spinner('Производится сканирование ваших секретиков...'):
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        string_data = obfuscate(parser,string_data)
        res = predict(model, tokenizer, string_data, device, do_linelevel_preds = True)
        df = pd.DataFrame(res)
    st.markdown('**Результаты:**')
    if 1 in df['vulnerable'].unique():
        st.error('Найдены уязвимости', icon="🚨")
        st.dataframe(df[df['vulnerable'] == 1])
    else:
        st.success('Уязвимостей не обнаружено', icon="✅")
