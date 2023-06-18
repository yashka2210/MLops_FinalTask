import re
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, RobertaConfig, get_linear_schedule_with_warmup, set_seed
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import os 
from scipy.special import softmax
from model import Model


class Input(object):
    def __init__(self,
                 input_tokens,
                 input_ids,
                 func):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.func=func
        
           
class TextData(Dataset):
    def __init__(self, tokenizer, funcs):
        self.examples = []
        for i in tqdm(range(len(funcs))):
            self.examples.append(tokenize_samples(funcs[i], tokenizer))
                                 
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids), str(self.examples[i].func)

    
def cleaner(code):
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    code = re.sub('\r','',code)
    code = re.sub('\t','',code)
    code = code.split('\n')
    code = [line.strip() for line in code if line.strip()]
    code = '\n'.join(code)
    return(code)


def set_seed(n_gpu, seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    
    
def tokenize_samples(func, tokenizer, block_size = 512):
    clean_func = cleaner(func)
    code_tokens = tokenizer.tokenize(str(clean_func))[:block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return Input(source_tokens, source_ids, clean_func)


def clean_special_token_values(all_values, padding=False):
    all_values[0] = 0
    if padding:
        idx = [index for index, item in enumerate(all_values) if item != 0][-1]
        all_values[idx] = 0
    else:
        all_values[-1] = 0
    return all_values


def get_word_att_scores(all_tokens: list, att_scores: list) -> list:
    word_att_scores = []
    for i in range(len(all_tokens)):
        token, att_score = all_tokens[i], att_scores[i]
        word_att_scores.append([token, att_score.cpu().detach().numpy()])
    return word_att_scores


def get_all_lines_score(word_att_scores: list):
    # verified_flaw_lines = [''.join(l) for l in verified_flaw_lines]
    # word_att_scores -> [[token, att_value], [token, att_value], ...]
    separator = ["Ċ", " Ċ", "ĊĊ", " ĊĊ"]
    all_lines_score = []
    score_sum = 0
    line_idx = 0
    flaw_line_indices = []
    line = ""
    for i in range(len(word_att_scores)):
        if ((word_att_scores[i][0] in separator) or (i == (len(word_att_scores) - 1))) and score_sum != 0:
            score_sum += word_att_scores[i][1]
            all_lines_score.append(score_sum)
            line = ""
            score_sum = 0
            line_idx += 1
        elif word_att_scores[i][0] not in separator:
            line += word_att_scores[i][0]
            score_sum += word_att_scores[i][1]
    return all_lines_score


def find_vul_lines(tokenizer, inputs_ids, attentions):  
    ids = inputs_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(ids)
    all_tokens = [token.replace("Ġ", "") for token in all_tokens]
    all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]
    original_lines = ''.join(all_tokens).split("Ċ")
    
    attentions = attentions[0][0]
    attention = None
    for i in range(len(attentions)):
        layer_attention = attentions[i]
        layer_attention = sum(layer_attention)
        if attention is None:
            attention = layer_attention
        else:
            attention += layer_attention
    attention = clean_special_token_values(attention, padding=True)
    word_att_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attention)
    all_lines_score = get_all_lines_score(word_att_scores)
    all_lines_score_with_label = sorted(range(len(all_lines_score)), key=lambda i: all_lines_score[i], reverse=True)  
    return all_lines_score_with_label




def predict(model, tokenizer, funcs, device, best_threshold = 0.5, do_linelevel_preds = True):

    check_dataset = TextData(tokenizer, funcs)
    check_sampler = SequentialSampler(check_dataset)
    check_dataloader = DataLoader(check_dataset, sampler=check_sampler, batch_size=1, num_workers=0)

    model.to(device)
    y_preds = []
    all_vul_lines = []
    orig_funcs = []
    model.eval()
    for batch in check_dataloader:
        inputs_ids =  batch[0].to(device)
        func = batch[1]
        with torch.no_grad():
            logit, attentions = model(input_ids=inputs_ids, output_attentions=True)
            pred = logit.cpu().numpy()[0][1] > best_threshold
            if pred:
                vul_lines = find_vul_lines(tokenizer, inputs_ids, attentions)
                y_preds.append(1)
            else:
                vul_lines = [None]
                y_preds.append(0)
                
            all_vul_lines.append(vul_lines[:10])
            #y_preds.append(pred)
            orig_funcs.append(func)
    if do_linelevel_preds:
        result = {'methods': orig_funcs, 'vulnerable': y_preds, 'vul_lines': all_vul_lines}
        return result
    else:
        result = {'methods': orig_funcs, 'vulnerable': y_preds}
        return result
    

def main():
    base = 'microsoft/unixcoder-base' # 'microsoft/unixcoder-base' 'microsoft/codebert-base'
    model_id = "saved_models" #'/kaggle/input/linevul-adapter' '/kaggle/input/unixcoder-adapter'

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

    df = pd.read_csv('https://www.dropbox.com/s/zma3l4arxehtwvx/Test_dataset_c%23.csv?dl=1')
    func = [df['Snippet'][1]]
    func.append(df['Snippet'][2])
    result = predict(model, tokenizer, func, device, do_linelevel_preds = True)
    print(result['methods'][0])
    print(result['vulnerable'][0])
    print(result['vul_lines'][0])

if __name__ == "__main__":
    main()
