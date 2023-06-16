import re
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, RobertaConfig, get_linear_schedule_with_warmup, set_seed
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
from peft import PeftModel, PeftConfig
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.optim import AdamW
# import evaluate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os 
from scipy.special import softmax
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from model import Model
from train import InputFeatures, TextDataset, cleaner, convert_examples_to_features, set_seed

def test(model, tokenizer, data, device, best_threshold = 0.5):  
    test_sampler = SequentialSampler(data)
    test_dataloader = DataLoader(data, sampler=test_sampler, batch_size=1, num_workers=0)
    eval_loss = 0.0
    model.eval()
    logits=[]  
    y_trues=[]
    orig_funcs = []
    bar = tqdm(test_dataloader,total=len(test_dataloader))
    for batch in bar:
        inputs_ids =  batch[0].to(device)
        labels = batch[1].to(device)
        funcs = batch[2]
        orig_funcs += funcs
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels = labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    print('vulnerable:', y_preds)
    logits = [l[1] for l in logits]
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    confusion = confusion_matrix(y_true=y_trues, y_pred=y_preds)
    tn, fp, fn, tp = confusion.ravel()
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold": best_threshold,
        "test_loss": eval_loss,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }
    print("***** Test results *****")
    print('Confusion matrix: \n',confusion)
    for key in  sorted(result.keys()):
        print(key, str(round(result[key],4)))
    return result

def main():
    base = 'microsoft/unixcoder-base' # 'microsoft/unixcoder-base' 'microsoft/codebert-base'
    model_id = "saved_models"
    test_data_file = 'data/test.csv'

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
    test_dataset = TextDataset(tokenizer, test_data_file = test_data_file, file_type='test')
    result = test(model, tokenizer, test_dataset, device)


if __name__ == "__main__":
    main()