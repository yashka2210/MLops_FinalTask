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


class InputFeatures(object):
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 original_func,
                 cwe):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.original_func=original_func
        self.cwe=cwe

        
class TextDataset(Dataset):
    def __init__(self, tokenizer, train_data_file=None, eval_data_file=None, test_data_file=None,  file_type="train"):
        if file_type == "train":
            file_path = train_data_file
        elif file_type == "eval":
            file_path = eval_data_file
        elif file_type == "test":
            file_path = test_data_file
        self.examples = []
        df = pd.read_csv(file_path)
        funcs = df["Snippet"].tolist()
        labels = df["Target"].tolist()
        cwes = df["CWE"].tolist()
        for i in tqdm(range(len(funcs))):
            self.examples.append(convert_examples_to_features(funcs[i], labels[i], tokenizer, cwes[i]))
        if file_type == "train":
            for example in self.examples[:3]:
                    print("*** Example ***")
                    print("label: {}".format(example.label))
                    print("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    print("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label), str(self.examples[i].original_func), str(self.examples[i].cwe)

    
def cleaner(code):
    ## Remove code comments
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    code = re.sub('\r','',code)
    code = re.sub('\t','',code)
    code = code.split('\n')
    code = [line.strip() for line in code if line.strip()]
    code = '\n'.join(code)
    return(code)


def convert_examples_to_features(func, label, tokenizer, cwe, block_size=512):
    clean_func = cleaner(func)
    code_tokens = tokenizer.tokenize(str(clean_func))[:block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label, func, cwe)


def set_seed(n_gpu, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def train(model, tokenizer, train_dataset, eval_dataset, device, lr=3e-4, train_batch_size=1, eval_batch_size=1, num_epochs=1):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, num_workers=0)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=eval_batch_size,num_workers=0)

    optimizer = AdamW(params=model.parameters(), lr=lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    n_gpu = torch.cuda.device_count()
    
    model.zero_grad()
    for epoch in range(num_epochs):
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        model.train()
        print(f'Epoch: {epoch+1}, num epochs: {num_epochs}')
        for step, batch in enumerate(bar):
            inputs_ids =  batch[0].to(device)
            labels = batch[1].to(device)
            loss, outputs = model(input_ids=inputs_ids, labels=labels)
            if n_gpu > 1:
                    loss = loss.mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            bar.set_description("epoch {} loss {}".format(epoch,loss))

        model.eval()
        logits = []
        y_trues = []
        eval_loss = 0
        for step, batch in enumerate(tqdm(eval_dataloader,total=len(eval_dataloader))):
            with torch.no_grad():
                inputs_ids =  batch[0].to(device)
                labels = batch[1].to(device)
                lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                y_trues.append(labels.cpu().numpy())

        logits = np.concatenate(logits,0)
        y_trues = np.concatenate(y_trues,0)
        best_threshold = 0.5
        best_f1 = 0
        y_preds = logits[:,1]>best_threshold
        recall = recall_score(y_trues, y_preds)
        precision = precision_score(y_trues, y_preds)   
        f1 = f1_score(y_trues, y_preds)             
        result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
            "eval_threshold": best_threshold,
            "eval_loss": eval_loss
        }

        print("***** Eval results *****")
        for key in  sorted(result.keys()):
            print(key, str(round(result[key],4)))
        if result['eval_f1']>best_f1:
            best_f1=result['eval_f1']
            print("  "+"*"*20)  
            print("  Best f1:%s",round(best_f1,4))
            print("  "+"*"*20)                          
            output_dir = os.path.join('saved_models')                      
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            model_to_save.save_pretrained(output_dir, from_pt=True)
            print("Saving model to ", output_dir)
    

def main():
    # pretrain_model_path = "/kaggle/working/12heads_linevul_model.bin"
    base = 'microsoft/unixcoder-base' # 'microsoft/unixcoder-base' 'microsoft/codebert-base'
    train_data_file = 'data/train.csv'
    eval_data_file = 'data/val.csv'

    tokenizer = AutoTokenizer.from_pretrained(base)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    set_seed(n_gpu)
    
    config = RobertaConfig.from_pretrained(base)
    config.num_labels = 1         
    model = RobertaForSequenceClassification.from_pretrained(base, config=config, ignore_mismatched_sizes=True).to(device)
    model = Model(model, config, tokenizer)
    model.to(device)
    
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    train_dataset = TextDataset(tokenizer, train_data_file = train_data_file, file_type='train')
    eval_dataset = TextDataset(tokenizer, eval_data_file = eval_data_file, file_type='eval')
    train(model, tokenizer, train_dataset, eval_dataset, device)


if __name__ == "__main__":
    main()
