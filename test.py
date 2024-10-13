
# %%
import numpy as np
import pandas as pd
import peft
from peft import LoraConfig, get_peft_model
from transformers import BioGptTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, BioGptModel, BioGptConfig, BioGptForCausalLM, pipeline, set_seed
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json
import os
import transformers
import time
import pickle
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
from torch.utils.data import DataLoader
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
training = 0;
import gc
from torch.optim import AdamW
from datasets import Dataset
import json
import sys
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import zipfile
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device);
begin_training = time.time();
time_limit = 9*3600; #nine hours for training.

def load_model(checkpoint_name, training=0, optimizer=0):
#     checkpoint_name = "/kaggle/input/checkpoints/total_1800";
    config = PeftConfig.from_pretrained(checkpoint_name);
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    if(training != 0):
        model = prepare_model_for_kbit_training(model); 
    model = PeftModel.from_pretrained(model, checkpoint_name, device_map={"":0}, is_trainable=(training!=0)); #since we want to train it further.
    if(optimizer > 0):
        optimizer = AdamW(model.parameters(), lr=0.00005); #a lower LR for now.
        optimizer.load_state_dict(torch.load(checkpoint_name + "_optimizer"))
    return model, tokenizer, optimizer

# %%
model_wildcard = "google/flan-T5-base";
model = AutoModelForSeq2SeqLM.from_pretrained(model_wildcard, load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_wildcard);

# %%
train_datasets = [0,0];
input_dir = sys.argv[1]; #path to data.
model_dir = sys.argv[2];
model_path = os.path.join(model_dir, "FINAL_MODEL");
output_dir = sys.argv[3]; #path to save
PLOS_train = os.path.join(input_dir, "PLOS_test.jsonl");
eLife_train = os.path.join(input_dir, "eLife_test.jsonl"); #loading the train datasets of both here.
with open(PLOS_train) as f:
   train_datasets[0] = [json.loads(line) for line in f];
with open(eLife_train) as f:
   train_datasets[1] = [json.loads(line) for line in f];
# %%
max_inp_len_PLOS = 1024;
max_inp_len_eLife = 1024;
max_sum_len_PLOS = 320;
max_sum_len_eLife = 530;
model, tokenizer, optimizer = load_model()
# %%
def preprocess(example, name, index, max_inp_len, max_sum_len):
    inputs = "summarize the following biological scientific article : " + example["article"] + "\n summarize the above article.";
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=max_inp_len);
    try:
        tokenized_summaries = tokenizer(text_target=example["lay_summary"], padding="max_length", truncation=True, max_length=max_sum_len); 
        tokenized_inputs["output_ids"] = tokenized_summaries["input_ids"];
    except:
        print("no lay summary present. Skipping");
    tokenized_inputs["id"] = example["id"]; #storing the id as well just in case.
    tokenized_inputs["headings"] = example["headings"];
    tokenized_inputs["keywords"] = example["keywords"];
    tokenized_inputs["index"] = (name, index); #stores the name of the dataset and the index of the example incase we need to see additional things later.
    # tokenizer(text_target)
    return tokenized_inputs;

PLOS_processed_dataset = [];
eLife_processed_dataset = [];
num_preprocess = 2;
for j in range(num_preprocess):
    start = time.time();
    name = "PLOS_processed_dataset" if j == 0 else "eLife_processed_dataset";
    total_len = len(train_datasets[j]);
    print("preprocessing ", name);
    for i in range(total_len):
        if(j == 0):
            cur = preprocess(train_datasets[j][i], name, i, max_inp_len=max_inp_len_PLOS, max_sum_len=max_sum_len_PLOS);
            PLOS_processed_dataset.append(cur);
        else:
            cur = preprocess(train_datasets[j][i], name, i, max_inp_len=max_inp_len_eLife, max_sum_len=max_sum_len_eLife);
            eLife_processed_dataset.append(cur);
        if(i%10 == 0):
            print("preprocessing done with ", i, " out of ", total_len, " = ", i/total_len, "%, time = ", time.time() - start);
    end = time.time();
    print("time taken for preprocessing", name, " = ", end - start, end =  "                       \n")

# %%
#loading the model and getting it ready to be trained via LoRA.


# %%
# with open("/kaggle/input/preprocessed/PLOS_processed_dataset.pkl", "rb") as f:
#     PLOS_processed_dataset = pickle.load(f);
# with open("/kaggle/input/preprocessed/eLife_processed_dataset.pkl", "rb") as f:
#     eLife_processed_dataset = pickle.load(f); #pickle.dump(eLife_processed_dataset, f);

# %%
def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()

# %%
import random
all_processed_dataset = PLOS_processed_dataset + eLife_processed_dataset
random.shuffle(all_processed_dataset) #randomly shuffling it around once.

# %%
def save_model(model, tokenizer, optimizer, name):
    model.save_pretrained(name);
    tokenizer.save_pretrained(name);
    torch.save(optimizer.state_dict, name+"_optimizer");
#     optimizer.save_state_dict(name+"_optimizer")

try:
    model, tokenizer, optimizer = load_model(model_dir, training=0, optimizer=0);
except:
    print("model not found. Must have been zipped");
    path = os.path.join(model_dir, "FINAL_MODEL.zip");
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(model_dir);
    model, tokenizer, optimizer = load_model(model_dir, training=0, optimizer=0);
    pass
# %%
class biodataset(torch.utils.data.Dataset):
    def __init__(self, data, sum_max, inp_max):
        self.data = data;
        self.sum_max = sum_max;
        self.inp_max = inp_max;
    def __len__(self):
        return len(self.data); #should be a very high number, right.
    def __getitem__(self, idx):
        #here we will pad it ourselves without needing another function to do it.
        pad_index = tokenizer.pad_token_id;
        pad_len = max(0, self.inp_max - len(self.data[idx]['input_ids']));
        pad_sum_len = max(0, self.sum_max - len(self.data[idx]['output_ids']));
        inp = torch.tensor(self.data[idx]['input_ids'][:self.inp_max] + [pad_index] * pad_len);
        out = torch.tensor(self.data[idx]['output_ids'][:self.sum_max] + [pad_index] * pad_sum_len);
        return inp, out

class biodataset_test(torch.utils.data.Dataset):
    def __init__(self, data, sum_max, inp_max):
        self.data = data;
        self.sum_max = sum_max;
        self.inp_max = inp_max;
    def __len__(self):
        return len(self.data); #should be a very high number, right.
    def __getitem__(self, idx):
        #here we will pad it ourselves without needing another function to do it.
        pad_index = tokenizer.pad_token_id;
        pad_len = max(0, self.inp_max - len(self.data[idx]['input_ids']));
        inp = torch.tensor(self.data[idx]['input_ids'][:self.inp_max] + [pad_index] * pad_len);
        return inp
# %%
# %%
try:
    model = model.to(device);
except:
    print("model to device err");
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# %%
import re
def post_processor(input_string):
    # Use a regular expression to replace all tags with an empty string
    cleaned_string = re.sub('<.*?>', '', input_string)
    cleaned_string = re.sub('\n', ' ', cleaned_string) #to ensure no new lines.
    return cleaned_string

def evaluate_and_save_predictions(model, tokenizer, test_loader, file_path, max_new_tokens=550, top_p = 0.9):
    model.to(device)
    model.eval()
    total_len = len(test_loader);
    print("starting evaluation with top_p = ", top_p);
    with open(file_path, 'w') as f:
        done = 0;
        start = time.time();
        for inp in test_loader:
            input_ids = inp.to(device)
            outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9)
            all_results = tokenizer.batch_decode(outputs.detach().cpu().numpy());
            for j in range(len(outputs)):
                result = post_processor(all_results[j]);
                f.write(result + "\n");
            done += 1;
            time_per = (time.time() - start)/done;
            print("done", done, "out of ", total_len, "time per: ",time_per, "left = ", ((total_len - done)*time_per)//60);
    print(f"Predictions saved to {file_path}")

PLOS_loader = DataLoader(biodataset_test(PLOS_processed_dataset, max_sum_len_PLOS, max_inp_len_PLOS), batch_size=8, shuffle=False);
PLOS_path = os.path.join(output_dir, "plos.txt");
evaluate_and_save_predictions(model, tokenizer, PLOS_loader,PLOS_path, 400, 0.77);
eLife_loader = DataLoader(biodataset_test(eLife_processed_dataset, max_sum_len_eLife, max_inp_len_eLife), batch_size=8, shuffle=False);
eLife_path = os.path.join(output_dir, "elife.txt");
evaluate_and_save_predictions(model, tokenizer, eLife_loader,eLife_path, 600, 0.77);
# %%
