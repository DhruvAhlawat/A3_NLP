
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
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device);
begin_training = time.time();
time_limit = 9*3600; #nine hours for training.

# %%
model_wildcard = "google/flan-T5-base";
model = AutoModelForSeq2SeqLM.from_pretrained(model_wildcard, load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_wildcard);

# %%
import json
import os
train_datasets = [0,0];
input_dir = sys.argv[1]; #path to data.
output_dir = sys.argv[2]; #path to save
PLOS_train = os.path.join(input_dir, "PLOS_train.jsonl");
eLife_train = os.path.join(input_dir, "eLife_train.jsonl"); #loading the train datasets of both here.
with open(PLOS_train) as f:
   train_datasets[0] = [json.loads(line) for line in f];
with open(eLife_train) as f:
   train_datasets[1] = [json.loads(line) for line in f];

# %%
max_inp_len_PLOS = 1024;
max_inp_len_eLife = 1024;
max_sum_len_PLOS = 320;
max_sum_len_eLife = 530;

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
#     with open("PLOS_processed_dataset.pkl", "wb") as f:
#         pickle.dump(PLOS_processed_dataset, f);
#     with open("eLife_processed_dataset.pkl", "wb") as f:
#         pickle.dump(eLife_processed_dataset, f); #there is no real need to dump them tbh. Can keep in memory if memory is large enough.
    print("time taken for preprocessing", name, " = ", end - start, end =  "                       \n")

# %%
#loading the model and getting it ready to be trained via LoRA.
# Define LoRA Config 
lora_config = LoraConfig(
 r=64, 
 lora_alpha=128,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="lora_only",
 task_type=TaskType.SEQ_2_SEQ_LM
)
# model.gradient_checkpointing_enable() #setting up gradient checkpointing for memory.
# prepare int-8 model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

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

def load_model(checkpoint_name, training=1, optimizer=0):
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

# %%
all_dataloader = DataLoader(biodataset(all_processed_dataset,max_sum_len_eLife, max_inp_len_PLOS), batch_size = 32, shuffle=True);

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# %%
model = model.to(device);
model.train();
optimizer = AdamW(model.parameters(), lr=0.0001); #a higher LR since batch size is more. but properly scheduling it.
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
#exponential runnning loss.

# %%
name = "FINAL_MODEL"
import os
save_path = os.path.join(output_dir, name)

# %%
alpha = 0.96;
total_size = len(all_dataloader);
try:
    for epoch in range(50):
        running_loss = 0;
        total = 1;
        start = time.time();
        print("starting epoch ", epoch);
        reduction = 1.0005;
        if(epoch == 1):
            reduction = 1.0004;
        if(epoch == 2):
            reduction = 1.0003;
        if(epoch >= 3):
            reduction = 1.0002;
        for inp, out in all_dataloader:
            elapsed = time.time() - begin_training;
            # if(time_limit - elapsed < 200): #if less than 10 minutes remaining then we break the training loop.
            #     break;
            for g in optimizer.param_groups:
                g['lr'] = g['lr']/reduction; #effectively reduces by a factor of 3 every epoch.
            optimizer.zero_grad() #bruh we have to zero grad at the start obviously.
            input_ids = inp.to(device);
            output_ids = out.to(device);
            outputs = model(input_ids, labels=output_ids);
            loss = outputs[0];
            loss.backward();
            optimizer.step();
            total += 1;
            running_loss = running_loss * (alpha) + (1-alpha)*loss.item() if total > 1 else loss.item();
            left = total_size - total;
            curtime = time.time() - start;
            time_per_batch = curtime/total;
            predicted_time = left * time_per_batch;
            hours = predicted_time//3600; mins = (predicted_time - hours*3600)//60;
            if(total % 5 == 0):
                print(epoch, "curloss:", loss.item(), "|running loss:", running_loss, "done:", total, "per:", int(100*time_per_batch)/100,"s, pred: ", int(hours), "h", int(mins), "m");
            if(total % 200 == 0):
                save_model(model, tokenizer, optimizer, save_path);
            clear_gpu_cache()
        save_model(model, tokenizer,optimizer, save_path);
        print("epoch: ", epoch, " loss is : ", running_loss/total);
except Exception as e:
    print("exception caught in training: \n", e);

# %%
# lora_model.eval()
# #ready for evaluation after this point.
# #evaluate using the following method
# test_batch_size = 2;
# test_loader = DataLoader(biodataset(PLOS_processed_dataset,max_sum_len_PLOS, max_inp_len_PLOS), batch_size = test_batch_size, shuffle=False);
# ind = 0;
# for inp, out in test_loader:
#     input_ids = inp.to(device);
#     output_ids = out.to(device);
#     outputs = lora_model.generate(input_ids=input_ids, max_new_tokens=530, do_sample=True, top_p=0.9)
#     for j in range(len(outputs)):
#         result = tokenizer.batch_decode(outputs.detach().cpu().numpy())[j];
#         actual = tokenizer.batch_decode(out, skip_special_tokens=True)[j];
#         print("summary: ",len(result), '\n', result);
#         print("actual: ", len(actual), '\n', actual);
#         break;
#     ind += test_batch_size;

# %%



