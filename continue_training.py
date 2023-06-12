import json
import time
from tqdm import tqdm
import argparse
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch
import os
from itertools import chain
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import DataParallel
import numpy as np
from datetime import datetime
import transformers
import random

seeds=(1234)

random.seed(seeds)
np.random.seed(seeds)
torch.manual_seed(seeds)

train_path = "CMV/cmv_train.jsonl"
is_mtl = "true"
dev_path = "CMV/cmv_dev.jsonl"
test_path = "CMV/cmv_test.jsonl"

class ArgenPlanS2sDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_len=512, batch_first=True, is_mtl=True):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.is_mtl = is_mtl
        self.examples = self.load_data()
        self.batch_first = batch_first
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id

    def process(self, line_json, with_eos=True):   
        if line_json["task"] == "revise_shuffle" or line_json["task"] == "revise_kp":
            context = random.sample(line_json["input_list"], 1)[0]
            response = line_json["output"]
        elif line_json["task"] == "distingush":
            flag_int = random.randint(0, 9)
            if flag_int <= 4:  # positive
                context = line_json["pos"]["input"]
                response = line_json["pos"]["output"]
            else:
                sample_res = random.sample(line_json["neg_list"], 1)[0]
                context = sample_res["input"]
                response = sample_res["output"]
        else:
            context = line_json["input"]
            response = line_json["output"]

        src_id = self.tokenizer.encode(context)
        tgt_id = self.tokenizer.encode(response)

        instance = {}
        instance["input_ids"] = src_id[:self.max_len]
        instance["lm_labels"] = tgt_id[:self.max_len]
        instance["original_json"] = line_json
        return instance

    def load_data(self):
        data = [json.loads(ln) for ln in open(self.data_path).readlines()]
        data_filter = data
        if not self.is_mtl:
            data_filter = [elem for elem in data_filter if elem["task"] == "generation"]

        print("used task: ", set([elem["task"] for elem in data_filter]))

        return data_filter

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        sample = self.examples[index]
        instance = self.process(sample)
        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad_id)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad_id)
        original_json = [instance["original_json"] for instance in batch]
        return input_ids, labels, original_json
    

def eval_model_loss(model, tokenizer, eval_dataloader, device, epoch_id):
    tot_loss = []
    tot_sample = []
    tot_ppl = []
    with torch.no_grad():
        for step, cur_batch in enumerate(eval_dataloader):
            batch_input_ids = cur_batch[0].to(device)
            batch_labels = cur_batch[1].to(device)
            batch_context_mask = (batch_input_ids != tokenizer.pad_token_id).long()

            outputs = model.forward(
                input_ids=batch_input_ids,
                attention_mask=batch_context_mask,
                labels=batch_labels,
            )

            loss = outputs["loss"]
            ppl = torch.exp(loss)
            
            n_sample = batch_input_ids.shape[0]
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)
    print("eval: ", len(tot_loss))
    print(f"Step {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)}")
    return np.sum(tot_loss) / np.sum(tot_sample)


def generate(batch_size, model_path, test_path, write_path, max_length, fp16=False, fp16_opt_level='O1'):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('using device:', device)

    # print("load model from: ", model_path)
    # tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    # model = T5ForConditionalGeneration.from_pretrained(model_path)
    # model.to(device)
    model.eval()

    print("load data from: ", test_path)
    print("write results to: ", write_path)
    print("decoding max length: ", max_length)

    f_w = open(write_path, "w")

    eval_data = ArgenPlanS2sDataset(
        tokenizer=tokenizer, 
        data_path=test_path, 
        batch_first=True,
        is_mtl=False
    )
    eval_dataloader = DataLoader(
        eval_data, 
        batch_size=batch_size, 
        drop_last=False, 
        collate_fn=eval_data.collate, 
        shuffle=False,
        num_workers=8
    )

    print("number of test: {}".format(eval_data.__len__()))



    print('starting decoding')
    for cur_epoch in range(1):
        for batch_step, cur_batch in enumerate(eval_dataloader):
            batch_input_ids = cur_batch[0].to("cuda")
            batch_labels = cur_batch[1].to("cuda")
            batch_context_mask = (batch_input_ids != tokenizer.pad_token_id).long()

            batch_original_json = cur_batch[2]

            # topp-sampling
            outputs = model.generate(
                input_ids=batch_input_ids, 
                attention_mask=batch_context_mask,
                max_length=max_length, 
                do_sample=True,
                top_k=10,
                top_p=0.9,
                use_cache=True,
            )

            for i in range(outputs.size()[0]):
                cur_gen = tokenizer.decode(outputs[i])
                cur_ref = tokenizer.decode(batch_labels[i])
                cur_input = tokenizer.decode(batch_input_ids[i])
                cur_json = batch_original_json[i]
                cur_json["input_src"] = cur_input.replace("<pad>", "").strip()
                cur_json["gen"] = cur_gen.replace("<pad>", "").strip()
                f_w.write(json.dumps(cur_json, ensure_ascii=False) + "\n")

    print("finish decoding")



import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


if torch.cuda.is_available():
    dev = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

print(dev)


output_dir = 'Data'
tokenizer_path="t5-base"
model_path=output_dir + '/best_eval'

tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.to(device)

# optimizer = Adafactor(
#     model.parameters(),
#     lr=1e-3,
#     eps=(1e-30, 1e-3),
#     clip_threshold=1.0,
#     decay_rate=-0.8,
#     beta1=None,
#     weight_decay=0.0,
#     relative_step=False,
#     scale_parameter=False,
#     warmup_init=False
# )


train_data = ArgenPlanS2sDataset(tokenizer=tokenizer, data_path=train_path, batch_first=True, is_mtl=is_mtl)
eval_data = ArgenPlanS2sDataset(tokenizer=tokenizer, data_path=dev_path, batch_first=True, is_mtl=is_mtl)
train_dataloader = DataLoader(train_data, batch_size=1, drop_last=True, collate_fn=train_data.collate, shuffle=True, num_workers=2)
eval_dataloader = DataLoader(eval_data, batch_size=1, drop_last=True, collate_fn=eval_data.collate, shuffle=True,num_workers=2)
if is_mtl:
        eval_gen_data = ArgenPlanS2sDataset(
            tokenizer=tokenizer, 
            data_path=dev_path, 
            batch_first=True,
            is_mtl=False
        )
        eval_gen_dataloader = DataLoader(
            eval_gen_data, 
            batch_size=1, 
            drop_last=True, 
            collate_fn=eval_data.collate, 
            shuffle=True,
            num_workers=4
        )


print('starting training')
running_loss = 0
running_ppl = 0

current_step = 0
cur_epoch = 0
best_eval_loss = 10000
best_eval_gen_loss = 100000
save_step = 2000
max_length = 200


for cur_epoch in range(2):
    for batch_step, cur_batch in enumerate(train_dataloader):
        current_step += 1
        batch_input_ids = cur_batch[0].to(dev)
        batch_labels = cur_batch[1].to(dev)
        batch_context_mask = (batch_input_ids != tokenizer.pad_token_id).long()

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_context_mask,
            labels=batch_labels,
        )

        loss = outputs.loss
        loss_num=loss.item()
        # note: this is only estimated value for ppl: needed to imporved
        ppl = torch.exp(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        print(cur_epoch, loss_num, batch_step)
        
        # if save_step != -1 and  current_step % save_step == 0:
        #     print("start evaluation")
        #     model.eval()
        #     eval_loss = eval_model_loss(model, tokenizer, eval_dataloader, dev, current_step) 
        #     if is_mtl:
        #         eval_gen_loss = eval_model_loss(model, tokenizer, eval_gen_dataloader, dev, current_step) 
        #     model.train()
        
        #     print('saving model for step {}'.format(current_step))
        #     if not os.path.exists(output_dir + '/model_step{}'.format(current_step)):
        #         os.mkdir(output_dir + '/model_step{}'.format(current_step))
        #     model_to_save = model.module if hasattr(model, 'module') else model
        #     model_to_save.save_pretrained(output_dir + '/model_step{}'.format(current_step))

        #     if eval_loss < best_eval_loss:
        #         print("save best.....")
        #         if not os.path.exists(output_dir + '/best_eval'):
        #             os.mkdir(output_dir + '/best_eval')
        #         model_to_save = model.module if hasattr(model, 'module') else model
        #         model_to_save.save_pretrained(output_dir + '/best_eval')
        #         best_eval_loss = eval_loss
            
        #     if is_mtl:
        #         if eval_gen_loss < best_eval_gen_loss:
        #             print("save best gen model.....")
        #             if not os.path.exists(output_dir + '/best_eval_gen'):
        #                 os.mkdir(output_dir + '/best_eval_gen')
        #             model_to_save = model.module if hasattr(model, 'module') else model
        #             model_to_save.save_pretrained(output_dir + '/best_eval_gen')
        #             best_eval_gen_loss = eval_gen_loss
    
    if not os.path.exists(output_dir + '/best_eval'):
        os.mkdir(output_dir + '/best_eval')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + '/best_eval')
    best_eval_loss = eval_loss
    
    print('Model Saved for Epoch', cur_epoch)
                    

generate(
            batch_size=150, 
            model_path=output_dir + '/best_eval', 
            test_path=test_path,
            write_path=output_dir + "/cmv_output_best.jsonl", 
            max_length=max_length,
            )