import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from torch.utils.data import DataLoader
from datasets import load_dataset, Features, Value
from torch.optim import AdamW
from tqdm import trange

class BatchSampler():
    def __init__(self, data, batch_size):
        self.pooled_indices = []
        self.data = data
        self.batch_size = batch_size

        self.len = len(list(data))
    def __iter__(self):
        self.pooled_indices = []
        indices = [(index, len(data['content'])) for index, data in enumerate(self.data)]
        random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size * 100): #batch_size * 100 can adjust number to reduce RAM
            self.pooled_indices.extend(sorted(indices[i: i + self.batch_size * 100], key = lambda x: x[1], reverse = True))
        self.pooled_indices = [x[0] for x in self.pooled_indices]
        
        for i in range(0, len(self.pooled_indices), self.batch_size):
            yield self.pooled_indices[i: i + self.batch_size]
    def __len__(self):
        return (self.len + self.batch_size - 1) // self.batch_size

class LLM_Model(BatchSampler):
    def __init__(self, plm:str, revision:str):
        self.bos = '<|endoftext|>'
        self.eos = '|END|'
        self.pad = '|pad|'
        self.sep = '\n\n####\n\n'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.tokenizer = AutoTokenizer.from_pretrained(plm, revision=revision, bos_token=self.bos, eos_token=self.eos, pad_token=self.pad, sep_token=self.sep)
        self.tokenizer = AutoTokenizer.from_pretrained(plm, revision=revision)
        self.special_token_dict = {'eos_token' : self.eos, 'bos_token' : self.bos, 'pad_token': self.pad, 'sep_token': self.sep}
        num_add_toks = self.tokenizer.add_special_tokens(self.special_token_dict)
        self.config = AutoConfig.from_pretrained(plm,
                                                bos_token_id = self.tokenizer.bos_token_id,
                                                eos_token_id = self.tokenizer.eos_token_id,
                                                pad_token_id = self.tokenizer.pad_token_id,
                                                sep_token_id = self.tokenizer.sep_token_id,
                                                output_hidden_states = False
                                                 )
        self.model = AutoModelForCausalLM.from_pretrained(plm, revision = revision, config = self.config)
        self.data_list = None
        
         
    def load_data(self, path:str):
        dataset = load_dataset("csv", data_files=path,
                            delimiter="\t",
                            features=Features({
                                'fid': Value('string'),
                                'idx': Value('string'),
                                'content': Value('string'),
                                'label': Value('string')
                            }),
                            column_names=['fid', 'idx', 'content', 'label'],
                            keep_default_na=False
                            )
        self.data_list = (list(dataset['train']))
        
        
    def __collate_batch(self, batch ,IGNORED_PAD_IDX = -100):
        text = [f"{self.bos} {sample['content']} {self.sep} {sample['label']} {self.eos}" for sample in batch]
        encoded_seq = self.tokenizer(text, padding = True)
        indexed_tks = torch.Tensor(encoded_seq['input_ids']).long()
        attention_mask = torch.Tensor(encoded_seq['attention_mask']).long()
        encoded_label = torch.Tensor(encoded_seq['input_ids']).long()
        encoded_label[encoded_label == self.tokenizer.pad_token_id] = IGNORED_PAD_IDX
        return indexed_tks, encoded_label, attention_mask
    
    
    def train(self, epoch:int, batch_size:int, path:str, trigger:float):
        train_dataloader = DataLoader(self.data_list, batch_size = 2, shuffle = False , collate_fn = self.__collate_batch)
        bucket_train_dataloader = DataLoader(self.data_list, batch_sampler = BatchSampler(self.data_list, batch_size),
                                     collate_fn = self.__collate_batch, pin_memory = True)
        tks , labels, masks = next(iter(train_dataloader))
        optimizer = AdamW(self.model.parameters(), lr = 5e-5)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.train()
        last_loss = 100
        trigger_times = 0
        for _ in trange(epoch, desc = "Epoch"):
            self.model.train()
            total_loss = 0
            for step, (seqs, labels, masks) in enumerate(bucket_train_dataloader):
                seqs = seqs.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)
                self.model.zero_grad()
                outputs = self.model(seqs, labels = labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_train_loss = total_loss / len(bucket_train_dataloader)
            torch.save(self.model.state_dict(), path)
            print("\nModel Saved")
            #ealry stop
            diff = last_loss - avg_train_loss
            if (diff > trigger):
                last_loss = avg_train_loss
                print("average train loss : {}".format(avg_train_loss))
                continue   
            else:
                print('Early Stop Triggering')
                trigger_times += 1
                if trigger_times == 3 :
                    break
                else:
                    continue
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
if __name__ == '__main__':
    plm = "EleutherAI/pythia-70m-deduped"
    revision = "step3000"
    load_path = 'train.tsv'
    save_path = '70m.pt'
    model = LLM_Model(plm = plm, revision = revision)
    model.load_data(path = load_path)
    model.train(epoch = 1000, batch_size = 32, path = save_path, trigger = 0.03)
    model.save(path = save_path)