import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer
from torch.nn import functional as F
from datasets import load_dataset, Features, Value
from transformers import AutoConfig


class Load_Model():
    def __init__(self, plm:str, revision:str, path:str):
        print('---LODING MODEL---')
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
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.data_list = None
        self.result = None
        print('FINISHED')


    def load_data(self, path:str):
        dataset = load_dataset("csv", data_files=path,
                            delimiter="\t",
                            features=Features({
                                'fid': Value('string'),
                                'idx': Value('string'),
                                'content': Value('string'),
                            }),
                            column_names=['fid', 'idx', 'content'],
                            keep_default_na=False
                            )
        self.data_list = (list(dataset['train']))


    def predict_one(self, input:str, max_new_tokens = 200):
        self.model.eval()
        prompt = input
        tks_info = self.tokenizer(prompt)
        text = tks_info['input_ids']
        inputs, past_key_values = torch.tensor([text]), None
        outputs = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.model(inputs.to(self.device), past_key_values = past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values
                log_probs = F.softmax(logits[:, -1], dim = -1)
                inputs = torch.argmax(log_probs, 1).unsqueeze(0)
                if self.tokenizer.decode(inputs.item()) == self.eos :
                    break
                text.append(inputs.item())
            pred = self.tokenizer.decode(text)
            pred = pred[pred.find(self.sep) + len(self.sep):].replace(self.sep, "").replace(self.eos, "").strip()
            print(pred)
            if pred == self.eos:
                return outputs
            phis = pred.split('\n')
            return phis


    def predict_all(self):
        print("---RUNNING PREDICTION(ALL DATA)---")
        self.result = [self.predict_one(input = f"{self.bos} {sample['content']} {self.sep}", max_new_tokens = 100) for sample in self.data_list]
        print('FINISHED')


    def save_original_result(self, path):
        print('---Saving---')
        with open(path, "w", encoding = "utf-8") as file:
            for idx, line in enumerate(self.result):
                file.write(self.data_list[idx]['fid']+'\t'+self.data_list[idx]['idx']+'\t'+self.data_list[idx]['content']+'\t'+line[0] + "\n")
        print(f'SAVED IN : {path}')


if __name__ == '__main__':
    plm = "EleutherAI/pythia-1b-deduped"
    revision = "step3000"
    model_path = './model.pt'
    data_path = './data.tsv'
    save_path = './reslut.tsv'
    model = Load_Model(plm = plm, revision = revision, path = model_path)
    model.load_data(path = data_path)
    # a = model.predict_one(input = '11.38am on 28/2/13',  max_new_tokens = 100)
    # print(a)
    model.predict_all()
    model.save_original_result(save_path)