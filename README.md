
# 隱私保護與醫學數據標準化競賽:解碼臨床病例、讓數據說故事競賽 TEAM_4145
  
   以下程式碼是提供於競賽審查使用，接下來將介紹程式碼。
  
# 環境需求
  
   本次競賽過程中使用python3.8.10版本以及torch、transformer、tqdm、datasets、random、re套件，下方提供程式碼以及版本參考:  
  pytorch == 2.0.1  
 ```python
!pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
transformer==4.35.0
```python
!pip install transformer == 4.35.0
```
tqdm ==4.66.1
```python
!pip install datasets == 2.14.6
```

	 

## train.py檔案
  
train.py檔案為訓練模型檔案，其為訓練本次競賽中模型所使用的程式。  
  
訓練過程中所要求訓練資料形式為:  
檔案名稱\t起始位置\t內容\t標籤  
  
以下為主要設定參數:  
plm 代表「預訓練語言模型」(String)  
revision 為指定模型的特定版本 (String)  
epoch  為訓練回合數 (int)  
batch_size  數據樣本的數量 (int)   
path 模型儲存位置 (String)  
trigger 每回合依照損失函數遞減程度觸發Early Stopping 閥門 (float)    
time 為Early Stopping觸發次數，達到指定觸發次數將停止訓練 (int)  
  
本次競賽小組選擇使用的參數為  
plm = 'EleutherAI/pythia-1B-deduped'  
 revision = ' step3000'  
epoch  =  1000  
batch_size  =  32  
path  =  './test.pt'  
trigger =  0.03  
time  =  5  
  
主要程式碼:  
LLM_Model(self, plm:str, revision :str) 物件，並且讀取預訓練模型  
範例:  
```python
model = LLM_Model(plm = 'EleutherAI/pythia-1B-deduped', revision = "step3000")
```
--------------------------------------------------------
LLM_Model.load_data(self, path: str)讀取.tsv檔案  
範例:  
```python
LLM_Model.load_data(path = './test.tsv')
```
--------------------------------------------------------
LLM_Model.train(self, epoch:int, batch_size:int, path:str, trigger:float, time: int) 訓練模型  
範例:  
```python
LLM_Model.train(epoch = 1000, batch_size = 32, path = './test.pt', trigger = 0.03, time = 5)
```
--------------------------------------------------------
LLM_Model.save(self, path:str) 儲存模型位置儲存檔案為.pt檔  
範例:  
```python
LLM_Model.(path = './my_model.pt')
```
--------------------------------------------------------
以下為使用範例提供參考:  
```python
plm = "EleutherAI/pythia-70m-deduped"
revision = "step3000"
load_path = './Dataset/train.tsv'
save_path = './test.pt'
model = LLM_Model(plm = plm, revision = revision)
model.load_data(path = load_path)
model.train(epoch = 1000, batch_size = 32, path = save_path, trigger = 0.03, time = 5)
model.save(path = save_path)
```




## load.py檔案

load.py檔案為讀取模型檔案並且進行預測。  
  
訓練過程中所要求預測資料形式為:  
檔案名稱\t起始位置\t內容  
  
以下為主要設定參數:  
plm 代表「預訓練語言模型」 (String)  
revision 為指定模型的特定版本  (String)  
path 為讀取模型檔案、預測資料、儲存結果路徑，依照情況所分(String)  
max_new_tokens 預測結果字數上限設定 預設200(int)  

主要程式碼:  
Load_Model(self, plm:str , revision :str, path:str) 物件，並讀取預訓練模型且根據路徑讀取自己的模型權重  
範例:  
```python
model = Load_Model(plm = 'EleutherAI/pythia-1B-deduped', revision = "step3000", path = "./ test.pt")
```
--------------------------------------------------------
Load_Model.load_data(self, path:str) 讀取需預測.tsv檔案    
範例:  
```python
Load_Model.load_data(path = "./data.tsv")
```
--------------------------------------------------------
Load_Model.predict_one(self, input:str, max_new_tokens = 200) 使用模型預測一句話並且將結果回傳。  
範例:  
將結果預測字符限制更改在一百字內  
```python
result = Load_Model.predict_one(input = '11.38am on 28/2/13', max_new_tokens = 100)
```
將結果預測字符限制不做更改，其預設為200字為上限  
```python
result = Load_Model.predict_one(input = '11.38am on 28/2/13')
```
--------------------------------------------------------
Load_Model.predict_all(self)將依照路徑所讀取的檔案一次性預測全部  
範例:  
```python
Load_Model.predict_all()
```
--------------------------------------------------------
Load_Model.save_original_result(path:str)將Load_Model.predict_all()預測結果儲存至指定路徑  
範例:  
```python
Load_Model.save_original_result(path = './result.tsv')
```
--------------------------------------------------------
以下為使用範例提供參考:  

如需預測全部  
```python
plm = "EleutherAI/pythia-1b-deduped"
revision = "./step3000"
model_path = './my_model.pt'
data_path = './data.tsv'
save_path = './result.tsv'
model = Load_Model(plm  =  plm, revision  =  revision, path  =  model_path)
model.load_data(path  =  data_path)
# a = model.predict_one(input = '11.38am on 28/2/13', max_new_tokens = 100)
# print(a)
model.predict_all()
model.save_original_result(path  =  save_path)
```

如需簡易預測  
```python
plm = "EleutherAI/pythia-1b-deduped"
revision = "step3000"
model_path = './my_model.pt'
data_path = './data.tsv'
save_path = './result.tsv'
model = Load_Model(plm = plm, revision = revision, path  =  model_path)
result = model.predict_one(input = '11.38am on 28/2/13', max_new_tokens = 100)
print(a)
```



## extract.py檔案

load.py檔案為讀取模型檔案並且進行預測。  

訓練過程中所要求預測資料形式為:  
檔案名稱\t起始位置\t內容  
  
以下為主要設定參數:  
path 為讀取預測結果、儲存結果路徑，依照情況所分(String)  
主要程式碼:  
extract_information(self, path:str)物件，讀取預測結果檔案並且初始化放入self.predict_list  
範例:  
```python
info = extract_information(path = data_path)
```
--------------------------------------------------------
extract_information.extract(self, fid, idx, content, label) 將其預測結果提取出正確且有用的資訊並，fid檔案名、idx字符起始位置、content內文、label標籤  
範例:  
```python
extract_information.extract('1.txt', '100', 'dog', 'PHI: NULL')
```
--------------------------------------------------------
extract_informationmanual_extract_location_other()根據正規化模組手動提取LOCATION-OTHER內容  
範例:  
```python
extract_informatio.manual_extract_location_other()
```
--------------------------------------------------------
save_result(self, path:str) 依據路徑保存處裡過後結果  
範例:  
```python
extract_informatio.save_result(path = './result.txt')
```
--------------------------------------------------------
以下為使用範例提供參考:  
  
```python
data_path = './data.tsv'
save_path = './answer.txt'
info = extract_information(path = data_path)
for idx, predict in enumerate(info.predict_list):
elements = set()
labels = predict['label'].split('\\n')
for label in labels:
if len(label) > 2 and label not in elements:
info.extract(predict['fid'], predict['idx'], predict['content'], label)
elements.add(label)
info.manual_extract_location_other()
info.save_result(path  =  save_path)
```
## 建議事項
其train、load、extract內部已寫好程式碼，只需修改下方參數並且依照train > load > extract 順序執行即可。  
