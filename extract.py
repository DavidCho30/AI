from datasets import load_dataset, Features, Value
import re

class extract_information():
    
    def __init__(self, path:str):
        self.predict_list = self.load_dataset(path)
        self.tag = [
            'DOCTOR', 'PATIENT', 'PHONE', 'AGE', 'IDNUM', 'MEDICALRECORD', 'COUNTRY', 'CITY', 'STATE', 'STREET', 'ZIP',
            'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'LOCATION-OTHER', 'DATE', 'TIME', 'DURATION', 'PHI'
        ]
        self.result = []

    def load_dataset(self, path: str):
        dataset = load_dataset("csv", data_files=path,
                               delimiter="\t",
                               features=Features({
                                   'fid': Value('string'),
                                   'idx': Value('string'),
                                   'content': Value('string'),
                                   'label': Value('string'),
                                   'other': Value('string'),
                               }),
                               column_names=['fid', 'idx', 'content', 'label', 'other'],
                               keep_default_na=False
                               )
        return list(dataset['train'])
    
    def DOCTOR(self, fid, idx, content, value):
        if((content.find(value) != -1) and (len(value) > 1)):
            start_idx = int(idx)  + content.find(value)
            self.result.append(f"{fid}\tDOCTOR\t{start_idx}\t{start_idx + len(value)}\t{value}")

    def PATIENT(self, fid, idx, content, value):
        if(content.find(value) != -1):
            start_idx = int(idx) + content.find(value)
            self.result.append(f"{fid}\tPATIENT\t{start_idx}\t{start_idx + len(value)}\t{value}")

    def AGE(self, fid, idx, content, value):
        if 0 < len(value) <= 3:
            if(content.find(value) != -1):
                keywords =  ['year', 'yr ', 'yo', 'weeks', 'F', 'M', 'Age', 'age', 'female', 'male']
                for keyword in keywords:
                    if keyword in content[:content.find(value) +10]:      
                        start_idx = int(idx) + content.find(value)
                        self.result.append(f"{fid}\tAGE\t{start_idx}\t{start_idx + len(value)}\t{value}")
                        break
                                        
    def PHONE(self, fid, idx, content, value):
        if(content.find(value) != -1 and len(value) > 2):
            start_idx = int(idx) + content.find(value)
            self.result.append(f"{fid}\tPHONE\t{start_idx}\t{start_idx + len(value)}\t{value}")
            
    def IDNUM(self, fid, idx, content, value):
        if(content.find(value) != -1 and len(value) > 2):
            k = content.find(value)
            start_idx = int(idx)  + content.find(value)
            while(k > -1):
                self.result.append(f"{fid}\tIDNUM\t{start_idx}\t{start_idx + len(value)}\t{value}")
                k = content.find(value, k + 1)
                start_idx = int(idx) + k

    def MEDICALRECORD(self, fid, idx, content, value):
        if(content.find(value) != -1 and len(value) > 2):
            start_idx = int(idx) + content.find(value)
            self.result.append(f"{fid}\tMEDICALRECORD\t{start_idx}\t{start_idx + len(value)}\t{value}")
                
    def COUNTRY(self, fid, idx, content, value):
        if(content.find(value) != -1):
            start_idx = int(idx) + content.find(value)
            self.result.append(f"{fid}\tCOUNTRY\t{start_idx}\t{start_idx + len(value)}\t{value}")
        return
            
    def CITY(self, fid, idx, content, value):
        if(content.find(value) != -1 and len(value) > 2):
            start_idx = int(idx)  + content.find(value)
            self.result.append(f"{fid}\tCITY\t{start_idx}\t{start_idx + len(value)}\t{value}")

    def STATE(self, fid, idx, content, value):
        if(content.find(value) != -1 and len(value) > 2):
            start_idx = int(idx)  + content.find(value)
            self.result.append(f"{fid}\tSTATE\t{start_idx}\t{start_idx + len(value)}\t{value}")
            
    def STREET(self, fid, idx, content, value):
        if(content.find(value) != -1 and len(value) > 2):
            start_idx = int(idx)  + content.find(value)
            self.result.append(f"{fid}\tSTREET\t{start_idx}\t{start_idx + len(value)}\t{value}")
            
    def ZIP(self, fid, idx, content, value):
        if(content.find(value) != -1 and len(value) > 2):
            start_idx = int(idx)  + content.find(value)
            self.result.append(f"{fid}\tZIP\t{start_idx}\t{start_idx + len(value)}\t{value}")
            
    def DEPARTMENT(self, fid, idx, content, value):
        if(content.find(value) != -1 and len(value) > 2):
            start_idx = int(idx)  + content.find(value)
            self.result.append(f"{fid}\tDEPARTMENT\t{start_idx}\t{start_idx + len(value)}\t{value}")
        
    def HOSPITAL(self, fid, idx, content, value):
        if "HOSPITAL" in value or 'SERVICE' in value or 'HEALTH' in value:
            if(content.find(value) != -1):
                hospitals = value.split("AND")
                element = set()
                for hospital in hospitals:
                    if hospital not in element:
                        start_idx = int(idx) + content.find(hospital)
                        self.result.append(f"{fid}\tHOSPITAL\t{start_idx}\t{start_idx + len(hospital)}\t{hospital}")
                        element.add(hospital)
                
    def ORGANIZATION(self, fid, idx, content, value):
        if(content.find(value) != -1 and len(value) > 2):
            start_idx = int(idx)  + content.find(value)
            self.result.append(f"{fid}\tORGANIZATION\t{start_idx}\t{start_idx + len(value)}\t{value}")
            
    def LOCATION_OTHER(self, fid, idx, content, value):
        # if(content.find(value) != -1 and len(value) > 2):
        #     start_idx = int(idx)  + content.find(value)
        #     self.result.append(f"{fid}\tLOCATION-OTHER\t{start_idx}\t{start_idx + len(value)}\t{value}")
        return
            
    def DATE(self, fid, idx, content, value):
        if "=>" in value:
            time_1, time_2 = value.split('=>', 1)
            time_3 = self.normalize_date(time_1)
            if ('at' in time_1 or  ":" in time_1):
                self.TIME(fid, idx, content, value)
            elif time_3 != None:
                if(content.find(time_1) != -1 and 'now' not in time_1 and 'today' not in time_1):
                    start_idx = int(idx) + content.find(time_1)
                    self.result.append(f"{fid}\tDATE\t{start_idx}\t{start_idx + len(time_1)}\t{time_1}\t{time_3}")
            elif(len(time_1) == 4):
                start_idx = int(idx) + content.find(time_1)
                self.result.append(f"{fid}\tDATE\t{start_idx}\t{start_idx + len(time_1)}\t{time_1}\t{time_1}")
            else:
                if(content.find(time_1) != -1 and 'now' not in time_1 and 'today' not in time_1):
                    start_idx = int(idx) + content.find(time_1)
                    self.result.append(f"{fid}\tDATE\t{start_idx}\t{start_idx + len(time_1)}\t{time_1}\t{time_2}")
                
    def TIME(self, fid, idx, content, value):

        if "=>" in value:
            time_1, time_2 = value.split('=>', 1)
            if(content.find(time_1) != -1 ):
                if(":" in time_1):
                    time_3 = self.manual_extract_time_semicolon(time_1)
                    if time_3 != None:
                        start_idx = int(idx) + content.find(time_1)
                        self.result.append(f"{fid}\tTIME\t{start_idx}\t{start_idx + len(time_1)}\t{time_1}\t{time_3}")
                elif ":" not in time_1:
                    time_3 = self.manual_extract_time_dot(time_1)
                    if time_3 != None:
                        start_idx = int(idx) + content.find(time_1)
                        self.result.append(f"{fid}\tTIME\t{start_idx}\t{start_idx + len(time_1)}\t{time_1}\t{time_3}")
                    else:
                        start_idx = int(idx) + content.find(time_1)
                        self.result.append(f"{fid}\tTIME\t{start_idx}\t{start_idx + len(time_1)}\t{time_1}\t{time_2}")
            
                
    def DURATION(self, fid, idx, content, value):
        if "=>" in value:
            time_1, time_2 = value.split('=>', 1)
            if(content.find(time_1) != -1 and '-' not in time_2):
                start_idx = int(idx) + content.find(time_1)
                self.result.append(f"{fid}\tDURATION\t{start_idx}\t{start_idx + len(time_1)}\t{time_1}\t{time_2}")
                
                
    def extract(self, fid, idx, content, label):
        parts = label.split(": ", 1)
        if len(parts) != 2:
            return
        if 'DOCTOR' in parts[0]:
            self.DOCTOR(fid, idx, content, parts[1])
        elif 'PATIENT' in parts[0]:
            self.PATIENT(fid, idx, content, parts[1])
        elif 'AGE' in parts[0]:
            self.AGE(fid, idx, content, parts[1])
        elif 'PHONE' in parts[0]:
            self.PHONE(fid, idx, content, parts[1])
        elif 'IDNUM' in parts[0]:
            self.IDNUM(fid, idx, content, parts[1])
        elif 'MEDICALRECORD' in parts[0]:
            self.MEDICALRECORD(fid, idx, content, parts[1])
        elif 'COUNTRY' in parts[0]:
            self.COUNTRY(fid, idx, content, parts[1])
        elif 'CITY' in parts[0]:
            self.CITY(fid, idx, content, parts[1])
        elif 'STATE' in parts[0]:
            self.STATE(fid, idx, content, parts[1])
        elif 'STREET' in parts[0]:
            self.STREET(fid, idx, content, parts[1])
        elif 'ZIP' in parts[0]:
            self.ZIP(fid, idx, content, parts[1])
        elif 'DEPARTMENT' in parts[0]:
            self.DEPARTMENT(fid, idx, content, parts[1])
        elif 'HOSPITAL' in parts[0]:
            self.HOSPITAL(fid, idx, content, parts[1])
        elif 'ORGANIZATION' in parts[0]:
            self.ORGANIZATION(fid, idx, content, parts[1])
        elif 'LOCATION-OTHER' in parts[0]:
            self.LOCATION_OTHER(fid, idx, content, parts[1])
        elif 'DATE' in parts[0]:
            self.DATE(fid, idx, content, parts[1])
        elif 'TIME' in parts[0]:
            self.TIME(fid, idx, content, parts[1])
        elif 'DURATION' in parts[0]:
            self.DURATION(fid, idx, content, parts[1])
            
    def manual_extract_location_other(self):
        for idx, List in enumerate(self.predict_list):
            pattern = re.compile(r'P\.?\s?O\.?\s?\s?BOX\s?\d+', re.IGNORECASE)
            matches = pattern.findall(List['content'])
            if len(matches) != 0:
                start_idx = int(List['idx']) 
                self.result.append(f"{List['fid']}\tLOCATION-OTHER\t{start_idx}\t{start_idx + len(matches[0])}\t{matches[0]}")
            
            pattern = re.compile(r'BOX\s+\d+')
            matches = pattern.findall(List['content'])
            if len(matches) != 0 and List['content'].find(matches[0][0]) == 0:
                start_idx = self.predict_list[idx-1]['idx']
                end_idx = int(List['idx'])+len(List['content'])
                self.result.append(f"{List['fid']}\tLOCATION-OTHER\t{start_idx}\t{end_idx}\t{self.predict_list[idx-1]['content']} {List['content']}")
                
    
    def convert_to_24hr(self, time_str):
            if "pm" in time_str.lower() and "12:" not in time_str.lower() and "12." not in time_str.lower():
                # 如果是pm且不是12點，則加12小時
                hour, minute = map(int, re.findall(r'\d+', time_str))
                hour += 12
                return f"{hour:02d}:{minute:02d}"
            else:
                # 其他情況保持不變
                hour, minute = map(int, re.findall(r'\d+', time_str))
                return f"{hour:02d}:{minute:02d}"
            
    def normalize_date(slef, date_str):
            date_pattern = re.compile(r'(\d{1,2})[/\.](\d{1,2})[/\.](\d{2,4})')
            match = date_pattern.match(date_str)
            if match:
                day, month, year = map(int, match.groups())
            # 將日期重新格式化為YYYY-MM-DD
                if year < 100:
                    year += 2000
                formatted_date = f"{year:02d}-{month:02d}-{day:02d}"
                return formatted_date
                
    def manual_extract_time_semicolon(self, text:str):
        date_pattern = re.compile(r'\b(\d{1,2}[/\.]\d{1,2}[/\.]\d{2,4})\b')
        time_pattern = re.compile(r'\b(\d{1,2}:\d{2}(?:\s?[ap]m)?)\b')
        date_matches = date_pattern.findall(text)
        time_matches = time_pattern.findall(text)
        if(len(date_matches) != 0 and len(time_matches) != 0):
            converted_date = self.normalize_date(date_matches[0])
            converted_time = self.convert_to_24hr(time_matches[0])
            result_text = converted_date + 'T' + converted_time
            return result_text
        else:
            return None

    def manual_extract_time_dot(self, text:str):
        converted_time = None
        converted_date = None
        parts = text.split()
        for part in parts:
            if(len(part.split('.')) == 2):
                converted_time = self.convert_to_24hr(part)
                
            if(len(part.split('.')) == 3):
                converted_date = self.normalize_date(part)
            elif(len(part.split(r'/')) == 3):
                converted_date = self.normalize_date(part)
                
            if(converted_date != None and converted_time != None):
                result_text = converted_date + 'T' + converted_time
                return result_text
            else:
                return None

            
    def save_result(self, path:str):
        print('---SAVING---')
        with open(path, "w", encoding = "utf-8") as file:
            for line in self.result:
                file.write(line + "\n")
        print('---DONE---')
        
                            
                        
    
            
    
    
if __name__ == '__main__':
    data_path = './reslut.tsv'
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
    info.save_result(path = save_path)

    
    
    

    