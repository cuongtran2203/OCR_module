import os
import pandas as pd
from tqdm import tqdm
import shutil
SOURCE = "../datasets_ocr_vn"
FILE_NAME = []
LABEL = []
os.makedirs("data_ocr",exist_ok=True)
for root,dir,files in os.walk(SOURCE):
    if len(dir) == 0:
        for file in tqdm(files):
            if file.endswith(".jpg") or file.endswith(".png"):
                FILE_NAME.append(file)
                src_file = os.path.join(root,file)
                save_file = os.path.join("data_ocr",file)
                with open(os.path.join(root,file.split(".")[0]+".txt"),"r",encoding="utf8") as f:
                    data = f.read()
                LABEL.append(data)
                shutil.copyfile(src_file,save_file)
                
df = pd.DataFrame({"file_name":FILE_NAME,"text":LABEL})
df.to_csv("datasets_VN_OCR.csv",index=False)