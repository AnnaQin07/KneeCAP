
import os
import math
import json 
import pandas as pd 

from collections import defaultdict

def Gen_hkaa_labels(patients2imgs_path, imgs2hkaa_path, save_path, img_dir, HKA_head_name="V01HKANGLE"):
    
    patients2img_dict = defaultdict(list)
    patients2imgs = pd.read_excel(patients2imgs_path, dtype={"Corresponding patient ID": str})
    for _, row in patients2imgs.iterrows():
        filename = os.path.splitext(row["Filename"])[0][:-4]
        filename = f"{filename}.png"
        patient_id = row["Corresponding patient ID"]
        patients2img_dict[patient_id].append(filename)
    
    imgs = set(map(lambda x: os.path.splitext(x)[0], os.listdir(img_dir)))
    for k, v in patients2img_dict.items():
        if len(v) > 1:
            for vi in v:
                if os.path.splitext(vi)[0] not in imgs:
                    patients2img_dict[k].remove(vi)
    
    result = {}
    patients2imgs = pd.read_excel(imgs2hkaa_path, dtype={"ID": str})
    for _, row in patients2imgs.iterrows():
        patient_id = row["ID"]
        side = row["SIDE"]
        side = 'l' if side == 2 else "r"

        if patients2img_dict[patient_id]:
            for name in patients2img_dict[patient_id]:
                name = f"{os.path.splitext(name)[0]}_{side}.png"

                if row[HKA_head_name] != " " and not math.isnan(row[HKA_head_name]):
                    result[name] = float(row[HKA_head_name])
    
    with open(save_path, 'w') as jf:
        json.dump(result, jf)
        
def Gen_hkaa_labels_multi_sheets(xlsx_path, patients2imgs_sheet, imgs2hkaa_sheet, save_path, img_dir, HKA_head_name="V01HKANGLE"):
    patients2img_dict = defaultdict(list)
    patients2imgs = pd.read_excel(xlsx_path, sheet_name=patients2imgs_sheet, dtype={"patient ID": str})
    for _, row in patients2imgs.iterrows():
        filename = os.path.splitext(row["Image name"])[0][:-4]
        filename = f"{filename}.png"
        patient_id = row["Patient ID"]
        patients2img_dict[patient_id].append(filename)
    
    imgs = set(map(lambda x: os.path.splitext(x)[0], os.listdir(img_dir)))
    for k, v in patients2img_dict.items():
        if len(v) > 1:
            for vi in v:
                if os.path.splitext(vi)[0] not in imgs:
                    patients2img_dict[k].remove(vi)

    result = {}
    patients2imgs = pd.read_excel(xlsx_path, sheet_name=imgs2hkaa_sheet, dtype={"ID": str})

    for _, row in patients2imgs.iterrows():
        patient_id = row["ID"]
        side = row["SIDE"]
        side = 'l' if side == 2 else "r"

        if patients2img_dict[patient_id]:
            for name in patients2img_dict[patient_id]:
                name = f"{os.path.splitext(name)[0]}_{side}.png"

                if row[HKA_head_name] != ' ':
                    if not math.isnan(row[HKA_head_name]):
                        result[name] = float(row[HKA_head_name])
    
    with open(save_path, 'w') as jf:
        json.dump(result, jf)
    
def gen_label_24_48(xlsx_path, sheet_name, HKA_head_name="V03HKANGLE"):
    result = {}
    patients2imgs = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype={"Image_ID_lookup": str})
    for _, row in patients2imgs.iterrows():
        side = row["SIDE"]
        side = 'l' if side == 2 else "r"
        name = row["Image_ID_lookup"]
        name = f"{name}_{side}.png"
        if row[HKA_head_name] != " " and not math.isnan(row[HKA_head_name]):
            result[name] = row[HKA_head_name]
    
    with open(save_path, 'w') as jf:
        json.dump(result, jf)
    
        
     
     
     
        
        
    

if __name__ == "__main__":
    
    # patients2imgs_path = "D:/Datasets/comp8603/Lower_limb_Xrays/ground_truth_hka/patient_id.xlsx"
    # imgs2hkaa_path = "D:/Datasets/comp8603/Lower_limb_Xrays/ground_truth_hka/36m_hka.xlsx"
    # save_path = "36m_hkaa.json"
    # img_dir = "D:/Datasets/comp8603/Lower_limb_Xrays/36m_dicoms/processed"
    # Gen_hkaa_labels(patients2imgs_path, imgs2hkaa_path, save_path, img_dir, HKA_head_name="V05HKANGLE")
    
    xlsx_path = "D:/Datasets/comp8603/Lower_limb_Xrays/ground_truth_hka/Longleg_indices_24_48_months.xlsx"
    sheet_name = "48m_HKAs"
    save_path = "48m_hkaa.json"
    img_dir = "D:/Datasets/comp8603/Lower_limb_Xrays/24m_dicoms/processed"
    gen_label_24_48(xlsx_path, sheet_name, HKA_head_name="V06HKANGLE") 
        