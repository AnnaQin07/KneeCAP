
import os 
import json
import math
import pydicom

import pandas as pd 
import numpy as np 
import pingouin as pg

from collections import defaultdict
from matplotlib import pyplot as plt 


def evaluation(args):
    
    eval_args = args.evaluation
    gt_paths = eval_args.gt_paths
    img_type_path = eval_args.img_type_path
    csv_file = os.path.join('experiments', args.exp_name, 'measurements.csv')
    for k, v in gt_paths.items():
        case2error, pred_a, gt_a, pred_d, gt_d, pred, gt, num_fail = compute_metrics(csv_file, v, img_type_path, ltype=k)
        analysis_plot(pred, gt, os.path.join('experiments', args.exp_name, 'plots'), title=k, plot_type='Bal')
        analysis_plot(pred, gt, os.path.join('experiments', args.exp_name, 'plots'), title=k, plot_type='regress')
        all_errors = {k: v for k, v in case2error.items() if v is not None}
        high_errors = sorted(all_errors.items(), key=lambda y: y[1])[-20:]
        print(high_errors)
    
    save_path = os.path.join('experiments', args.exp_name, 'analysis.xlsx')
    generate_statistical_file(csv_file, gt_paths, save_path)


def analysis_plot(pred, gt, save_dir, title='HKA', plot_type='Bal', dpi=100):
    """SHow analysis plot 

    Args:
        pred (np.array): pred results
        gt (np.array): ground truth results
        title (str, optional): showed title. Defaults to 'HKA'.
        plot_type (str, optional): plot type choose from {Bal, regress}. Defaults to 'Bal'.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    means = np.mean([pred, gt], axis=0)
    diffs =  pred - gt
    md = np.mean(diffs)
    sd = np.std(diffs, axis=0)
    upper_bound = md + 1.96 * sd
    lower_bound = md - 1.96 * sd
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(111)
    if plot_type == 'Bal':


        ax1.scatter(means, diffs)
        ax1.axhline(0, color='black')
        ax1.axvline(0, color='black')
        ax1.axhline(md, color='red', linestyle='-')
        ax1.axhline(md + 1.96 * sd, color='gray', linestyle='--')
        ax1.axhline(md - 1.96 * sd, color='gray', linestyle='--')
        ax1.text(2.86, 0.2, 'mean diff {:.1f}°'.format(md), fontsize=40)
        ax1.text(-1.86, 3.30, 'upper 95% LOA = {:.1f}°'.format(upper_bound), fontsize=40)
        ax1.text(-1.86, -3.30, 'lower  95% LOA =  {:.1f}°'.format(lower_bound), fontsize=40)
        ax1.set_ylim(-10, 10) # +-4
        ax1.set_xlabel(f"[[{title}(GT) + {title}(Pred)]/2] (°)", fontsize=40)
        ax1.set_ylabel(f"[{title}(GT) - {title}(Pred)] (°)", fontsize=40)
        ax1.tick_params(labelsize=35)
    elif plot_type == 'regress':
        k, b = np.polyfit(pred, gt, 1)
        xs = np.linspace(pred.min(), pred.max(), 1000)
        ys = k * xs + b
        gt_mean = np.mean(gt)
        tot = np.mean(np.square(gt - gt_mean))
        mse = np.mean(np.square(pred - gt))
        R_2 = 1 - mse / tot
        
        ax1.scatter(pred, gt)
        ax1.plot(xs, ys, c='r')
        ax1.axhline(0, color='black')
        ax1.axvline(0, color='black')
        ax1.set_xlabel(f"Pred {title}(°)", fontsize=40)
        ax1.set_ylabel(f"GT {title}(°)", fontsize=40, labelpad=-40)
        ax1.text(-9.86, 12.30, f'gt = {round(k, 3)}*pred + {round(b, 3)}'.format(upper_bound), fontsize=40)
        ax1.text(-9.86, 9.30, f'R^2 = {round(R_2, 3)}'.format(upper_bound), fontsize=40)
        ax1.tick_params(labelsize=30)
    
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{title}_{plot_type}.jpg"), bbox_inches='tight', pad_inches = -0.1, dpi=dpi*7)
    plt.clf() 
    
    
def compute_metrics(csv_file, gt_hka_path, img_type_path, ltype='hka'):
    """compute the metrics val for algorithm performance evaluation

    Args:
        csv_file (str): measurement file path produced by postprocessing algorithm path/to/*.csv
        gt_hka_path (str): ground truth file path produced by postprocessing algorithm path/to/*.json
        img_type_path (str): X-ray type file path produced by postprocessing algorithm path/to/*.json
        ltype (str, optional): angle for evaluation. Defaults to 'hka'.

    Returns:
        _type_: _description_
    """
    pred_frame = pd.read_csv(csv_file)
    num_fail_a, num_fail_d = 0, 0
    gt_hkas_a, hkas_a = [], []
    gt_hkas_d, hkas_d = [], []
    count = 0
    case2error = {}
    worst_error = 0
    with open(gt_hka_path, 'r') as jf:
            hka_labels = json.load(jf)
    
    with open(img_type_path, 'r') as jf:
            img_type = json.load(jf)
    num_a = len([name for name in img_type.keys() if img_type[name] == 'a'])
    num_d = len(img_type) - num_a
    for i in range(pred_frame.shape[0]):
        series = pred_frame.iloc[i]
        name = series['img_name']
        side = series['side']
        hka = series[ltype]
        name = os.path.splitext(name)[0]
        if 'pre' in name or 'post' in name:
            suffix = 'jpg'
        else:
            suffix = 'png'
        if side == 2:
            query = f"{name}_l.{suffix}"
            side = 'l'
        else:
            query = f"{name}_r.{suffix}"
            side = 'r'
        if math.isnan(hka) or hka is None:
            if img_type.get(query) == 'a':
                num_fail_a += 1
            elif img_type.get(query) == 'd':
                num_fail_d += 1
            # fail_case.append(f"{name}_{side}.png")
            case2error[f"{os.path.split(csv_file)[0]}/{name}_{side}.{suffix}"] = None
        else:
            count += 1
            gt_hka = hka_labels.get(query, None)
            if gt_hka == ' ' or gt_hka is None:
                continue
            if img_type[query] == 'a':
                gt_hkas_a.append(gt_hka)
                hkas_a.append(hka)
            else:
                gt_hkas_d.append(gt_hka)
                hkas_d.append(hka)
            error = abs(gt_hka - hka)
            case2error[f"{os.path.split(csv_file)[0]}/{name}_{side}.{suffix}"] = error
            if error > worst_error:
                worst_error = error
                worst_img = f"{name}_{side}.{suffix}"
            
    gt_hkas_a = np.array(gt_hkas_a)
    hkas_a = np.array(hkas_a)
    gt_hkas_d = np.array(gt_hkas_d)
    hkas_d = np.array(hkas_d)
    
    print(f"worst prediction is {worst_img} with error : {worst_error}")
    print(f"There are {num_fail_a+num_fail_d} failed cases in {pred_frame.shape[0]} examples which cannot get hka, {num_fail_a} analog images and {num_fail_d} digital images")
    num_fail = num_fail_a+num_fail_d
    print(f"error rate {round(num_fail * 100 / pred_frame.shape[0], 3)}")
    if num_a > 0:
        print(f"error rate analog {round(num_fail_a * 100 / num_a, 3)}\%")
    if num_d > 0:
        print(f"error rate digital {round(num_fail_d * 100 / num_d, 3)}\%")
    
    
    if gt_hkas_a.size > 0:
        print("== analog==")
        get_info(gt_hkas_a, hkas_a)
    if gt_hkas_d.size > 0:
        print("== digital==")
        get_info(gt_hkas_d, hkas_d)
    print("== all==")
    hkas = np.concatenate([hkas_a, hkas_d], axis=0)
    gt_hkas = np.concatenate([gt_hkas_a, gt_hkas_d], axis=0)
    get_info(gt_hkas, hkas)
    
    return case2error, hkas_a, gt_hkas_a, hkas_d, gt_hkas_d, hkas, gt_hkas, num_fail


def get_info(gt_hkas, hkas):
    """show mean absolute errors and icc scores

    Args:
        gt_hkas (np.array): ground turth results
        hkas (_type_): prediction results
    """
    print(f"MAE: {round(np.mean(np.abs(gt_hkas - hkas)), 3)} ± {round(np.std(np.abs(gt_hkas - hkas)), 3)}")
    
    # report icc 
    hka_combines = np.stack([hkas, gt_hkas], axis=1)
    num_sample = len(hka_combines)
    items = ['pred', 'gt']
    outframe = defaultdict(list)
    for i in range(2):
        name = [items[i]] * num_sample
        imgs = np.arange(num_sample).tolist()
        hka_vals = hka_combines[:, i].tolist()
        
        outframe['imgs'].extend(imgs)
        outframe['evaluator'].extend(name)
        outframe['hka'].extend(hka_vals)
    outframe = pd.DataFrame(outframe)
    icc = pg.intraclass_corr(data = outframe, targets = 'imgs', raters = 'evaluator',ratings = 'hka')   
    print(icc)    



def generate_statistical_file(preds, gt_files, save_path):
    """generate a .xlsx file for further evaluation 

    Args:
        preds (str): path of prediction file path/to/*.csv
        gt_files (str): path of ground truth file path/to/*.json
        save_path (str): path you wanna save path/to/*.xlsx
    """
    akys = list(gt_files.keys())
    gt_collects = []
    for path in gt_files.values():
        with open(path, 'r') as jf:
            gt = json.load(jf)
        gt_collects.append(gt)   
    
    preds = pd.read_csv(preds)
    saved_sheets = [defaultdict(list) for _ in range(len(akys))]
    
    for i, r in preds.iterrows():
        name = r['img_name']
        side = 'r' if r['side'] == 1 else 'l'
        if 'pre' in name or 'post' in name:
            suffix = 'jpg'
        else:
            suffix = 'png'
        query = f"{os.path.splitext(name)[0]}_{side}.{suffix}"
        gts = [gt.get(query, None) for gt in gt_collects]
        for i in range(4):
            if gts[i] is not None and r[akys[i]] is not None:
                saved_sheets[i]['name'].append(query)
                saved_sheets[i]['pred'].append(r[akys[i]])
                saved_sheets[i]['gt'].append(gts[i])
                saved_sheets[i]['error'].append(r[akys[i]] - gts[i])
                saved_sheets[i]['abs_error'].append(abs(r[akys[i]] - gts[i]))
    writer = pd.ExcelWriter(save_path, engine='openpyxl')
    for i in range(4):
        df = pd.DataFrame(saved_sheets[i])
        df = df.dropna(axis=0, how='any') 
        df.to_excel(writer, sheet_name=akys[i], index=False)
    # writer.save()
    writer.close()
    
    
def find_good_results(output_path, gt_path, vis_path, erange):
    
    good_results = set()
    with open(gt_path, 'r') as fcc_file:
        gt_labels = json.load(fcc_file)
    df = pd.read_csv(output_path)
    
    for _, row in df.iterrows():
        filename = row['img_name']
        side = row['side']
        side = 'r' if side == 1 else 'l'
        name = f"{os.path.splitext(filename)[0]}_{side}.png"
        hka = row["hka"]

        if hka != " " and not math.isnan(hka) and gt_labels.get(name) is not None and gt_labels.get(name) != " ":
            error = float(hka) - float(gt_labels.get(name))
            if erange[0] <= error <= erange[1]:
                good_results.add(filename)
    
    vis_imgs = os.listdir(vis_path)
    for result in good_results:
        if result in vis_imgs:
            os.remove(os.path.join(vis_path, result))
            

    
            
                
    
    
if __name__ == "__main__":
    

    
    csv_file = r"experiments/gt_canberra/measurements.csv"
    gt_paths = {
                'hka': r"F:/Datasets/comp8603/Lower_limb_Xrays/Canberra_hospital/HKA_labels.json", 
                'LDFA': r"F:/Datasets/comp8603/Lower_limb_Xrays/Canberra_hospital/LDFA_labels.json",
                'MPTA': r"F:/Datasets/comp8603/Lower_limb_Xrays/Canberra_hospital/MPTA_labels.json", 
                'MPTA-LDFA':r"F:/Datasets/comp8603/Lower_limb_Xrays/Canberra_hospital/aHKA_labels.json"}
    
    img_type_path = r"F:/Datasets/comp8603/Lower_limb_Xrays/Canberra_hospital/type_label.json"
    
    for k, v in gt_paths.items():
        print("==========Item {k}==========\n")
        case2error, pred_a, gt_a, pred_d, gt_d, pred, gt, num_fail = compute_metrics(csv_file, v, img_type_path, ltype=k)
        analysis_plot(pred, gt, r"experiments/gt_canberra/plots", title=k, plot_type='Bal')
        analysis_plot(pred, gt, r"experiments/gt_canberra/plots", title=k, plot_type='regress')
        all_errors = {k: v for k, v in case2error.items() if v is not None}
        high_errors = sorted(all_errors.items(), key=lambda y: y[1])[-20:]
        print(high_errors)
    
    save_path = r"experiments/gt_canberra/analysis.xlsx"
    generate_statistical_file(csv_file, gt_paths, save_path)
    
    # output_path = r"experiments/inf_48m_finetune/measurements.csv"
    # gt_path = r"F:/Datasets/comp8603/Lower_limb_Xrays/48m_dicoms/hka_labels.json"
    # vis_path = r"F:/Datasets/comp8603/results/48m_finetune/kpts_new"
    # find_good_results(output_path, gt_path, vis_path, [-1.8, 1.7])
    
    # # 12m
    # print("12m--------------- \n")
    # gt_labels = r"E:/learn_ANU/COMP8603/codes/MA_detections/data/12m_hkaa.json"
    # gt_hka_path = r"F:/Datasets/comp8603/Lower_limb_Xrays/12m_dicoms/pretrain/hka_labels.json"
    # # pred_hkaas = "E:/learn_ANU/COMP8603/codes/human_knee_joints_extractions/exps/version1_inf12/key_pointsANDhka_new.csv"
    # pred_hkaas = r"experiments/inf_12m_finetune/measurements.csv"
    # img_type_path = r"F:/Datasets/comp8603/Lower_limb_Xrays/12m_dicoms/type_label.json"
    # err12, hkas_a_12, gt_hkas_a_12, hkas_d_12, gt_hkas_d_12, hkas_12, gt_hkas_12, num_fail_12 = check_hka_mse(pred_hkaas, gt_hka_path, img_type_path)


    # # 24m 
    # print("24m---------------------- \n")
    # gt_labels = r"E:/learn_ANU/COMP8603/codes/MA_detections/data/24m_hkaa.json"
    # gt_hka_path = r"F:/Datasets/comp8603/Lower_limb_Xrays/24m_longlegs/hka_labels.json"
    # # pred_hkaas = "E:/learn_ANU/COMP8603/codes/human_knee_joints_extractions/exps/version1_inf24/key_pointsANDhka_new.csv"
    # pred_hkaas = r"experiments/inf_24m_finetune/measurements.csv"
    # img_type_path = r"F:/Datasets/comp8603/Lower_limb_Xrays/24m_longlegs/type_label.json"
    # err24, hkas_a_24, gt_hkas_a_24, hkas_d_24, gt_hkas_d_24, hkas_24, gt_hkas_24, num_fail_24 = check_hka_mse(pred_hkaas, gt_hka_path, img_type_path)

    
    # # 36
    # print("36m----------------------- \n")
    # gt_labels = r"E:/learn_ANU/COMP8603/codes/MA_detections/data/36m_hkaa.json"
    # gt_hka_path = r"F:/Datasets/comp8603/Lower_limb_Xrays/36m_dicoms/split_all/hka_labels.json"
    # # pred_hkaas = "E:/learn_ANU/COMP8603/codes/human_knee_joints_extractions/exps/version1_inf36/key_pointsANDhka_new.csv"
    # pred_hkaas = r"experiments/inf_36m_finetune/measurements.csv"
    # img_type_path = r"F:/Datasets/comp8603/Lower_limb_Xrays/36m_dicoms/type_label.json"
    # err36, hkas_a_36, gt_hkas_a_36, hkas_d_36, gt_hkas_d_36, hkas_36, gt_hkas_36, num_fail_36 = check_hka_mse(pred_hkaas, gt_hka_path, img_type_path)

    
    # # 48
    # print("48m----------------------- \n")
    # gt_labels = r"E:/learn_ANU/COMP8603/codes/MA_detections/data/48m_hkaa.json"
    # gt_hka_path = r"F:/Datasets/comp8603/Lower_limb_Xrays/48m_dicoms/hka_labels.json"
    # # pred_hkaas = "E:/learn_ANU/COMP8603/codes/human_knee_joints_extractions/exps/version1_inf48/key_pointsANDhka_new.csv"
    # pred_hkaas = r"experiments/inf_48m_finetune/measurements.csv"
    # img_type_path = r"F:/Datasets/comp8603/Lower_limb_Xrays/48m_dicoms/type_label.json"
    # err48, hkas_a_48, gt_hkas_a_48, hkas_d_48, gt_hkas_d_48, hkas_48, gt_hkas_48, num_fail_48 = check_hka_mse(pred_hkaas, gt_hka_path, img_type_path)

    
    # hkas = np.concatenate([hkas_12, hkas_24, hkas_36, hkas_48], axis=0)
    # gt_hkas = np.concatenate([gt_hkas_12, gt_hkas_24, gt_hkas_36, gt_hkas_48], axis=0)
    
    # hkas_a = np.concatenate([hkas_a_12, hkas_a_24, hkas_a_36, hkas_a_48], axis=0)
    # gt_hkas_a = np.concatenate([gt_hkas_a_12, gt_hkas_a_24, gt_hkas_a_36, gt_hkas_a_48], axis=0)
    
    # hkas_d = np.concatenate([hkas_d_12, hkas_d_24, hkas_d_36, hkas_d_48], axis=0)
    # gt_hkas_d = np.concatenate([gt_hkas_d_12, gt_hkas_d_24, gt_hkas_d_36, gt_hkas_d_48], axis=0)
    
    # errors_a = np.abs(hkas_a - gt_hkas_a)
    # print(f"Analog: mean {round(np.mean(errors_a), 3)}, std {round(np.std(errors_a), 3)}")
    
    # errors_d = np.abs(hkas_d - gt_hkas_d)
    # print(f"digital: mean {round(np.mean(errors_d), 3)}, std {round(np.std(errors_d), 3)}")
        
    # errors = np.abs(hkas - gt_hkas)
    # print(f"mean {round(np.mean(errors), 3)}, std {round(np.std(errors), 3)}")
    
    # # # get_info(gt_hkas_a, hkas_a)
    # # # get_info(gt_hkas_d, hkas_d)
        
    # get_info(hkas, gt_hkas)
    # print(f"number of failure {((num_fail_12 + num_fail_24 + num_fail_36 + num_fail_48) / (len(err12) + len(err24) + len(err36) + len(err48)))}")
    # bland_altman_plot(hkas, gt_hkas)
    
    # all_errors = {**err12, **err24, **err36, **err48}
    # all_errors = {k: v for k, v in all_errors.items() if v is not None}
    # high_errors = sorted(all_errors.items(), key=lambda y: y[1])[-23:]

    # res = [[] for _ in range(3)]
    # for sample in high_errors:
    #     name = os.path.split(sample[0])[1]
    #     if '12m' in sample[0]:
    #         res[0].append(name)
    #     elif '24m' in sample[0]:
    #         res[1].append(name)
    #     else:
    #         res[2].append(name)
    # print(res)
        
    


    
    
    
    
    # print(f"#------12m-------{hkas_12.shape[1]} valid examples")
    # print(f"error of 12m dataset is: {round(error_12, 3)}")
    # bland_altman_plot(hkas_12[0], hkas_12[1])
    
    # print(f"#------24m-------{hkas_24.shape[1]} valid examples")
    # print(f"error of 24m dataset is: {round(error_24, 3)}")
    # bland_altman_plot(hkas_24[0], hkas_24[1])
    
    # print(f"#------36m-------{hkas_36.shape[1]} valid examples")
    # print(f"error of 36m dataset is: {round(error_36, 3)}")
    # bland_altman_plot(hkas_36[0], hkas_36[1])
    
    # print(f"#------48m-------{hkas_48.shape[1]} valid examples")
    # print(f"error of 48m dataset is: {round(error_48, 3)}")
    # bland_altman_plot(hkas_48[0], hkas_48[1])
    
    # print('all')
    # hkas_all = np.concatenate([hkas_12, hkas_24, hkas_36, hkas_48], axis=1)
    # bland_altman_plot(hkas_all[0], hkas_all[1])
    
    

    
            
        
    
    



