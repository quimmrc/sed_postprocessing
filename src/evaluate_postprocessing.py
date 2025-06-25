import json
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score


def prepare_evaluation_set(gt_path, class_names, files_path):
    """
    Prepares the evaluation set by filtering ground truth labels for files present in the results directory.

    Parameters:
        gt_path (str): Path to the ground truth CSV file.
        class_names (list): List of class names.
        files_path (str): Path to the directory containing the files to be evaluated.

    Returns:
        list: A list of detection lists, where each detection list contains binary values indicating the presence of each class.
    """
    class_idx = {name.replace(' ','_'): idx for idx,name in enumerate(class_names)}

    gt_df = pd.read_csv(gt_path)
    gt_df = gt_df.sort_values(by="fname")
    gt_labels_idx = [[class_idx[label] for label in labels]
        for labels in (l.split(',') for l in gt_df['labels'])]  

    gt_util = list(zip(gt_df['fname'],gt_labels_idx))

    files = sorted([int(f.split('.')[0]) for f in os.listdir(files_path)])
    gt_filtered = [p for p in gt_util if int(p[0]) in files]
    gt_dict = {p[0]: p[1] for p in gt_filtered}
    detections = []
    files_order = []
    for fname in files:
        files_order.append(str(fname))
        labels = gt_dict.get(fname, []) 
        file_detections = np.array([1 if i in labels else 0 for i in range(len(class_names))])
        detections.append(file_detections)
    return (detections,files_order)    

def best_thresholds(f1_folder, class_names, thresholds_folder, percentiles):

    # I want kernel to be adaptive in the class threhsold decision process
    # so that kernel ran throug the class probabilities for each frame depends
    # on the class

    # the kernel used in post_processing can be the one with the best f1 score and that's that
    kernels_postprocessing = {k: 0 for k in os.listdir(f1_folder) if os.path.exists(k)}
    best_result_per_class = {c: (0,None) for c in class_names}
    best_result_per_class = {c: {'f1':0, 'method':None,'thr':0, 'k':0} for c in class_names}
    f1_global = (None, 0)
    for kernel in os.listdir(f1_folder):
        kernel_folder = os.path.join(f1_folder,kernel)
        f1_per_kernel = []
        if os.path.isdir(kernel_folder):
            for scores in os.listdir(kernel_folder):
                scores_path = os.path.join(kernel_folder,scores)
                scores_dict = json.load(open(scores_path))
                method = scores.split('_')[0]
                kernel_key = int(method.split('k')[-1])
                for (label, f1) in scores_dict.items():
                    if label == 'Macro':
                        f1_per_kernel.append(f1)
                        if f1 > f1_global[1]:
                            f1_global = (method,f1)
                        continue
                    elif best_result_per_class[label]['f1'] < f1:
                        thr_file_path = os.path.join(thresholds_folder,f'class_thresholds_{method}.json')
                        if os.path.exists(thr_file_path):
                            best_thr = json.load(open(thr_file_path))
                            best_result_per_class[label]['thr'] = best_thr[label]
                            best_result_per_class[label]['k'] = kernel_key
                        else:
                            best_result_per_class[label]['thr'] = 0
                            print(f"WARNING: Threshold file not found for class {label}, method {method}"
                                  f"File: {thr_file_path}")
                        best_result_per_class[label]['f1'] = f1
                        best_result_per_class[label]['method'] = method
                kernels_postprocessing[kernel] = (np.median(f1_per_kernel))
    method_freq = [str(best_result_per_class[l]['method'])[-2:] for l in class_names]
    kernel_freq = [(k,method_freq.count(k)/len(class_names)) for k in set(method_freq) if k.startswith('k')]
    methods_list = [l['method'] for l in best_result_per_class.values()]
    avg_dist = {'avg': sum('avg' in m for m in methods_list if m is not None),
                'wgt': sum('wgt' in m for m in methods_list if m is not None)}
    percentiles_dist = {str(p): sum(str(p) in m for m in methods_list if m is not None) for p in percentiles}
    report = { 'Averaging distribution': avg_dist,
            'Percentiles distribution': percentiles_dist,
            'Best Kernels distribution': kernel_freq,
            'Best global method': f1_global,
            'Best results per class': best_result_per_class}
    result = [(best_result_per_class[label]['thr'], int(best_result_per_class[label]['k'])) for label in best_result_per_class.keys()]
    return report, result