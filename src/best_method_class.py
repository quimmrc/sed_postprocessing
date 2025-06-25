import json
import numpy as np
import os

if __name__ == "__main__":

    class_names = json.load(open('./model/fsd-sinet-vgg42-tlpf_aps-1.json'))['classes']
    best_method_per_class = {k: (0, None) for k in class_names}
    best_thresholds = {k: 0 for k in class_names}
    f1_scores_path = '/home/usuario/master-thesis/dataset_results/f1_scores'
    output = '/home/usuario/master-thesis/dataset_results/best_adaptive_thresholds'

    for file in os.listdir(f1_scores_path):
        
        full_score = json.load(open(f'{f1_scores_path}/{file}'))
        fname = os.path.splitext(file)[0]
        method = fname.split('_')[-1]
        if method=='00':
            continue
        
        for label, f1 in full_score.items():
            if label == 'Macro':
                continue
            elif best_method_per_class[label][0] < f1:
                best_method_per_class[label] = (f1, method)           

    selected_methods = [m[1] for m in best_method_per_class.values()]
    selected_methods = [(m, selected_methods.count(m)) for m in set(selected_methods)]
    
    for label, (f1,method) in best_method_per_class.items():
        format_label = label.replace(' ','_')
        try:
            threshold_dict = json.load(open(f"class_thresholds_{method}.json"))
        except FileNotFoundError:
            continue
        best_thresholds[label] = threshold_dict[format_label]

    print(selected_methods)
    
    json.dump(best_thresholds,open(f"{output}/best_thresholds.json",'w'),indent=3)