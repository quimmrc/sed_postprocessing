import json
import pandas as pd
import os
import numpy as np

from threshold_per_class import compute_thresholds
from postprocessing import post_process
from evaluate_postprocessing import prepare_evaluation_set, best_thresholds, evaluate_predictions
from utils import get_unique_path
import hparams
import utils

def run_threshold_computation(activations_path, output_results_path, class_names, 
                              files, labels, percentiles, weighted, kernels):
    """Step 1: Compute thresholds per class"""

    thresholds_path = os.path.join(output_results_path, "thresholds_per_class")
    checkpoint_file = os.path.join(thresholds_path,".threshold.complete")

    if os.path.exists(checkpoint_file):
        print("Thresholds already computed. Skipping...")
        return thresholds_path
    
    print("Computing thresholds...")
    os.makedirs(thresholds_path, exist_ok=True)
    for kernel in kernels:
        for wgt in weighted:
            compute_thresholds(files, labels, class_names, activations_path, percentiles, wgt, thresholds_path, kernel)
    
    with open(checkpoint_file,'w') as f:
        f.write("threshold_computation_complete")

    print('Threshold computation completed.')
    return thresholds_path

def run_postprocessing(thresholds_path, activations_path, output_results_path, 
                      class_names):
    """Step 2: Run postprocessing for all threshold files"""

    postprocessing_checkpoint = os.path.join(output_results_path,'.postprocessing_complete')

    if os.path.exists(postprocessing_checkpoint):
        print("Postprocessing already completed. Skipping...")
        return
    
    print("Starting postprocessing...")
    experiments_counter = 0
    for threshold_file in os.listdir(thresholds_path):

        if not threshold_file.endswith('.json'):
            continue
        kernel = int(os.path.splitext(threshold_file)[0].split('k')[-1])
        experiment_name = os.path.splitext(threshold_file)[0].split('_')[-1]
        print(f"Postprocessing for experiment {experiment_name} initialized.")   
        postprocesing_path = os.path.join(output_results_path, "predictions",experiment_name)
        input_path = os.path.join(thresholds_path, threshold_file)
        classes_thresholds = json.load(open(input_path))
        classes_thresholds = [t for t in classes_thresholds.values()]
        post_process(method='empirical',input=activations_path,output=postprocesing_path, threshold = classes_thresholds, 
                    class_names=class_names, kernel = kernel)
        experiments_counter += 1
        print(f"Postprocessing for {experiment_name} completed. {experiments_counter}/{len(os.listdir(thresholds_path))-1}")
    
    with open(postprocessing_checkpoint,'w') as f:
        f.write("postprocessing_complete")
    
    print('Postprocessing complete.')


def run_evaluation(ground_truth_path, output_results_path, class_names, thresholds_path):

    evaluation_checkpoint = os.path.join(output_results_path,'.evaluation_complete')

    print("Starting evaluation...")

    f1_folder = os.path.join(output_results_path,'f1-scores')
    os.makedirs(f1_folder, exist_ok=True)
    predictions_folder = os.path.join(output_results_path,'predictions')
    for pred_file in os.listdir(predictions_folder):
        if not pred_file.endswith('.json'):
            continue
        
        experiment_name = pred_file.split('.')[0]
        files_path = os.path.join(predictions_folder,experiment_name,'jsons')
        
        pred_file_path = os.path.join(predictions_folder, pred_file)
        pred_dict = json.load(open(pred_file_path))

        f1_class = evaluate_predictions(ground_truth_path, class_names, files_path, pred_dict, experiment_name)
        
        k_folder = experiment_name[experiment_name.find('k'):]
        os.makedirs(os.path.join(f1_folder,k_folder), exist_ok=True)
        json.dump(f1_class, open(f"{os.path.join(f1_folder, k_folder,f'{experiment_name}_scores')}.json", 'w'),indent=3)            

    report, result = best_thresholds(f1_folder,class_names, thresholds_path, hparams.percentiles)
    json.dump(report, open(f"{os.path.join(output_results_path,'report')}.json",'w'), indent = 3)
    json.dump(result, open(f"{os.path.join(output_results_path,'result')}.json",'w'))
    with open(f"{os.path.join(output_results_path,'hparams')}.txt", "w") as f:
        for key in dir(hparams):
            if not key.startswith("__"):
                value = getattr(hparams, key)
                f.write(f"{key}: {value}\n")
    with open(evaluation_checkpoint,'w') as f:
        f.write("evaluation_complete")


def reset_pipeline(output_results_path, step=None):
    """Reset pipeline from a specific step"""
    checkpoints = {
        'thresholds': os.path.join(output_results_path, "thresholds_per_class", ".threshold.complete"),
        'postprocessing': os.path.join(output_results_path, ".postprocessing_complete"),
        'evaluation': os.path.join(output_results_path, ".evaluation_complete")
    }
    
    if step is None:
        # Reset all
        for checkpoint in checkpoints.values():
            if os.path.exists(checkpoint):
                os.remove(checkpoint)
        print("All pipeline checkpoints reset.")
    else:
        # Reset from specific step
        steps_order = ['thresholds', 'postprocessing', 'evaluation']
        reset_from_idx = steps_order.index(step)
        
        for i in range(reset_from_idx, len(steps_order)):
            checkpoint = checkpoints[steps_order[i]]
            if os.path.exists(checkpoint):
                os.remove(checkpoint)
        print(f"Pipeline reset from {step} step.")

def _run_test_evaluation(ground_truth, pred_file, predictions_path, class_names, output_results_path):
    """Common test evaluation logic for both empirical and percentual methods"""
    if os.path.exists(pred_file):
        with open(pred_file) as f:
            pred_dict = json.load(f)
        files_path = os.path.join(predictions_path, "jsons")
        
        f1_class = evaluate_predictions(ground_truth, class_names, files_path, pred_dict, pred_file)
        
        os.makedirs(output_results_path, exist_ok=True)
        with open(os.path.join(output_results_path, 'test_evaluation.json'), 'w') as tf:
            json.dump(f1_class, tf, indent=3)

def run_test(ground_truth, result, activations_path, output_results_path, class_names):

    result_data = json.load(open(result))
    predictions_path = os.path.join(output_results_path,"predictions")
    post_process(method='empirical',input=activations_path,output=predictions_path,threshold=None,class_names=class_names,
                kernel=0, test=True, result=result_data)
    
    pred_file = os.path.join(output_results_path,'predictions.predictions.json')
    _run_test_evaluation(ground_truth, pred_file, predictions_path, class_names, output_results_path)


if __name__ == "__main__":   

    args = utils.parse_args(result_path = hparams.output_path)
    result_path = os.path.join(args.output_path,args.name)

    # Load data
    gt = pd.read_csv(hparams.ground_truth_path) 
    files = list(gt["fname"])
    labels = (gt["labels"])
    eval_labels = set()
    for label_str in labels:
        eval_labels.update(label_str.split(','))

    class_names = json.load(open(hparams.class_names))

    # Validate class names
    if not all(label in class_names for label in eval_labels):
        class_names = [c.replace(' ','_') for c in class_names]
        if not all(label in class_names for label in eval_labels):
            missing_labels = [label not in class_names for label in labels]
            raise ValueError("Some labels in your ground truth are not present in class_names file." \
            "Make sure both sets of labels share the same format. ")
        
    if args.method == 'empirical':
        if args.test:
            result_file = args.test_file
            output_results_path = os.path.join(result_path,'test')
            run_test(args.method, hparams.ground_truth_path, result_file, hparams.activations_path,output_results_path,class_names)
        
        else:
            if os.path.exists(result_path):
                reset_pipeline(result_path, step=args.step)  # Reset from postprocessing onwards

            thresholds_path = run_threshold_computation(hparams.activations_path, result_path, class_names, 
                                    files, labels, hparams.percentiles, hparams.weighted, hparams.kernels)

            run_postprocessing(thresholds_path,hparams.activations_path,result_path,class_names)
            run_evaluation(hparams.ground_truth_path,result_path,class_names, thresholds_path)
    
    elif args.method == 'percentual':
        
        if args.test:
            pred_file = os.path.join(hparams.output_path, f"{args.name}.predictions.json")
            predictions_path = os.path.join(hparams.output_path, args.name)
            output_results_path = os.path.join(hparams.output_path, 'test')
            _run_test_evaluation(hparams.ground_truth_path, pred_file, predictions_path, class_names, output_results_path)            
        else:
            thr = hparams.min_threshold
            post_process(method=args.method, input=hparams.activations_path, output=result_path, threshold=thr, class_names=class_names, kernel=0)
        


     

