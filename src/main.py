import json
import pandas as pd
import os
import numpy as np
from sklearn.metrics import f1_score

from threshold_per_class import compute_thresholds
from postprocessing import post_process, test_postprocess
from evaluate_postprocessing import prepare_evaluation_set, best_thresholds
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
        post_process(input=activations_path,output=postprocesing_path, threshold = classes_thresholds, 
                    class_names=class_names, kernel = kernel)
        experiments_counter += 1
        print(f"Postprocessing for {experiment_name} completed. {experiments_counter}/{len(os.listdir(thresholds_path))-1}")
    
    with open(postprocessing_checkpoint,'w') as f:
        f.write("postprocessing_complete")
    
    print('Postprocessing complete.')


def run_evaluation(ground_truth_path, output_results_path, class_names, thresholds_path):
    """
    Evaluates postprocessed prediction results against ground truth data, computes F1 scores, and saves evaluation reports.
    Args:
        ground_truth_path (str): Path to the ground truth data.
        output_results_path (str): Directory where prediction results and evaluation outputs are stored.
        class_names (list): List of class names for evaluation.
        thresholds_path (str): Path to the thresholds file used for evaluation.
    Side Effects:
        - Computes and saves per-class and macro F1 scores for each experiment in JSON format.
        - Determines best thresholds and saves a summary report and result files.
        - Writes hyperparameters and an evaluation completion checkpoint to output directories.
    Raises:
        AssertionError: If the shape of predictions does not match the shape of detections.
    """

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
        detections, files_order = prepare_evaluation_set(ground_truth_path, class_names,files_path)
        detections = np.array(detections)
        
        
        pred_file_path = os.path.join(predictions_folder, pred_file)
        pred_dict = json.load(open(pred_file_path))

        # Ensure predictions is a list of lists and matches detections in shape
        predictions = np.array(list(pred_dict.values()))

        assert predictions.shape == detections.shape, "Shape mismatch between detections and predictions"
        print(f"Evaluating experiment: {pred_file}")

        f1_macro = f1_score(detections, predictions, average='macro', zero_division=0)
        per_class_f1 = f1_score(detections, predictions, average=None, zero_division=0)

        f1_class = {c[0]:round(c[1],4) for c in list(zip(class_names,per_class_f1))}
        f1_class['Macro'] = round(f1_macro,4)    

        print(f"F1-Macro for experiment {experiment_name}: {f1_macro:.4f}")
        
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

def run_test(ground_truth, result, activations_path, output_results_path, class_names):

    result_data = json.load(open(result))
    predictions_path = os.path.join(output_results_path,"predictions")
    test_postprocess(input=activations_path,output=predictions_path, result=result_data, 
                    class_names=class_names)

    pred_file = os.path.join(output_results_path,'predictions.predictions.json')

    if os.path.exists(pred_file):
        pred_dict = json.load(open(pred_file))
        files_path = os.path.join(predictions_path,"jsons")
        detections, _ = prepare_evaluation_set(ground_truth, class_names,files_path)
        detections = np.array(detections)
        predictions = np.array(list(pred_dict.values()))

        assert predictions.shape == detections.shape, "Shape mismatch between detections and predictions"
        print(f"Evaluating experiment: {pred_file}")

        f1_macro = f1_score(detections, predictions, average='macro', zero_division=0)
        per_class_f1 = f1_score(detections, predictions, average=None, zero_division=0)

        f1_class = {c[0]:round(c[1],4) for c in list(zip(class_names,per_class_f1))}
        f1_class['Macro'] = round(f1_macro,4)    

        print(f"F1-Macro for experiment: {f1_macro:.4f}")
        json.dump(f1_class, open(os.path.join(output_results_path,'test_evaluation.json'),'w'), indent=3)


if __name__ == "__main__":   

    args = utils.parse_args(result_path = hparams.output_path)
    output_results_path = os.path.join("./results",args.name)

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

    if args.test:
        result_file = args.test_file
        output_results_path = os.path.join(output_results_path,'test')
        run_test(hparams.ground_truth_path, result_file, hparams.activations_path,output_results_path,class_names)
    
    else:
        result_path = os.path.join("./results",args.name)
        if os.path.exists(result_path):
            reset_pipeline(output_results_path, step=args.step)  # Reset from postprocessing onwards

        thresholds_path = run_threshold_computation(hparams.activations_path, output_results_path, class_names, 
                                files, labels, hparams.percentiles, hparams.weighted, hparams.kernels)

        run_postprocessing(thresholds_path,hparams.activations_path,output_results_path,class_names)

        run_evaluation(hparams.ground_truth_path,output_results_path,class_names, thresholds_path)


     

