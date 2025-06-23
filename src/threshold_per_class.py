# The goal of this python script is to extract the threshold for the classes
# when fsd-sinet_v1 activation matches eval_audio activation to check on their behaviours
# In fact, we will also extract the threshold for the non-active cases.
# Our final objective is to set a threshold for each class of the ones that cannot be mapped
# into DESED classes.

import pandas as pd
import json
import os
import numpy as np
from scipy.ndimage import median_filter

def thresholds_postprocessing(thresholds, kernel_size=5, weighted=True):
    """For a given array of thresholds for a class in a file, return the valid threshold.
    The final threshold is computed as follows:
        First we run a median filtering through the whole class probabilities.
        Second we compute for each 
    Args:
        thresholds (_type_): _description_
    """
    med_filter = median_filter(volume = thresholds, size = kernel_size, mode='reflect')
    if all(p == 0 for p in med_filter):
        med_filter = np.array(thresholds)
    mask = med_filter>0
    diff = np.diff(np.concatenate(([False],mask,[False])).astype(int))    
    start = np.where(diff==1)[0]
    end = np.where(diff==-1)[0]

    segments = [med_filter[start:end] for start, end in zip(start,end)]
    segment_lengths = [len(s) for s in segments]
    segmented_thresholds = [np.mean(s)+np.std(s) for s in segments]
    # print(f"Mask: {mask}, Diff: {diff}, Start: {start}, End: {end}")
    if weighted:
        final_threshold = min(round(np.average(segmented_thresholds, weights = segment_lengths),2),1.0)
    else:
        final_threshold = min(round(np.average(segmented_thresholds),2),1.0)
    
    #print(f"Og: {thresholds}\nMedian-filtered: {med_filter}\nSegments: {segments}")
    #print(f"Final threshold: {final_threshold}")
    return float(final_threshold)

def compute_thresholds(files, labels, class_names, activations_path, percentiles, weighted, output_path):

    classes_thr = [[] for _ in range(200)]
    for analysis in os.listdir(activations_path):
        filename = os.path.splitext(analysis)[0]
        sound_id = int(filename.split('_')[0])
        with open(f"{activations_path}/{analysis}",'r') as f:
            activations = json.load(f)

        file_idx = files.index(sound_id)
        labels_in_file = labels[file_idx].split(',')
        thresholds = {class_names.index(label):[0]*len(activations) for label in labels_in_file}
        # Process each frame
        for i, frame_activations in enumerate(activations):
        # Process the activations in each frame
            for class_idx, confidence in frame_activations:
                if class_idx in thresholds.keys():
                    thresholds[class_idx][i] = confidence
                    # print(f"Class: {class_idx}, Frame: {i}: {confidence}")
        
        for key, value in thresholds.items():
            if any(p != 0 for p in value):
                thr = thresholds_postprocessing(label=key, thresholds=value, weighted=weighted)
                if np.isnan(thr):
                    print(f"WARNING: NaN threshold found for class {key}. Threshold vector: {value}")
                classes_thr[key].append(thr)
    

    for percentile in percentiles:
        classes_thr_dict = {class_names[j]: round(float(np.percentile(thrs,percentile)),2) for j,thrs in enumerate(classes_thr)}
        w = 'wgt' if weighted else 'avg'
        fname = os.path.join(output_path, f'class_thresholds_{w}{percentile}.json')
        with open(fname,'w') as f:
            json.dump(classes_thr_dict,f, indent=3)
        

