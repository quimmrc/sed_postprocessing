import numpy as np 
from scipy.ndimage import median_filter
import json
import os

def rebuild_activations(raw_data,num_classes):

    activations = np.zeros((len(raw_data),num_classes))
    for i, frame_activations in enumerate(raw_data):
        for idx, p in frame_activations:
            activations[i,idx] = p
    return activations

def post_process(input, output, threshold, class_names, kernel):
    os.makedirs(output+'/jsons/', exist_ok=True)
    os.makedirs(output+'/tbe/', exist_ok=True)
    evaluation_predictions = {}
    files_order = []
    for filename in os.listdir(input):
        files_order.append(filename)
        # NOTE: this could be done just once
        fname = filename.split('_')[0]
        raw_data_path = os.path.join(input,filename)
        raw_data = json.load(open(raw_data_path))
        activations = rebuild_activations(raw_data,len(class_names))
        
        hop_size = 0.5
        detections = []
        filtered_detections = []
        tbe_output = []

        if not isinstance(kernel, int) or kernel <= 0:
            filtered_activations = activations
        else:
            filtered_activations = np.zeros_like(activations)
            for class_idx in range(activations.shape[1]):
                filtered_activations[:,class_idx] = median_filter(activations[:, class_idx], size = kernel, mode='reflect')

        for i, frame_activations in enumerate(filtered_activations):
            classes_confidences = [(class_names[idx[0]], frame_activations[idx[0]]) for idx in np.argwhere(frame_activations > threshold)]
            for class_name, confidence in classes_confidences:
                detections.append({
                    'start_time': i * hop_size,
                    'end_time': (i + 1) * hop_size,
                    'name': class_name,
                    'confidence': float("{:.2f}".format(float(confidence)))  # Reduce precision
                })

        for detection in detections:
            start_time = detection['start_time']
            end_time = detection['end_time']
            name = detection['name']
            is_contiguous_detection = False
            for filtered_detection in filtered_detections:
                if name == filtered_detection['name'] and start_time == filtered_detection['end_time']:
                    # The current detection is continguous to a previous detection with the same name
                    # Update the end time of the existing detection and add no new entry to the list
                    filtered_detection['end_time'] = end_time
                    is_contiguous_detection = True
                    break
            if not is_contiguous_detection:
                filtered_detections.append(detection)
                tbe_output.append(class_names.index(detection['name']))
        
        # put 1 into the index of the class in evaluation predictions
        file_predictions = [1 if i in set(tbe_output) else 0 for i in range(len(class_names))]
        evaluation_predictions[int(fname)] = file_predictions

        output_dict = {
            'detections': filtered_detections, 
            'num_detections': len(filtered_detections),
            'detected_classes': list(set(item['name'] for item in filtered_detections)),
        }

        # Save results to file
        output_result = os.path.join(output, 'jsons', f'{fname}.json')
        json.dump(output_dict, open(output_result, 'w'))
        # Save data for post processing evaluation
        output_eval = os.path.join(output, 'tbe', fname)
        tbe_output = [(l, tbe_output.count(l)) for l in set(tbe_output)]
        json.dump(tbe_output, open(output_eval + '.tbe.post.json', 'w'))
        # print(f'Saved analysis for file {filename}')"""

    json.dump(dict(sorted(evaluation_predictions.items())), open(output +'.predictions.json','w'))

