# useful paths
ground_truth_path = "/home/usuario/FSD50K/FSD50K.ground_truth/eval.csv"
activations_path = "/home/usuario/FSD50K/eval_analysis_frames" 
class_names = '/home/usuario/FSD50K/fsd_sinet_class_names.json'
output_path = '/home/usuario/master-thesis/dataset_results/per_sed_postprocessing'

# useful parameters for empirical method
percentiles = [50,60]
weighted = [True, False]
kernels = [1,3]

# useful parameters for percentual method
min_threshold = 0.5