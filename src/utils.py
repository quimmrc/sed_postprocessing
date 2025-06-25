import os
import argparse


def parse_args(result_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='Name for the experiment')
    parser.add_argument('--test', action='store_true', help='Enable test mode')
    parser.add_argument('--test-file', type=str, help='Test file path (required with --test): json file containing thresholds per sorted set of classes')
    args = parser.parse_args()
    args.step = None

    if args.test:
        result_path = os.path.join(result_path,args.name,'result.json')
        if os.path.exists(result_path):
            args.test_file = result_path
        else:
            parser.error("Could not find a result.json file in the experiments results folder.")
    else:
        result_path = os.path.join("./results",args.name)
        if os.path.exists(result_path):
            print("Warning: this experiment already exists and its results will be overwritten from the pipeline step you choose. Do you want to continue? [y/N]")
            confirm = input().strip().lower()
            if confirm == 'y':
                print("Choose the step from which you want to reset the pipeline: [thresholds (t), postprocess (p), evaluation (e)]")
                choice = input().strip().lower()
                if choice == 't':
                    args.step = 'thresholds'
                elif choice == 'p':
                    args.step = 'postprocessing'
                elif choice == 'e':
                    args.step = 'evaluation'
                else:
                    print("Invalid step. Operation cancelled.")
                    exit(0)
            else:
                print("Operation cancelled by user.")
                exit(0)

    return args

def get_unique_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    
    base, ext = os.path.splitext(base_path)
    counter = 1
    
    while True:
        new_path = f"{base}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1