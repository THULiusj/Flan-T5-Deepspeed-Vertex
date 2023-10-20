import os
import json
import sys
import subprocess
import time
import multiprocessing
import argparse

def parse_args():        
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/flan-t5-large",
        help="Default output directory where all artifacts will be written."
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="/gcs/lsj-public/deepspeed/data/train",
        help="train dataset directory"
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="/gcs/lsj-public/deepspeed/data/eval",
        help="test dataset directory"
    )
    #parser.add_argument(
    #    "--aip_model_dir",
    #    type=str,
    #    default=os.environ['AIP_MODEL_DIR'],
    #    help="Default output directory where all artifacts will be written."
    #)
    #parser.add_argument(
    #    "--aip_tensorboard_log_dir",
    #    type=str,
    #    default=os.environ["AIP_TENSORBOARD_LOG_DIR"],
    #    help="Default Tensorboard log directory."
    #)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=3,
        help="epoch number"
    )
    parser.add_argument(
        "--GPU_count",
        type=int,
        default=2,
        help="GPU count per node"
    )
    args = parser.parse_args()
    
    return args

def launch(cmd):
    """ launch deepspeed process
    """
    return subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)


if __name__ == '__main__':
    args = parse_args()
    cluster_config_str = os.environ.get('CLUSTER_SPEC')
    print(cluster_config_str)
    cluster_config_dict  = json.loads(cluster_config_str)
    print(cluster_config_dict)
    print(cluster_config_dict['task']['type'])
    print(cluster_config_dict["cluster"]["workerpool0"])
    print(cluster_config_dict["cluster"]["workerpool1"])
    workerpool_type = cluster_config_dict['task']['type']
    
    with open("./hostfile", "w") as f:
        for index in range(0,2):
            for node in cluster_config_dict["cluster"][f"workerpool{index}"]:
                f.write(node.split(":")[0] + f" slots={args.GPU_count}\n")
    f.close()
    with open("./hostfile", "rb+") as f:
        f.seek(-1, os.SEEK_END)
        f.truncate()
    f.close()
    with open("./hostfile", "r") as f:
        print(f.read())
    
    
    print("===============pring parameters here==================")
    print(args.batch_size)
    print(args.epoch)
    
    launch('sudo service ssh start')
    launch('cp /etc/environment environment')
    launch('env >> environment')
    launch('sudo cp environment /etc/environment')
    time.sleep(120)
       
    if workerpool_type == "workerpool0":
        launch(f'deepspeed --hostfile=./hostfile run_seq2seq_deepspeed-args.py --batch_size={args.batch_size} --epoch={args.epoch} --train_dataset_path={args.train_dataset_path} --test_dataset_path={args.test_dataset_path} --model_id={args.model_id} --model_output_dir=$AIP_MODEL_DIR --tensorboard_log_dir=$AIP_TENSORBOARD_LOG_DIR')

    elif workerpool_type == "workerpool1":
        time.sleep(5 * 24 * 60 * 60) #5 * 24hours

    
