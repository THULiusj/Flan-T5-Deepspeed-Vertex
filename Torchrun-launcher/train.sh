#!/bin/bash

# Entrypoint of SD training docker container.
# Discover multi nodes topology and call trainer.

# example:
# CLUSTER_SPEC='{"cluster":{"workerpool0":["cmle-training-workerpool0-4c1d6d97d0-0:2222"],"workerpool1":["cmle-training-workerpool1-4c1d6d97d0-0:2222","cmle-training-workerpool1-4c1d6d97d0-1:2222","cmle-training-workerpool1-4c1d6d97d0-2:2222"]},"environment":"cloud","task":{"type":"workerpool1","index":3},"job":"{\"python_module\":\"\",\"package_uris\":[],\"job_args\":[]}","open_ports":[3333]}'

# NCCL setup
# export NCCL_DEBUG=INFO
# export NCCL_NSOCKS_PERTHREAD=1
# export NCCL_SOCKET_NTHREADS=2
# export NCCL_MIN_NCHANNELS=1

# huggingface setup
# export HF_HUB_OFFLINE=1
#export HF_HUB_DISABLE_TELEMETRY=1

if [[ -z $CLUSTER_SPEC ]]

then
    echo "========== Launch on local machine =========="
    
    set -x
    torchrun --nnodes=1 --node_rank=0 run_seq2seq_deepspeed.py

else
    echo "========== Launch on cloud =========="
    echo "CLUSTER_SPEC:" $CLUSTER_SPEC
    
    primary_node=`echo $CLUSTER_SPEC | jq -r '.cluster.workerpool0[0]'`
    
    IFS=':' read -ra primary_node_split <<< $primary_node
    primary_node_addr=${primary_node_split[0]}
    primary_node_port=${primary_node_split[1]}

    workerpool=`echo $CLUSTER_SPEC | jq -r '.task.type'`
    if [[ $workerpool = "workerpool0" ]]
    then
        node_rank=0
    else
        node_rank=`echo $CLUSTER_SPEC | jq -r '.task.index'`
        node_rank=$(($node_rank + 1))
    fi
    workerpool1_nodes=`echo $CLUSTER_SPEC | jq -r '.cluster.workerpool1 | length'`
    num_nodes=$(($workerpool1_nodes + 1))
    
    echo "primary node address: " $primary_node_addr
    echo "primary node port: " $primary_node_port
    echo "num nodes: " $num_nodes
    echo "node rank: " $node_rank
    
    if [[ $num_nodes = 1 ]]
    then
        set -x
        torchrun --nnodes=1 --node_rank=0 $@
    else
        set -x
        torchrun \
        --nnodes=$num_nodes \
        --node_rank=$node_rank \
        --master_addr=$primary_node_addr \
        --master_port=$primary_node_port \
        $@
    fi

fi