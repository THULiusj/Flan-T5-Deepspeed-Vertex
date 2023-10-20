# Flan-T5 fine-tuning on Veretx AI using Deepspeed in distributed cluster

## Introduction
This repo is based on [this repo](https://github.com/rafaelsf80/vertex-flant5xxl-multitask-finetuning), which is Flan-T5 fine-tuning with Deepspeed on Vertex AI but in single node. This repo modifies the code to distributed server.

Distributed training are implemented in two modes. One mode is SSH passwordless authentication for Deepspeed launcher, the other mode is to leverage torchrun laucher to replace Deepspeed launcher, which will not require SSH passwordless authentication. So you can see codes of two modes in the two folders.

## Details

### Job details
- Data and downstream task: CNN-dailymail, News summary
- Model: google/flan-T5-large, 3GB
- Full Fine tune
- Based Library: Huggingface Transformers + Deepspeed

### Deepspeed launcher
#### SSH no password authentication
Deepspeed launcher will automatically ssh to each node, and launch the training process in it. So SSH no password authentication needs to be configured in Dockerfile.
Reference: https://github.com/Azure/azureml-examples/blob/main/cli/jobs/deepspeed/deepspeed-autotuning/src/start-deepspeed.sh

```
# install SSH server
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev
RUN sudo apt-get install -y telnet

#config ssh port and no authentication
ENV SSH_PORT=2222
RUN cat /etc/ssh/sshd_config > ${STAGE_DIR}/sshd_config && \
        sed "0,/^#Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\\s*required\\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# copy keys
ADD id_rsa /root/.ssh/id_rsa
RUN chmod 0400 /root/.ssh/id_rsa
RUN touch /root/.ssh/config && \
    echo -e "Port 2222\n StrictHostKeyChecking no\n  UserKnownHostsFile=/dev/null" > /root/.ssh/config
RUN chmod 600 /root/.ssh/config
ADD id_rsa.pub /root/.ssh/id_rsa.pub
ADD authorized_keys /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/config
RUN chmod 600 /root/.ssh/authorized_keys
RUN chmod 700 /root/.ssh/
```

#### Hostfile
Deepspeed needs hostfile to get master node and worker nodes name.
Vertex AI dynamic provisioned machines/nodes through GKE. The node names can only be known after provisioned successfully, and acquired by environment variables CLUSTER_SPEC set by Vertex AI.

CLUSTER_SPEC:
```
# workerpool0: master node.
# workerpool1: worker node.
# workerpool2: reduction server.
{
   "cluster":{
      "workerpool0":[
         "cmle-training-workerpool0-ab-0:2222"
      ],
      "workerpool1":[
         "cmle-training-workerpool1-ab-0:2222",
         "cmle-training-workerpool1-ab-1:2222"
      ],
      "workerpool2":[
         "cmle-training-workerpool2-ab-0:2222",
         "cmle-training-workerpool2-ab-1:2222"
      ],
      "workerpool3":[
         "cmle-training-workerpool3-ab-0:2222",
         "cmle-training-workerpool3-ab-1:2222",
         "cmle-training-workerpool3-ab-2:2222"
      ]
   },
   "environment":"cloud",
   "task":{
      "type":"workerpool0",
      "index":0,
      "trial":""
   },
   "job": {
      ...
   }
}
```
Dynamic get nodes name and port, and create hostfile.
```
    cluster_config_str = os.environ.get('CLUSTER_SPEC')
    cluster_config_dict  = json.loads(cluster_config_str)
    workerpool_type = cluster_config_dict['task']['type']
    with open("./hostfile", "w") as f:
        for index in range(0,2):
            for node in cluster_config_dict["cluster"][f"workerpool{index}"]:
                f.write(node.split(":")[0] + " slots=2\n")
    f.close()
    # remove the enter/n
    with open("./hostfile", "rb+") as f:
        f.seek(-1, os.SEEK_END)
        f.truncate()
    f.close()
```
#### Environment variables update
When deepspeed launcher launches the multiple processes. Deepspeed starts to ssh to all the nodes(master node and worker nodes) and launch the python process in it. An issue is that all the environment variables are configured dynamically by Vertex AI in the environment. In ssh session, all environment variables are empty. We need to fix all the environment variables by adding it to the env file.

```
launch('cp /etc/environment environment')
launch('env >> environment')
launch('sudo cp environment /etc/environment')
```
#### Deepspeed start
We can use Deepspeed launcher to start the multiple processes. Then, we just use deepspeed command in the master node. Other worker nodes will be in sleep mode to make the node be active and not recycled by Vertex AI.
```
if workerpool_type == "workerpool0":
        launch('deepspeed --hostfile=./hostfile run_seq2seq_deepspeed.py')   
elif workerpool_type == "workerpool1":
        time.sleep(24 * 60 * 60)
```

### Torchrun launcher
If use torchrun, no need to configure SSH passwordless authentication and Hostfile. We just read node information from CLUSTER_SPEC and start the process using Torchrun in each node.
```
torchrun \
--nnodes=$num_nodes \
--node_rank=$node_rank \
--master_addr=$primary_node_addr \
--master_port=$primary_node_port \
$@
```

### Submit training job in Vertex AI
You can reference the both custom_job.py and flant5-train.ipynb. There are two different functions for Vertex custom job submission. One is *aiplatform.CustomContainerTrainingJob*, the other is *aiplatform.CustomJob*.

In the sample, training dataset is mounted from GCS. Veretx Tensorboard is integrated, you can directly view Tensorboard from Vertex AI page. 