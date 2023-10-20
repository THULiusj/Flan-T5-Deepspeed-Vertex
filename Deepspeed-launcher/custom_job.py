""" Custom training pipeline """

from google.cloud import aiplatform
from datetime import datetime

BUCKET = 'gs://lsj-public/deepspeed'
PROJECT_ID = 'PROJECT_ID'
LOCATION = 'us-central1'
SERVICE_ACCOUNT = 'SERVICE_ACCOUNT'
TENSORBOARD_RESOURCE = 'projects/703099487153/locations/us-central1/tensorboards/2069606338916253696'
TRAIN_IMAGE="us-central1-docker.pkg.dev/argolis-lsj-test/t5/finetuning_flan_t5_large:multi-node-args-vertexai-0905"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")


# Initialize the *client* for Vertex
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET, location=LOCATION)

job = aiplatform.CustomContainerTrainingJob(
    display_name="flant5_large_deepspeed_" + TIMESTAMP,
    container_uri=TRAIN_IMAGE,
    command=["python", "get_hostfile.py", "--epoch=1", "--batch_size=8", "--train_dataset_path=/gcs/lsj-public/deepspeed/split_data_2/train", "--test_dataset_path=/gcs/lsj-public/deepspeed/split_data_2/eval", "--GPU_count=2"],
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-12:latest"
)

# multi-node
# Reduction server does support Reduction Server
model = job.run(
    model_display_name='flan-t5-large-finetuning-gpu-deepspeed-multi-node',
    #args=["--epoch=1", "--batch_size=16"],
    replica_count=4,
    service_account = SERVICE_ACCOUNT,
    tensorboard = TENSORBOARD_RESOURCE,
    boot_disk_size_gb=600,
    machine_type="n1-standard-32",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count = 2,
    #reduction_server_replica_count=4,
    #reduction_server_machine_type="n1-standard-32",
    #reduction_server_container_uri="us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver:latest",
    enable_web_access=True
)
