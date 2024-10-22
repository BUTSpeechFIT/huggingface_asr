# Extension of the HuggingFace Transformers for Automatic Speech Recognition
This repository is still under active development. Proper README and documentation will be added soon.
## Setup

1. BUT SGE Cluster setup
   1. Clone repository and change directory to project root.
   2. Set ENVS_ROOT.
      ```shell
      ENVS_ROOT=/mnt/matylda5/ipoloka/envs
       ```
   3. Create conda environment
      ```shell
      conda create -p "${ENVS_ROOT}/huggingface_asr" python=3.10
       ```
   4. Activate conda environment
       ```shell
      conda activate "${ENVS_ROOT}/huggingface_asr"
       ```
   5. Install requirements
      ```shell
      pip install -r requirements_BUT_cluster.txt
      ```
   6. Extend PYTHONPATH with sources root
      ```shell
      export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
      ```
   7. Run following command and test if help message was printed.
      ```shell
      python src/trainers/train_enc_dec_asr.py -h
      ```

## Training
Recipes are provided in the `recipes` directory. Each recipe contains a `run.sh` script that can be used to train a model. The script contains all the necessary commands to train a model. The script can be run directly or it can be submitted to the SLURM cluster.
