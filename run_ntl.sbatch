#!/bin/bash
#SBATCH -t 0-2:01:00
#SBATCH --mem=20G
#SBATCH -J galaxy_ntl
#SBATCH --mail-type=END,FAIL,TIME_LIMIT  # send notification emails
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -N 1                          # request slots on 1 node
##SBATCH --gpus=v100:1                    # request 1 Volta V100 GPU
##SBATCH --partition=informatik-mind      # run on one of our DGX servers
#SBATCH --gres=gpu:a30:1
#SBATCH --output=junk/grid_%j_out.txt         # capture output
#SBATCH --error=junk/grid_%j_err.txt          # and error streams

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate anomaly_env


# python Launch_Exps.py --config-file config_galaxy.yml --dataset-name galaxy
# python Extract_img_features.py
# python umap_viz.py
python eval.py
# python eval/grid.py
# python eval/filter.py

conda deactivate