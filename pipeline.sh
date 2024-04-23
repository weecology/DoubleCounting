#!/bin/bash
#SBATCH --job-name=AirplanePrediction   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/Airplane_prediction_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/Airplane_prediction_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

source activate EvergladesTools

# Run the pipeline
python scripts/look_for_images.sh \
    --source UFdropbox:/Airplane_images_to_predict/ \
    --destination /blue/ewhite/everglades/Airplane/images_to_predict

python scripts/predict.py \
    --model_path /blue/ewhite/everglades/Zooniverse/20220910_182547/species_model.pl \
    --save_dir /blue/ewhite/everglades/Airplane/predictions \
    --user ben \
    --host serenity.ifas.ufl.edu \
    --key_filename /home/b.weinstein/.ssh/id_rsa.pub \
    --label_studio_url https://labelstudio.naturecast.org/ \
    --label_studio_project "Airplane Photos" \
    --label_studio_folder '/pgsql/retrieverdash/everglades-label-studio/everglades-data' \
    --image_dir /blue/ewhite/everglades/Airplane/images_to_predict