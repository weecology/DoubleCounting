# Run the pipeline
python create_image_pairs_for_annotation.py \
    --img_left /blue/ewhite/everglades/Airplane/annotations/Images/Jupiter/DSC_2520.JPG \
    --img_right /blue/ewhite/everglades/Airplane/annotations/Images/Jupiter/DSC_2522.JPG \
    --model_path /blue/ewhite/everglades/Zooniverse/20220910_182547/species_model.pl \
    --user ben \
    --host serenity.ifas.ufl.edu \
    --key_filename /home/b.weinstein/.ssh/id_rsa.pub \
    --label_studio_url https://labelstudio.naturecast.org/ \
    --label_studio_project "Airplane Colonies" \
    --label_studio_folder '/pgsql/retrieverdash/everglades-label-studio/everglades-data' \
