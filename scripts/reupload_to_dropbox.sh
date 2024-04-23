# Use rclone to look for images in dropbox
rclone copy /blue/ewhite/everglades/Airplane/predictions UFdropbox:/Airplane_images_to_predict/processed
rclone move UFdropbox:/Airplane_images_to_predict/* UFdropbox:/Airplane_images_to_predict/processed