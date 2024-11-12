from ultralytics import YOLO
import os

model = YOLO("yolov11_custom.pt")
# Set the directory containing the images
folder_path = "/home/ace/P_research/Data/archive/FP"

count = 0

# Loop through all .jpg files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        
        # Predict on the current image
        model.predict(
            source=image_path,
            show=False,
            save=True,
            conf=0.6,
            line_width=1,
            save_crop=False,
            save_txt=True,
            show_labels=True,
            show_conf=True,
            classes=[0, 1, 2, 3, 4]
        )
	# Increment the counter
        count += 1
        # Stop after the first 20 images
        if count >= 250:
            break

