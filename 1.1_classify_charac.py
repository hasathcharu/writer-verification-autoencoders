import os
import sys
from PIL import Image
from ultralytics import YOLO
from modules.utils import constants as c

MODEL_PATH = f'yolo.pt'
INPUT_ROOT = f'{c.DATA_PATH}/binary_lines'
OUTPUT_ROOT = f'{c.DATA_PATH}/{c.CHARS}'
CONF_THRESHOLD = 0.5
RESIZE_DIM = (40, 40)

model = YOLO(MODEL_PATH)

for writer_id in os.listdir(INPUT_ROOT):
    writer_path = os.path.join(INPUT_ROOT, writer_id)
    if not os.path.isdir(writer_path):
        continue

    for sample in os.listdir(writer_path):
        sample_path = os.path.join(writer_path, sample)
        if not os.path.isdir(sample_path):
            continue

        for speed in os.listdir(sample_path):
            speed_path = os.path.join(sample_path, speed)
            if not os.path.isdir(speed_path):
                continue

            for img_file in os.listdir(speed_path):
                if not img_file.lower().endswith('.png'):
                    continue

                img_path = os.path.join(speed_path, img_file)
                line_id = os.path.splitext(img_file)[0]
                dest_dir = os.path.join(OUTPUT_ROOT, writer_id, sample, speed, line_id)
                os.makedirs(dest_dir, exist_ok=True)

                results = model(img_path, conf=CONF_THRESHOLD)[0]
                image = Image.open(img_path)

                for i, box in enumerate(results.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    crop = image.crop((x1, y1, x2, y2))
                    resized = crop.resize(RESIZE_DIM, Image.LANCZOS)
                    crop_filename = f"{writer_id}_{sample}_{speed}_{line_id}_{i}.png"
                    resized.save(os.path.join(dest_dir, crop_filename))

print("Done! Cropped character images are saved under:", OUTPUT_ROOT)
