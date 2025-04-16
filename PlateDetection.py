import kagglehub
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import imutils
import pytesseract
from ultralytics import YOLO
import yolov5

# Download latest version
path = kagglehub.dataset_download("piotrstefaskiue/poland-vehicle-license-plate-dataset")
model = yolov5.load('keremberke/yolov5n-license-plate')
pytesseract.pytesseract.tesseract_cmd = r'H:\Python\tess\tesseract.exe'
photos_path = os.path.join(path, "photos")

# ===== Przetwarzanie zdjęć =====
for image_file in photos_path:
    image_path = os.path.join(photos_path, image_file)
    img = cv2.imread(image_path)

    results = model(img)[0]
    found_plate = False

    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        if label == "license-plate":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = img[y1:y2, x1:x2]

            # Pokaż wyciętą tablicę (opcjonalnie)
            cv2.imshow("Tablica", plate_img)
            cv2.waitKey(500)  # pokazuj 0.5 sekundy
            cv2.destroyAllWindows()

            # Przekształcenie do odczytu OCR
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR
            config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            text = pytesseract.image_to_string(thresh, config=config)
            print(f"[{image_file}] ➜ Odczytany numer tablicy: {text.strip()}")
            found_plate = True
            break

    if not found_plate:
        print(f"[{image_file}] ❌ Nie znaleziono tablicy")

# Display a few images from the 'photos' folder
if os.path.exists(photos_path):
    photos = os.listdir(photos_path)[:5]  # Get the first 5 images
    for photo in photos:
        photo_path = os.path.join(photos_path, photo)
        if os.path.isfile(photo_path):
            img = Image.open(photo_path)
            plt.imshow(img)
            plt.title(photo)
            plt.axis("off")
            plt.show()
else:
    print("Folder 'photos' not found in the dataset.")

