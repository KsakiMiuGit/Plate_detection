import yolov5
import kagglehub
import os
import random
import cv2
import pytesseract
import xml.etree.ElementTree as ET

model = yolov5.load('keremberke/yolov5n-license-plate')
path = kagglehub.dataset_download("piotrstefaskiue/poland-vehicle-license-plate-dataset")
images_path = os.path.join(path, "photos")

# parametry modelu
model.conf = 0.25 #minimalny próg pewności
model.iou = 0.45 #próg IoU
model.agnostic = False  # nakładające się wykrycia różnych klas nie są usuwane
model.multi_label = False  # wykrycie ma tylko jedną etykietę
model.max_det = 5  # maksymalna liczba wykryć

image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
num_images_to_sample = min(5, len(image_files))
random_images = random.sample(image_files, num_images_to_sample)

os.makedirs('results', exist_ok=True)
os.makedirs('results/plates', exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r'H:\Python\tess\tesseract.exe'

# funkcja do obliczania IoU
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxA[3])

    # oblicz obszar przecięcia
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # oblicz obszar obu prostokątów
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # oblicz IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# wczytaj plik annotations.xml
annotations_path = os.path.join(path, "annotations.xml")
tree = ET.parse(annotations_path)
root = tree.getroot()

# sparsuj dane z XML
annotations = {}
for image in root.findall('image'):
    img_name = image.get('name')
    boxes = []
    for box in image.findall('box'):
        xtl = float(box.get('xtl'))
        ytl = float(box.get('ytl'))
        xbr = float(box.get('xbr'))
        ybr = float(box.get('ybr'))
        plate_number = box.find("attribute[@name='plate number']").text
        boxes.append({'box': (xtl, ytl, xbr, ybr), 'plate_number': plate_number})
    annotations[img_name] = boxes

with open('results/plate_numbers.txt', 'w') as plate_file, open('results/iou_results.txt', 'w') as iou_file:
    for img_name in random_images:
        img_path = os.path.join(images_path, img_name)

        #wykrywanie obiektów za pomocą modelu
        results = model(img_path, size=640)
        results = model(img_path, augment=True) #augmentacja danych może poprawić wyniki wykrywania

        predictions = results.pred[0] #wyniki predykcji dla obrazu
        boxes = predictions[:, :4]  #współrzędne wykrytych obiektów x1, y1, x2, y2
        scores = predictions[:, 4] #pewność dla każdego wykrycia
        categories = predictions[:, 5] #klasy wykrytych obiektów

        img = cv2.imread(img_path)

        #iteracja przez wykryte tablice
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            plate_region = img[y1:y2, x1:x2]

            gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)

            alpha = 1.5  # współczynnik kontrastu
            beta = 20    # wartość jasności
            gray_plate = cv2.convertScaleAbs(gray_plate, alpha=alpha, beta=beta)

            # usuwanie szumów za pomocą rozmycia Gaussa
            gray_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)

            # zwiększenie rozdzielczości obrazu
            scale_factor = 2  # współczynnik skalowania
            height, width = gray_plate.shape
            gray_plate = cv2.resize(gray_plate, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)

            #progowanie Otsu pomaga w wyraźnym rozdzieleniu liter od tła
            _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # przycinanie marginesów
            height, width = thresh_plate.shape
            left_margin = int(width * 0.10)
            other_margin = int(min(height, width) * 0.10)
            thresh_plate = thresh_plate[other_margin:height - other_margin, left_margin:width - other_margin]

            #usuwanie małych szumów za pomocą operacji morfologicznych
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh_plate = cv2.morphologyEx(thresh_plate, cv2.MORPH_OPEN, kernel)

            plate_filename = f"results/plates/{os.path.splitext(img_name)[0]}_plate_{i}.png"
            cv2.imwrite(plate_filename, thresh_plate)

            #wyodrębnienie tekstu
            plate_text = pytesseract.image_to_string(thresh_plate, config='--psm 7')  # psm 7 to 1 linia tekstu

            plate_text = plate_text.strip()
            if plate_text:
                plate_file.write(f"{img_name}: {plate_text}\n")
                print(f"Detected plate: {plate_text} in {img_name}")

            ground_truth_boxes = annotations.get(img_name, [])

            for gt in ground_truth_boxes:
                gt_box = gt['box']
                iou = calculate_iou((x1, y1, x2, y2), gt_box)

                iou_file.write(f"{img_name}, Detected Box {i}: {(x1, y1, x2, y2)}, Ground Truth: {gt_box}, IoU: {iou:.4f}\n")
                print(f"{img_name}, Detected Box {i}: {(x1, y1, x2, y2)}, Ground Truth: {gt_box}, IoU: {iou:.4f}")

