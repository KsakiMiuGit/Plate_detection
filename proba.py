import yolov5
import kagglehub
import os
import random
import cv2
import pytesseract

# load model
model = yolov5.load('keremberke/yolov5n-license-plate')
path = kagglehub.dataset_download("piotrstefaskiue/poland-vehicle-license-plate-dataset")
images_path = os.path.join(path, "photos")

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# wybierz losowe zdjęcia z path (maksymalnie 5)
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
num_images_to_sample = min(5, len(image_files))  # ensure we don't sample more than available
random_images = random.sample(image_files, num_images_to_sample)

# ensure the results directory exists
os.makedirs('results', exist_ok=True)
os.makedirs('results/plates', exist_ok=True)  # directory for cropped plates

# initialize OCR (optional: specify tesseract executable path if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r'H:\Python\tess\tesseract.exe'

# create a file to save plate numbers
with open('results/plate_numbers.txt', 'w') as plate_file:
    for img_name in random_images:
        img_path = os.path.join(images_path, img_name)
        
        # perform inference
        results = model(img_path, size=640)

        # inference with test time augmentation
        results = model(img_path, augment=True)

        # parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        # read the image for OCR
        img = cv2.imread(img_path)

        # iterate over detected boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            plate_region = img[y1:y2, x1:x2]  # crop the plate region

            # preprocess the plate region
            gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # apply thresholding

            # Przycinanie marginesów (10% z lewej strony i 5% z pozostałych stron)
            height, width = thresh_plate.shape
            left_margin = int(width * 0.10)  # 10% margines z lewej strony
            other_margin = int(min(height, width) * 0.05)  # 5% margines z pozostałych stron
            thresh_plate = thresh_plate[other_margin:height - other_margin, left_margin:width - other_margin]

            # Usuwanie małych szumów za pomocą operacji morfologicznych
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh_plate = cv2.morphologyEx(thresh_plate, cv2.MORPH_OPEN, kernel)

            # save the cropped plate image
            plate_filename = f"results/plates/{os.path.splitext(img_name)[0]}_plate_{i}.png"
            cv2.imwrite(plate_filename, thresh_plate)

            # apply OCR to extract text
            plate_text = pytesseract.image_to_string(thresh_plate, config='--psm 7')  # psm 7 assumes a single line of text

            # clean up the OCR result
            plate_text = plate_text.strip()
            if plate_text:
                # save plate number to file
                plate_file.write(f"{img_name}: {plate_text}\n")
                print(f"Detected plate: {plate_text} in {img_name}")

