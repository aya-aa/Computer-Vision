import cv2
import numpy as np
import easyocr
import time
import threading

ocr_engine = None


def get_ocr_engine(use_gpu=True):
    global ocr_engine
    if ocr_engine is None:
        try:
            ocr_engine = easyocr.Reader(['en', 'ar'], gpu=use_gpu)
        except Exception as e:
            raise 
    return ocr_engine


def process_image_with_ocr(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {'text': 'Failed to load image', 'lines': []}

    target_height = 800
    h, w = img.shape[:2]
    if h > target_height:
        new_w = int(w * (target_height / h))
        img_resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_contrast = cv2.convertScaleAbs(gray, alpha=1.1, beta=0)

    reader = get_ocr_engine(use_gpu=True)
    results = reader.readtext(
        gray_contrast,
        text_threshold=0.4,
        low_text=0.3,
        link_threshold=0.3,
        detail=1,
        paragraph=False
    )

    all_lines = [text for (_, text, confidence) in results if confidence > 0.32]


    combined_text = " | ".join(all_lines)

    return {
        'text': combined_text
    }
