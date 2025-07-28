from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
from apis.ocrpy import process_image_with_ocr


import os
from django.conf import settings
from io import BytesIO

#model loading
MODEL_PATH = os.path.join(settings.BASE_DIR, "apis","models", "best (9).pt")
model = YOLO(MODEL_PATH)

#model's confidence
IMG_CONF=0.6
VID_CONF=0.8


def pil_image_to_bytes(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format='JPEG')
    return buf.getvalue()


@csrf_exempt
#image detection api
def imageapi(request):
    if request.method == "POST":
        if 'images' not in request.FILES:
            return JsonResponse({"error": "No image was uploaded"}, status=400)

        images = request.FILES.getlist('images')
        results = {}


        for img_file in images:
            try:
                # Converting uploaded file to PIL image ( mode rgb)
                img = Image.open(img_file).convert("RGB")
                detections = model(img, conf=IMG_CONF)
                img_bytes = pil_image_to_bytes(img)
                detected_text=process_image_with_ocr(img_bytes)


                detections_data = []
                for box in detections[0].boxes.data.tolist():
                    #coordinats, confidence and class of detections 
                    x1, y1, x2, y2, conf, cls = box
                    detections_data.append({
                        "class": model.names[int(cls)],
                        "confidence": float(conf),
                        
                        "box": [
                            round(x1,2), 
                            round(y1,2), 
                            round(x2,2), 
                            round(y2,2),],
                        "text":detected_text,
                    })

                results[img_file.name] = detections_data

            except Exception as e:
                results[img_file.name] = {"error": str(e)}

        return JsonResponse({"results": results}, status=200, json_dumps_params={'ensure_ascii': False})


    return JsonResponse({"error": "Only POST method is allowed."}, status=405)


@csrf_exempt
def videoapi(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)

    video_file = request.FILES.get('video')
    if not video_file:
        return JsonResponse({'error': 'No video uploaded'}, status=400)

    #Save to a temp file ( mp4)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        for chunk in video_file.chunks():
            tmp.write(chunk)
        video_path = tmp.name

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return JsonResponse({'error': 'Failed to open video'}, status=500)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    active_logos = {}
    intervals = []
    logo_confs = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=VID_CONF, verbose=False)
        current_logos = set()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            current_logos.add(label)
            conf = round(float(box.conf[0]), 2)

            if label in logo_confs:
                logo_confs[label].append(conf)
            else:
                logo_confs[label] = [conf]

            if label in active_logos:
                active_logos[label][1] = frame_idx
            else:
                active_logos[label] = [frame_idx, frame_idx]
                
        for label in list(active_logos):
            if label not in current_logos:
                start, end = active_logos[label]
                confs = logo_confs.get(label, [])
                avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
                intervals.append({
                    "logo": label,
                    "confidence": avg_conf,
                    "start": round(start / fps, 2),
                    "end": round(end / fps, 2)
                })
                del active_logos[label]
                del logo_confs[label] 

        frame_idx += 1

    for label, (start, end) in active_logos.items():
        confs = logo_confs.get(label, [])
        avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
        intervals.append({
            "logo": label,
            "confidence": avg_conf,
            "start": round(start / fps, 2),
            "end": round(end / fps, 2)
        })


    cap.release()
    os.remove(video_path)
    return JsonResponse(intervals, safe=False)
