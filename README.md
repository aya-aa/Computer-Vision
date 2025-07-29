
# YOLO Detection API with OCR (Django)

This Django-based API provides endpoints for detecting objects/logos in images and videos using a YOLOv11 model, along with OCR-based text extraction from detected regions in images.


##  Features

- **YOLOv11 Integration** for logo/object detection.
- **OCR support** for text extraction from detected areas in images.
- Accepts **multiple image uploads**.
- Supports **video upload** and returns logo appearance intervals with average confidence.

## Requirements

Make sure to install the dependencies listed in the requirements.txt
pip install -r requirements.txt




## Project Setup

1. **Model File**:  
   Place your YOLOv11 `.pt` model in the path:  
   ```
   <project_root>/apis/models/best (9).pt
   ```

2. **Update `settings.py`**:
   Ensure `BASE_DIR` is set correctly, and `DEBUG = True` (during development).



##  API Endpoints

###  `POST /image-upload/`


#### Example using `curl`:

```bash 
curl.exe -X POST http://127.0.0.1:8000/api/image-upload/ -F "images=@C:\Users\user\Downloads\images.png"   
```

#### Response:
```json
{"results": {"la.jpg": [{"class": "Vitalait", "confidence": 0.8971766829490662, "box": [578.49, 821.45, 705.35, 931.48], "text": {"text": "القلب عالقلب | /319 | 210"}}]}}
```

---

### `POST /videoapi/`

Detects logos in uploaded videos and returns the time intervals they appear in.

- **Request Type**: `multipart/form-data`
- **Field**: `video`

#### ✅ Example using `curl`:

``` bash

curl.exe -X POST http://127.0.0.1:8000/api/video-upload/ -F "video=@C:\\Users\\user\\Downloads\\story7.mp4"

```

#### 🔁 Response:
```json
[
  {
    "logo": "Adidas",
    "confidence": 0.78,
    "start": 2.15,
    "end": 6.34
  },
  {
    "logo": "Nike",
    "confidence": 0.84,
    "start": 10.0,
    "end": 15.7
  }
]
```

---

## Testing

Run your Django server:

```bash
python manage.py runserver
```

Then test the endpoints using Postman, curl, or your preferred HTTP client.

---

## Project Structure (Simplified)

```
project/
├── apis/
│   ├── __init__.py
│   ├── ocrpy.py
│   ├── views.py
│   └── models/
│       └── best (9).pt
├── project/
│   └── settings.py
└── manage.py
```
