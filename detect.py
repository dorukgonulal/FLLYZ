import cv2
from ultralytics import YOLO

# YOLOv8 modelini yükle (fine-tune ettiğin modele göre .pt dosyanı belirt)
model = YOLO('fine_tuned_model.pt')

# Tespit edilecek resmi oku
image_path = 'resim.jpg'  # Resim dosyanın yolu
image = cv2.imread(image_path)

# YOLO tahminini yap
results = model(image)

# Tüm tespitleri al
detections = results[0].boxes

# Sonuçlar üzerinde döngü kurarak bounding box çiz
for box in detections:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box koordinatları
    conf = box.conf[0]  # Güven skoru
    cls = int(box.cls[0])  # Sınıf etiketi

    # Bounding box çiz
    label = f'Class: {cls}, Conf: {conf:.2f}'
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil renk kutu
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Sonuç görüntüsünü göster ve kaydet
cv2.imshow('YOLOv8 Detection', image)
cv2.imwrite('sonuc.jpg', image)  # Tespit edilen resmi kaydet
cv2.waitKey(0)
cv2.destroyAllWindows()
