import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

# YOLOv8 modelinin yolunu belirtin
MODEL_PATH = 'runs/detect/train/weights/best.pt'

# İşlenecek görüntünün yolunu belirtin
IMAGE_PATH = 'dataset/train/images/lionfish1_10_png.rf.66d8bd349cafd737a7951348b5177574.jpg'

# Çıktı olarak kaydedilecek görüntü adı
OUTPUT_IMAGE_PATH = 'output_result.jpg'

def main():
    # Modeli yükle
    model = YOLO(MODEL_PATH)

    # Görsel var mı kontrol et
    if not os.path.exists(IMAGE_PATH):
        print(f"Görsel bulunamadı: {IMAGE_PATH}")
        return

    # Görseli okuyup NumPy array formatına çevir
    image = Image.open(IMAGE_PATH)
    image_np = np.array(image)

    # YOLO tahmini yap
    results = model(image_np)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box koordinatları
    num_fish = len(boxes)  # Tespit edilen aslan balığı sayısı

    # Bounding box çizimi
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # Koordinatları tamsayıya çevir
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil bounding box
        cv2.putText(
            image_np,
            "Aslan Balığı",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Konsola bilgi yazdır
    print(f"Tespit edilen aslan balığı sayısı: {num_fish}")
    if num_fish > 3:
        print("Uyarı: Çok fazla aslan balığı tespit edildi!")

    # Çıktı görselini diske kaydet
    # OpenCV ile kaydederken BGR formatı kullanılır, o yüzden dönüştürme yapıyoruz
    cv2.imwrite(OUTPUT_IMAGE_PATH, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    print(f"Çıktı resmi kaydedildi: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()
