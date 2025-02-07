import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# YOLOv8 modelini yükle
model = YOLO('runs/detect/train/weights/best.pt')

# Görselin bulunduğu path (otomatik olarak bu path'teki görsel işlenecek)
image_path = 'dataset/train/images/lionfish1_10_png.rf.66d8bd349cafd737a7951348b5177574.jpg'  # Buraya kendi görsel yolunu yaz

# Başlık
st.title("Aslan Balığı Tespiti Uygulaması")

# Görseli yükle ve numpy array'e çevir
if os.path.exists(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)

    # YOLO tahmini yap
    results = model(image_np)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box koordinatları
    num_fish = len(boxes)  # Tespit edilen aslan balığı sayısı

    # Bounding box çizimi
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # Koordinatları tamsayıya çevir
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil bounding box
        cv2.putText(image_np, "Aslan Balığı", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tespit edilen aslan balığı sayısını göster
    st.write(f"Tespit edilen aslan balığı sayısı: {num_fish}")

    # Uyarı verme
    if num_fish > 3:
        st.error("Uyarı: Çok fazla aslan balığı tespit edildi!")

    # Görseli göster
    st.image(image_np, caption="Tespit Sonucu", use_container_width=True)
else:
    st.error(f"Görsel bulunamadı: {image_path}")
