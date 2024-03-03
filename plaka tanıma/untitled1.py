import os
import cv2
import numpy as np
import pytesseract

def plaka_tanima():

    tesseract_path = r'C:\Program Files\Tesseract-OCR'
    os.environ['PATH'] += os.pathsep + tesseract_path

    cap = cv2.VideoCapture(0)
    cap.set(3, 960)
    cap.set(4, 480)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Kamera bağlantısı başarısız.")
            break
        
        plakalar = tesseract_plaka_tanima(frame)
        
        # Bulunan plakaları görsel olarak işaretle
        for plaka in plakalar:
            x, y, w, h = plaka
            plaka_img = frame[y:y+h, x:x+w]
            plaka_text = pytesseract.image_to_string(plaka_img, config='--oem 3 --psm 6')
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, plaka_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Plaka Tanıma', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def tesseract_plaka_tanima(image):
    # Görüntüyü gri tonlamaya dönüştür
    # Kenarları tespit etmek için Canny kenar dedektörünü kullan
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Kenarlar arasında konturları bul
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plaka_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Kontur alanına göre filtreleme yap
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            plaka_contours.append((x, y, w, h))
    
    return plaka_contours

plaka_tanima()
