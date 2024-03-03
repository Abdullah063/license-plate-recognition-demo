import cv2
import os
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def plaka_tanima():
    cap = cv2.VideoCapture(0)
    
    if not os.path.exists('plaka_goruntuleri'):
        os.makedirs('plaka_goruntuleri')
    
    tanima_baslangic_zamani = time.time()
    tanima_suresi = 10  # Saniye cinsinden
    plakalar = []
    
    while True:
        ret, frame = cap.read() 
            
        if not ret:
            print("Kamera bağlantısı başarısız.")
            break
        
        if time.time() - tanima_baslangic_zamani < tanima_suresi:
            plakalar.extend(tesseract_plaka_tanima(frame))
        else:
            for i, plaka in enumerate(plakalar):
                x, y, w, h = plaka
                plaka_img = frame[y:y+h, x:x+w]
                cv2.imwrite(f'plaka_goruntuleri/plaka_{i}.jpg', plaka_img)
            
            plakalar = []
            tanima_baslangic_zamani = time.time()
            
        cv2.imshow('Plaka Tanıma', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def tesseract_plaka_tanima(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blur, 50, 150)
    
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plaka_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / float(h)
            
            if aspect_ratio > 2.5 and aspect_ratio < 5.0:
                plaka_contours.append((x, y, w, h))
    
    return plaka_contours

plaka_tanima()
