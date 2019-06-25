import numpy as np
import cv2

yuzu_algila = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gozu_algila = cv2.CascadeClassifier('haarcascade_eye.xml')
gulumseme_algila = cv2.CascadeClassifier('haarcascade_smile.xml')

kamera = cv2.VideoCapture(0) #0 nolu kamerayý kullan

while 1:
    ret, goruntu = kamera.read() #kameradaki görüntüyü al 
    griKatmanli = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY) #görüntüye gri katman ekle
    yuzler = yuzu_algila.detectMultiScale(griKatmanli, 1.3, 5) #yüzleri bul

    for (x,y,w,h) in yuzler:
        cv2.rectangle(goruntu,(x,y),(x+w,y+h),(255,0,0),2) #dikdörtgen çizimi (yüz)
        roi_gray = griKatmanli[y:y+h, x:x+w]
        roi_color = goruntu[y:y+h, x:x+w]
        
        gozler = gozu_algila.detectMultiScale(roi_gray) #gozleri bul
        for (gx,gy,gw,gh) in gozler:
            cv2.rectangle(roi_color,(gx,gy),(gx+gw,gy+gh),(0,255,0),2) #dikdörtgen çizimi (gözler)
        gulumseme = gulumseme_algila.detectMultiScale(roi_gray)
        for (sx,sy,sw,sh) in gulumseme:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2) #dikdörtgen çizimi (gulumseme)

    cv2.imshow('Duygu Analizi',goruntu) #videoda görüntüleme
    
    if cv2.waitKey(30) & 0xff == ord('q'): #videoyu kapatmak için "q" bas.
        break

kamera.release()
cv2.destroyAllWindows()
