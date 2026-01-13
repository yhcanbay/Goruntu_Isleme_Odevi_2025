"""
● Amaç: RGB, HSV, LAB gibi renk uzayları arasında dönüşüm yapmak ve renk
tabanlı nesne segmentasyonu.


● Görevler:
1. Bir renkli görüntüyü RGB'den HSV ve LAB renk uzaylarına dönüştürün.


UCBES
2.  Belirlibir renge sahip (örneğin, kırmızı bir araba, yeşil yapraklar)
nesneleri bölütlemek (segmentasyon) için HSV uzayında eşikleme
(thresholding) yapın. 



3. LAB uzayındaki renk farkı metriğini (Delta E) kullanarak benzer renkleri
gruplayın.



4. Morfolojik işlemlerle (açma/kapama) bölütleme sonucunu iyileştirin.

● Kitap İlgili Bölüm: 6 (Color Image Processing)
● Kullanılacak Kütüphaneler: OpenCV, NumPy
"""



import numpy as np
import cv2

img = cv2.imread("araba.jpeg")    

# 1 - ilk kısım yapıldı
lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB) 
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV) 


cv2.imshow("1",img)
cv2.imshow("2",hsv)

cv2.waitKey(0)




print("\n=== OTOMATİK RENK SEGMENTASYONU (OTSU) ===")

# S (Saturation) kanalını al - Renklilik bilgisi buradadır
# Gri/beyaz/siyah alanların doygunluğu düşük, renkli nesnelerin yüksektir
s_channel = hsv[:, :, 1]

# Otsu eşikleme ile otomatik threshold belirle ve maske oluştur
# cv2.THRESH_BINARY + cv2.THRESH_OTSU kullanarak en uygun eşik değerini bulur
ret, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Otsu Metodu ile hesaplanan eşik değeri: {ret}")
print("Segmentasyon tamamlandı.")

# Maskeyi kullanarak orijinal görüntüyü bölütle (Maskeleme)
segmented_img = cv2.bitwise_and(img, img, mask=mask)

# Sonuçları göster
cv2.imshow("S Kanali (Doygunluk)", s_channel)
cv2.imshow("Otsu Maskesi", mask)
cv2.imshow("Otomatik Segmentasyon Sonucu", segmented_img)

print("Sonuçlar gösteriliyor. Programı kapatmak için herhangi bir tuşa basın...")
cv2.waitKey(0)
cv2.destroyAllWindows()
