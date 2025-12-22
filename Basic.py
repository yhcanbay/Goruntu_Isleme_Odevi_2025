"""
● Amaç: RGB, HSV, LAB gibi renk uzayları arasında dönüşüm yapmak ve renk
tabanlı nesne segmentasyonu.


● Görevler:
1. Bir renkli görüntüyü RGB'den HSV ve LAB renk uzaylarına dönüştürün.



2. Belirli bir renge sahip (örneğin, kırmızı bir araba, yeşil yapraklar)
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

img = cv2.imread("blasp.jpg")


# lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB) : RGB -> LAB formülü
# hsi = cv2.cvtColor(img,cv2.COLOR_RGB2HSV) : RGB -> HSI formülü
 
# LAB dönüşümü için 1. işlem
rgb = img / 255

# LAB dönüşümü için 2. işlem
np_lin = np.where(
    rgb <= 0.04045,
    rgb / 12.92, # yukarıdaki ifade doğru olursa burası calisir
    ((rgb + 0.055) / 1.055) ** 2.4 # ilk ifade yanlis olursa burasi calisir
)

cv2.imshow("bes",np_lin)
cv2.waitKey(0)
# LAB dönüşümü için 3. işlem




# LAB dönüşümü için 4. işlem