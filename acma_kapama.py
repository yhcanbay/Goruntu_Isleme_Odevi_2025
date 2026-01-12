import cv2
import numpy as np


image=cv2.imread("lena2.png")

# Resim okunamazsa hata vermemesi için kontrol
if image is None:
    print("HATA: Resim bulunamadı! Dosya yolunu kontrol edin.")
else:
    # 2. ADIM: Maske Oluşturma (Bölütleme)
    # Morfolojik işlem yapmak için önce elimizde bir 'maske' olmalı.
    # Örnek olarak resmi Griye çevirip basit bir eşikleme (threshold) yapalım:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 127'den parlak olan yerleri beyaz (255), karanlık yerleri siyah (0) yap.
    # Bu işlem bize 'mask' değişkenini verir.
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # NOT: Eğer renkli proje yapıyorsanız burası cv2.inRange(...) olacak.

    # 3. ADIM: Morfolojik İşlemler (İyileştirme)
    # Kernel boyutu (5,5). Daha büyük gürültüler için (7,7) veya (9,9) yapabilirsiniz.
    kernel = np.ones((5, 5), np.uint8)

    # AÇMA (Opening): Arka plandaki beyaz noktacıkları (gürültü) temizler
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # KAPAMA (Closing): Nesne içindeki siyah delikleri kapatır
    mask_final = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    # 4. ADIM: Sonuçları Göster
    # Orijinal resmi küçültelim ki ekrana sığsın (isteğe bağlı)
    img_resized = cv2.resize(image, (400, 400))
    mask_resized = cv2.resize(mask, (400, 400))
    final_resized = cv2.resize(mask_final, (400, 400))

    cv2.imshow('1. Orijinal Resim', img_resized)
    cv2.imshow('2. Ilk Maske (Gurultulu Olabilir)', mask_resized)
    cv2.imshow('3. Iyilestirilmis Maske (Acma+Kapama)', final_resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()