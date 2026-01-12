import cv2
import numpy as np

image = cv2.imread("araba.jpeg")

# Resim yüklenemezse hata verip durdurq
if image is None:
    print("Hata: Resim bulunamadı veya yol yanlış!")
    exit()

# Çok büyükse biraz küçültelim (İsteğe bağlı, ekrana sığsın diye)
image = cv2.resize(image, (600, 600))

# 2. Ön Hazırlık: LAB Dönüşümü ve Float çevirimi
# Bunu döngü dışında bir kere yapıyoruz ki bilgisayarı yormayalım
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
lab_image = lab_image.astype("float32")


# Tıklama olayını yönetecek fonksiyon
def tiklama_olayi(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Sol Tıklandığında

        # A. Tıklanan yerdeki rengi "Referans" al
        target_pixel = lab_image[y, x]

        # B. Delta E Hesapla (Tüm resim için)
        diff = lab_image - target_pixel
        delta_e = np.sqrt(np.sum(diff ** 2, axis=2))

        # C. Eşikleme (Gruplama)
        threshold = 30  # Bu değeri artırırsanız daha geniş bir renk aralığını alır
        mask = delta_e < threshold
        mask_uint8 = (mask * 255).astype("uint8")

        # D. Sonucu Oluştur
        result = cv2.bitwise_and(image, image, mask=mask_uint8)

        # E. Sonuçları Göster (Yan yana)
        # Orijinal ve Sonucu birleştirip gösterelim
        combined = np.hstack((image, result))
        cv2.imshow("Sol: Tiklayin | Sag: Sonuc", combined)


# 3. Pencere Ayarları
cv2.namedWindow("Sol: Tiklayin | Sag: Sonuc")
cv2.setMouseCallback("Sol: Tiklayin | Sag: Sonuc", tiklama_olayi)

print("Program çalıştı. Lütfen resim üzerinde bir noktaya TIKLAYIN.")
print("Çıkmak için 'q' tuşuna basın.")

# İlk açılışta sadece orijinali gösterelim (sağ taraf siyah başlasın)
initial_display = np.hstack((image, np.zeros_like(image)))
cv2.imshow("Sol: Tiklayin | Sag: Sonuc", initial_display)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # q tuşuna basınca çık
        break

cv2.destroyAllWindows()