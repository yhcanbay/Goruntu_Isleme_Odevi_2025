"""
HSV Tabanlı Renk Segmentasyonu
==============================
Belirli renkteki nesneleri görüntüden ayırır.

Bu modül HSV renk uzayında eşikleme yaparak belirli renkleri tespit eder.
HSV kullanmanın avantajı: Aydınlatma değişimlerine karşı daha dayanıklı
"""

import cv2  # OpenCV - Görüntü işleme kütüphanesi
import numpy as np  # NumPy - Sayısal işlemler için


# Yaygın renklerin HSV aralıkları (OpenCV: H:0-180, S:0-255, V:0-255)
# Her renk için alt ve üst sınır değerleri tanımlanmış
# OpenCV'de Hue değeri 0-180 arasında (normal 0-360'ın yarısı)
RENK_ARALIKLARI = {
    'kirmizi': [
        # Kırmızı renk HSV'de iki aralıkta (0-10 ve 160-180 arası)
        # Çünkü kırmızı renk tonu skalasının başında ve sonunda
        (np.array([0, 100, 100]), np.array([10, 255, 255])),      # Alt kırmızı (0-10)
        (np.array([160, 100, 100]), np.array([180, 255, 255]))    # Üst kırmızı (160-180)
    ],
    'turuncu': [(np.array([10, 100, 100]), np.array([25, 255, 255]))],   # 10-25 arası
    'sari': [(np.array([25, 100, 100]), np.array([35, 255, 255]))],      # 25-35 arası
    'yesil': [(np.array([35, 100, 100]), np.array([85, 255, 255]))],     # 35-85 arası (geniş yeşil aralığı)
    'cyan': [(np.array([85, 100, 100]), np.array([100, 255, 255]))],     # 85-100 arası
    'mavi': [(np.array([100, 100, 100]), np.array([130, 255, 255]))],    # 100-130 arası
    'mor': [(np.array([130, 100, 100]), np.array([160, 255, 255]))],     # 130-160 arası
}


def renk_segmentasyonu(goruntu: np.ndarray, renk: str) -> np.ndarray:
    """
    HSV uzayında belirli bir rengi segmente eder.
    
    Çalışma prensibi:
    1. Görüntüyü BGR'den HSV'ye çevir
    2. Tanımlı HSV aralıklarında maskeleme yap
    3. İkili (binary) maske döndür (0: siyah, 255: beyaz)
    
    Args:
        goruntu: BGR formatında görüntü (OpenCV formatı)
        renk: 'kirmizi', 'yesil', 'mavi' vb. (RENK_ARALIKLARI'ndan biri)
    
    Returns:
        np.ndarray: Binary maske (0 veya 255 değerlerinde, aynı boyutta tek kanallı)
    """
    # Rengin tanımlı olup olmadığını kontrol et
    if renk not in RENK_ARALIKLARI:
        raise ValueError(f"Bilinmeyen renk: {renk}. Seçenekler: {list(RENK_ARALIKLARI.keys())}")
    
    # BGR → HSV dönüşümü (HSV'de renk segmentasyonu daha kolay)
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    
    # Her aralık için maske oluştur ve birleştir
    # Boş bir maske ile başla (tüm pikseller siyah: 0)
    maske = np.zeros(goruntu.shape[:2], dtype=np.uint8)  # shape[:2] = (yükseklik, genişlik)
    
    # Seçilen renk için tanımlı tüm aralıkları döngü ile işle
    for alt, ust in RENK_ARALIKLARI[renk]:
        # cv2.inRange: Belirtilen aralıktaki pikselleri beyaz (255) yap
        maske_parcasi = cv2.inRange(hsv, alt, ust)
        # bitwise_or: İki maskeyi birleştir (VEYA işlemi, herhangi biri beyazsa beyaz)
        maske = cv2.bitwise_or(maske, maske_parcasi)
    
    return maske


def ozel_aralik_segmentasyonu(goruntu: np.ndarray, 
                               h_min: int, h_max: int,
                               s_min: int = 100, s_max: int = 255,
                               v_min: int = 100, v_max: int = 255) -> np.ndarray:
    """
    Özel HSV aralığı ile segmentasyon.
    
    Bu fonksiyon, önceden tanımlanmamış renkleri segmente etmek için kullanılır.
    Kullanıcı kendi HSV değerlerini belirleyerek segmentasyon yapabilir.
    
    Args:
        goruntu: BGR formatında görüntü
        h_min, h_max: Hue (Renk Tonu) aralığı (0-180) - OpenCV'de yarı açı kullanılır
        s_min, s_max: Saturation (Doygunluk) aralığı (0-255) - Varsayılan: 100-255
        v_min, v_max: Value (Parlaklık) aralığı (0-255) - Varsayılan: 100-255
    
    Returns:
        np.ndarray: Binary maske (0 veya 255)
    """
    # BGR'den HSV'ye dönüştür
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    
    # Alt ve üst sınırları NumPy dizisi olarak oluştur
    alt = np.array([h_min, s_min, v_min])
    ust = np.array([h_max, s_max, v_max])
    
    # Belirtilen aralıktaki pikselleri maskeye al
    return cv2.inRange(hsv, alt, ust)


def maskeyi_uygula(goruntu: np.ndarray, maske: np.ndarray) -> np.ndarray:
    """
    Maskeyi görüntüye uygula, sadece seçili bölgeyi göster.
    
    Çalışma prensibi:
    - Maske beyaz (255) olan yerlerde: Orijinal piksel değerini göster
    - Maske siyah (0) olan yerlerde: Siyah (0, 0, 0) göster
    
    Args:
        goruntu: BGR formatında orijinal görüntü
        maske: Binary maske (0 veya 255, tek kanallı)
    
    Returns:
        np.ndarray: Maskelenmiş görüntü (sadece istenen bölge görünür)
    """
    # bitwise_and: İki görüntünün bit düzeyinde VE işlemi
    # mask parametresi: Hangi piksellerin işleneceğini belirler
    # Sonuç: Maskenin beyaz olduğu yerlerde orijinal görüntü, siyah yerlerde siyah
    return cv2.bitwise_and(goruntu, goruntu, mask=maske)


# Bu modül direkt çalıştırılırsa (python segmentasyon.py)
if __name__ == "__main__":
    print("HSV Renk Segmentasyonu Modülü")
    print(f"Desteklenen renkler: {list(RENK_ARALIKLARI.keys())}")

