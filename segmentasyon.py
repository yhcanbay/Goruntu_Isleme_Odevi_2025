"""
HSV Tabanlı Renk Segmentasyonu
==============================
Belirli renkteki nesneleri görüntüden ayırır.
"""

import cv2
import numpy as np


# Yaygın renklerin HSV aralıkları (OpenCV: H:0-180, S:0-255, V:0-255)
RENK_ARALIKLARI = {
    'kirmizi': [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255]))
    ],
    'turuncu': [(np.array([10, 100, 100]), np.array([25, 255, 255]))],
    'sari': [(np.array([25, 100, 100]), np.array([35, 255, 255]))],
    'yesil': [(np.array([35, 100, 100]), np.array([85, 255, 255]))],
    'cyan': [(np.array([85, 100, 100]), np.array([100, 255, 255]))],
    'mavi': [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
    'mor': [(np.array([130, 100, 100]), np.array([160, 255, 255]))],
}


def renk_segmentasyonu(goruntu: np.ndarray, renk: str) -> np.ndarray:
    """
    HSV uzayında belirli bir rengi segmente eder.
    
    Args:
        goruntu: BGR formatında görüntü
        renk: 'kirmizi', 'yesil', 'mavi' vb.
    
    Returns:
        np.ndarray: Binary maske (0 veya 255)
    """
    if renk not in RENK_ARALIKLARI:
        raise ValueError(f"Bilinmeyen renk: {renk}. Seçenekler: {list(RENK_ARALIKLARI.keys())}")
    
    # BGR → HSV
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    
    # Her aralık için maske oluştur ve birleştir
    maske = np.zeros(goruntu.shape[:2], dtype=np.uint8)
    for alt, ust in RENK_ARALIKLARI[renk]:
        maske_parcasi = cv2.inRange(hsv, alt, ust)
        maske = cv2.bitwise_or(maske, maske_parcasi)
    
    return maske


def ozel_aralik_segmentasyonu(goruntu: np.ndarray, 
                               h_min: int, h_max: int,
                               s_min: int = 100, s_max: int = 255,
                               v_min: int = 100, v_max: int = 255) -> np.ndarray:
    """
    Özel HSV aralığı ile segmentasyon.
    
    Args:
        goruntu: BGR formatında görüntü
        h_min, h_max: Hue aralığı (0-180)
        s_min, s_max: Saturation aralığı (0-255)
        v_min, v_max: Value aralığı (0-255)
    
    Returns:
        np.ndarray: Binary maske
    """
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    alt = np.array([h_min, s_min, v_min])
    ust = np.array([h_max, s_max, v_max])
    return cv2.inRange(hsv, alt, ust)


def maskeyi_uygula(goruntu: np.ndarray, maske: np.ndarray) -> np.ndarray:
    """Maskeyi görüntüye uygula, sadece seçili bölgeyi göster."""
    return cv2.bitwise_and(goruntu, goruntu, mask=maske)


if __name__ == "__main__":
    print("HSV Renk Segmentasyonu Modülü")
    print(f"Desteklenen renkler: {list(RENK_ARALIKLARI.keys())}")
