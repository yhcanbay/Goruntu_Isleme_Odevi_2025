"""
Segmentasyon Modülü - HSV renk uzayında renk tabanlı segmentasyon
"""

import cv2
import numpy as np


def renk_segmentasyonu(goruntu: np.ndarray, renk: str) -> np.ndarray:
    """HSV renk uzayında belirli bir rengi segmente eder."""
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    
    renk_araliklari = {
        'kirmizi': [((0, 100, 100), (10, 255, 255)), ((160, 100, 100), (180, 255, 255))],
        'yesil': [((35, 50, 50), (85, 255, 255))],
        'mavi': [((100, 100, 100), (130, 255, 255))],
    }
    
    if renk not in renk_araliklari:
        raise ValueError(f"Bilinmeyen renk: {renk}")
    
    maske = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for alt, ust in renk_araliklari[renk]:
        kısmi_maske = cv2.inRange(hsv, np.array(alt), np.array(ust))
        maske = cv2.bitwise_or(maske, kısmi_maske)
    
    return maske


def maskeyi_uygula(goruntu: np.ndarray, maske: np.ndarray) -> np.ndarray:
    """Maskeyi orijinal görüntüye uygular."""
    return cv2.bitwise_and(goruntu, goruntu, mask=maske)
