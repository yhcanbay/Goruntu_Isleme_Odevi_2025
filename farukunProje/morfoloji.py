"""
Morfoloji Modülü - Morfolojik görüntü işleme operasyonları
"""

import cv2
import numpy as np


def acma(maske: np.ndarray, kernel_boyut: int = 5) -> np.ndarray:
    """Açma (Opening) işlemi: Küçük gürültüleri temizler."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_boyut, kernel_boyut))
    return cv2.morphologyEx(maske, cv2.MORPH_OPEN, kernel)


def kapama(maske: np.ndarray, kernel_boyut: int = 5) -> np.ndarray:
    """Kapama (Closing) işlemi: Küçük delikleri doldurur."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_boyut, kernel_boyut))
    return cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel)


def tam_iyilestirme(maske: np.ndarray, acma_boyut: int = 3, kapama_boyut: int = 7) -> np.ndarray:
    """Tam iyileştirme: Açma + Kapama kombinasyonu."""
    maske_acma = acma(maske, acma_boyut)
    return kapama(maske_acma, kapama_boyut)
