"""
Morfolojik İşlemler
===================
Segmentasyon sonuçlarını iyileştirmek için açma/kapama işlemleri.
"""

import cv2
import numpy as np


def kernel_olustur(boyut: int = 5, sekil: str = 'kare') -> np.ndarray:
    """
    Morfolojik işlemler için yapısal element oluşturur.
    
    Args:
        boyut: Kernel boyutu (tek sayı olmalı)
        sekil: 'kare', 'daire', veya 'capraz'
    
    Returns:
        np.ndarray: Yapısal element
    """
    if sekil == 'kare':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (boyut, boyut))
    elif sekil == 'daire':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boyut, boyut))
    elif sekil == 'capraz':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (boyut, boyut))
    else:
        raise ValueError(f"Bilinmeyen şekil: {sekil}")


def acma(maske: np.ndarray, kernel_boyut: int = 5) -> np.ndarray:
    """
    Açma (Opening) işlemi: Erozyon + Genişleme
    Küçük gürültüleri temizler.
    
    Args:
        maske: Binary maske
        kernel_boyut: Yapısal element boyutu
    
    Returns:
        np.ndarray: İşlenmiş maske
    """
    kernel = kernel_olustur(kernel_boyut, 'kare')
    return cv2.morphologyEx(maske, cv2.MORPH_OPEN, kernel)


def kapama(maske: np.ndarray, kernel_boyut: int = 5) -> np.ndarray:
    """
    Kapama (Closing) işlemi: Genişleme + Erozyon
    Küçük delikleri doldurur.
    
    Args:
        maske: Binary maske
        kernel_boyut: Yapısal element boyutu
    
    Returns:
        np.ndarray: İşlenmiş maske
    """
    kernel = kernel_olustur(kernel_boyut, 'kare')
    return cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel)


def erozyon(maske: np.ndarray, kernel_boyut: int = 3) -> np.ndarray:
    """Erozyon: Nesneyi küçültür, kenarları aşındırır."""
    kernel = kernel_olustur(kernel_boyut, 'kare')
    return cv2.erode(maske, kernel, iterations=1)


def genisleme(maske: np.ndarray, kernel_boyut: int = 3) -> np.ndarray:
    """Genişleme (Dilation): Nesneyi büyütür."""
    kernel = kernel_olustur(kernel_boyut, 'kare')
    return cv2.dilate(maske, kernel, iterations=1)


def tam_iyilestirme(maske: np.ndarray, 
                    acma_boyut: int = 3, 
                    kapama_boyut: int = 5) -> np.ndarray:
    """
    Tam iyileştirme: Önce açma (gürültü temizle), sonra kapama (delikleri doldur).
    
    Args:
        maske: Binary maske
        acma_boyut: Açma kernel boyutu
        kapama_boyut: Kapama kernel boyutu
    
    Returns:
        np.ndarray: İyileştirilmiş maske
    """
    # Önce açma ile küçük gürültüleri temizle
    sonuc = acma(maske, acma_boyut)
    # Sonra kapama ile delikleri doldur
    sonuc = kapama(sonuc, kapama_boyut)
    return sonuc


if __name__ == "__main__":
    print("Morfolojik İşlemler Modülü")
    print("Fonksiyonlar: acma(), kapama(), erozyon(), genisleme(), tam_iyilestirme()")
