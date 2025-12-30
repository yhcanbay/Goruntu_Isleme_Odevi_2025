"""
Delta E ile Renk Gruplama
=========================
LAB uzayında benzer renkleri gruplar.
"""

import cv2
import numpy as np
from collections import defaultdict


def delta_e_cie76(lab1, lab2):
    """İki LAB rengi arasındaki Delta E mesafesi."""
    return np.sqrt(np.sum((np.array(lab1) - np.array(lab2)) ** 2))


def dominant_renkler_bul(goruntu: np.ndarray, k: int = 5) -> np.ndarray:
    """
    K-Means ile görüntüdeki baskın renkleri bulur.
    
    Args:
        goruntu: BGR formatında görüntü
        k: Renk sayısı
    
    Returns:
        np.ndarray: Baskın renkler (LAB formatında)
    """
    # LAB'a dönüştür
    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    
    # K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    return centers


def renkleri_grupla(renkler_lab: list, esik: float = 10.0) -> dict:
    """
    Delta E eşiğine göre benzer renkleri gruplar.
    
    Args:
        renkler_lab: LAB renk listesi [(L, a, b), ...]
        esik: Maksimum Delta E değeri (aynı grup için)
    
    Returns:
        dict: {grup_id: [renkler]}
    """
    gruplar = defaultdict(list)
    atanmis = set()
    grup_id = 0
    
    for i, renk1 in enumerate(renkler_lab):
        if i in atanmis:
            continue
        
        gruplar[grup_id].append(renk1)
        atanmis.add(i)
        
        for j, renk2 in enumerate(renkler_lab):
            if j in atanmis:
                continue
            
            de = delta_e_cie76(renk1, renk2)
            if de <= esik:
                gruplar[grup_id].append(renk2)
                atanmis.add(j)
        
        grup_id += 1
    
    return dict(gruplar)


def renk_haritasi_olustur(goruntu: np.ndarray, esik: float = 15.0) -> np.ndarray:
    """
    Benzer renkleri gruplandırarak renk haritası oluşturur.
    
    Args:
        goruntu: BGR formatında görüntü
        esik: Delta E eşiği
    
    Returns:
        np.ndarray: Gruplandırılmış renk haritası
    """
    # LAB'a dönüştür
    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Baskın renkleri bul
    dominant = dominant_renkler_bul(goruntu, k=8)
    
    # Her pikseli en yakın dominant renge ata
    h, w = goruntu.shape[:2]
    sonuc = np.zeros_like(goruntu)
    
    for i in range(h):
        for j in range(w):
            piksel_lab = lab[i, j]
            min_mesafe = float('inf')
            en_yakin = dominant[0]
            
            for renk in dominant:
                mesafe = delta_e_cie76(piksel_lab, renk)
                if mesafe < min_mesafe:
                    min_mesafe = mesafe
                    en_yakin = renk
            
            # LAB'dan BGR'ye dönüştür
            lab_piksel = np.uint8([[en_yakin]])
            bgr_piksel = cv2.cvtColor(lab_piksel.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)
            sonuc[i, j] = bgr_piksel[0, 0]
    
    return sonuc


if __name__ == "__main__":
    print("Delta E Renk Gruplama Modülü")
    
    # Örnek: İki renk arasındaki fark
    lab1 = (50, 20, 30)
    lab2 = (55, 22, 28)
    de = delta_e_cie76(lab1, lab2)
    print(f"LAB{lab1} ve LAB{lab2} arası Delta E: {de:.2f}")
