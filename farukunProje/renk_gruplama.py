"""
Renk Gruplama Modülü - LAB renk uzayında K-Means ve Delta E hesaplamaları
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


def dominant_renkler_bul(goruntu: np.ndarray, k: int = 5) -> np.ndarray:
    """K-Means ile görüntüdeki baskın renkleri bulur (LAB formatında)."""
    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB)
    piksel_verisi = lab.reshape((-1, 3)).astype(np.float32)
    kriterler = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, merkezler = cv2.kmeans(piksel_verisi, k, None, kriterler, 10, cv2.KMEANS_PP_CENTERS)
    return merkezler


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIE76 Delta E formülü ile iki LAB rengi arasındaki mesafeyi hesaplar."""
    return np.sqrt(np.sum((lab1 - lab2) ** 2))


def renkleri_grupla(renkler: List[Tuple], esik: float = 20.0) -> Dict[int, List[Tuple]]:
    """Delta E mesafesine göre benzer renkleri gruplar."""
    n = len(renkler)
    ziyaret_edildi = [False] * n
    gruplar = {}
    grup_id = 0
    
    for i in range(n):
        if ziyaret_edildi[i]:
            continue
        gruplar[grup_id] = [renkler[i]]
        ziyaret_edildi[i] = True
        
        for j in range(i + 1, n):
            if ziyaret_edildi[j]:
                continue
            for grup_renk in gruplar[grup_id]:
                delta_e = delta_e_cie76(np.array(grup_renk), np.array(renkler[j]))
                if delta_e < esik:
                    gruplar[grup_id].append(renkler[j])
                    ziyaret_edildi[j] = True
                    break
        grup_id += 1
    
    return gruplar
