"""
LAB ↔ LCH Renk Uzayı Dönüşümleri
================================
Bu modül CIE LAB ve CIE LCH renk uzayları arasında dönüşüm fonksiyonlarını içerir.

CIE LCH:
- LAB'ın silindirik (polar) koordinat formu
- L*: Parlaklık (LAB ile aynı, 0-100)
- C*: Chroma (renk yoğunluğu, 0-~181)
- h°: Hue açısı (renk tonu, 0-360°)

Dönüşüm:
- Kartezyen (a*, b*) → Polar (C*, h°)
"""

import numpy as np
import math


def lab_to_lch(L: float, a: float, b: float) -> tuple:
    """
    LAB'dan LCH'ye dönüşüm.
    
    Formüller:
    - L* = L* (değişmez)
    - C* = √(a*² + b*²)
    - h° = atan2(b*, a*) × (180/π)
    
    Args:
        L: Parlaklık (0-100)
        a: Yeşil-Kırmızı ekseni
        b: Mavi-Sarı ekseni
    
    Returns:
        tuple: (L*, C*, h°)
    """
    # Chroma hesapla
    C = math.sqrt(a**2 + b**2)
    
    # Hue açısı hesapla (derece cinsinden)
    h = math.atan2(b, a) * (180 / math.pi)
    
    # Negatif açıları düzelt (0-360 aralığına getir)
    if h < 0:
        h += 360
    
    return (L, C, h)


def lch_to_lab(L: float, C: float, h: float) -> tuple:
    """
    LCH'den LAB'a dönüşüm.
    
    Formüller:
    - L* = L* (değişmez)
    - a* = C* × cos(h° × π/180)
    - b* = C* × sin(h° × π/180)
    
    Args:
        L: Parlaklık (0-100)
        C: Chroma (renk yoğunluğu)
        h: Hue açısı (0-360°)
    
    Returns:
        tuple: (L*, a*, b*)
    """
    # Açıyı radyana çevir
    h_rad = h * (math.pi / 180)
    
    # a* ve b* hesapla
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)
    
    return (L, a, b)


def lab_image_to_lch(lab_image: np.ndarray) -> np.ndarray:
    """LAB görüntüyü LCH'ye dönüştür."""
    lch_image = np.zeros_like(lab_image, dtype=np.float32)
    
    for i in range(lab_image.shape[0]):
        for j in range(lab_image.shape[1]):
            L, a, b = lab_image[i, j]
            L_out, C, h = lab_to_lch(L, a, b)
            lch_image[i, j] = [L_out, C, h]
    
    return lch_image


def lch_image_to_lab(lch_image: np.ndarray) -> np.ndarray:
    """LCH görüntüyü LAB'a dönüştür."""
    lab_image = np.zeros_like(lch_image, dtype=np.float32)
    
    for i in range(lch_image.shape[0]):
        for j in range(lch_image.shape[1]):
            L, C, h = lch_image[i, j]
            L_out, a, b = lch_to_lab(L, C, h)
            lab_image[i, j] = [L_out, a, b]
    
    return lab_image


def get_hue_name(h: float) -> str:
    """Hue açısından renk adı döndür."""
    if h < 30 or h >= 330:
        return "Kırmızı"
    elif h < 60:
        return "Turuncu"
    elif h < 90:
        return "Sarı"
    elif h < 150:
        return "Yeşil"
    elif h < 210:
        return "Cyan"
    elif h < 270:
        return "Mavi"
    elif h < 330:
        return "Mor/Magenta"
    return "Bilinmiyor"


# Test
if __name__ == "__main__":
    print("=== LAB ↔ LCH Dönüşüm Testi ===\n")
    
    # LAB test değerleri
    test_colors = [
        ("Beyaz", 100, 0, 0),
        ("Siyah", 0, 0, 0),
        ("Kırmızı", 53.23, 80.11, 67.22),
        ("Yeşil", 87.74, -86.18, 83.18),
        ("Mavi", 32.30, 79.20, -107.86),
        ("Sarı", 97.14, -21.56, 94.48),
        ("Cyan", 91.11, -48.09, -14.13),
        ("Magenta", 60.32, 98.25, -60.84),
    ]
    
    for name, L, a, b in test_colors:
        L_out, C, h = lab_to_lch(L, a, b)
        L2, a2, b2 = lch_to_lab(L_out, C, h)
        
        print(f"{name}:")
        print(f"  LAB({L:.2f}, {a:.2f}, {b:.2f})")
        print(f"  → LCH({L_out:.2f}, {C:.2f}, {h:.1f}°) [{get_hue_name(h)}]")
        print(f"  → LAB({L2:.2f}, {a2:.2f}, {b2:.2f})")
        match = abs(L-L2) < 0.01 and abs(a-a2) < 0.01 and abs(b-b2) < 0.01
        print(f"  Doğrulama: {'✓' if match else '✗'}\n")
