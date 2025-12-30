"""
RGB ↔ LAB Tam Dönüşüm Zinciri
=============================
RGB → XYZ → LAB ve LAB → XYZ → RGB tam yol dönüşümleri
"""

import numpy as np
from rgb_xyz import rgb_to_xyz, xyz_to_rgb, gamma_expand, gamma_compress
from xyz_lab import xyz_to_lab, lab_to_xyz


def rgb_to_lab(r: float, g: float, b: float) -> tuple:
    """
    RGB'den LAB'a doğrudan dönüşüm.
    
    Dönüşüm Zinciri: RGB → Linear RGB → XYZ → LAB
    
    Args:
        r, g, b: RGB değerleri (0-255 veya 0-1)
    
    Returns:
        tuple: (L*, a*, b*)
    """
    x, y, z = rgb_to_xyz(r, g, b)
    L, a, b_val = xyz_to_lab(x, y, z)
    return (L, a, b_val)


def lab_to_rgb(L: float, a: float, b: float) -> tuple:
    """
    LAB'dan RGB'ye doğrudan dönüşüm.
    
    Dönüşüm Zinciri: LAB → XYZ → Linear RGB → RGB
    
    Args:
        L, a, b: LAB değerleri
    
    Returns:
        tuple: (R, G, B) - 0-255 aralığında
    """
    x, y, z = lab_to_xyz(L, a, b)
    r, g, b_val = xyz_to_rgb(x, y, z)
    return (r, g, b_val)


def rgb_image_to_lab(image: np.ndarray) -> np.ndarray:
    """RGB görüntüyü LAB'a dönüştür."""
    lab_image = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            L, a, b_val = rgb_to_lab(r, g, b)
            lab_image[i, j] = [L, a, b_val]
    
    return lab_image


def lab_image_to_rgb(lab_image: np.ndarray) -> np.ndarray:
    """LAB görüntüyü RGB'ye dönüştür."""
    rgb_image = np.zeros((lab_image.shape[0], lab_image.shape[1], 3), dtype=np.uint8)
    
    for i in range(lab_image.shape[0]):
        for j in range(lab_image.shape[1]):
            L, a, b = lab_image[i, j]
            r, g, b_val = lab_to_rgb(L, a, b)
            rgb_image[i, j] = [r, g, b_val]
    
    return rgb_image


if __name__ == "__main__":
    print("=== RGB ↔ LAB Dönüşüm Testi ===\n")
    
    test_colors = [
        ("Beyaz", 255, 255, 255),
        ("Siyah", 0, 0, 0),
        ("Kırmızı", 255, 0, 0),
        ("Yeşil", 0, 255, 0),
        ("Mavi", 0, 0, 255),
        ("Gri", 128, 128, 128),
    ]
    
    for name, r, g, b in test_colors:
        L, a, b_val = rgb_to_lab(r, g, b)
        r2, g2, b2 = lab_to_rgb(L, a, b_val)
        print(f"{name}:")
        print(f"  RGB({r},{g},{b}) → LAB({L:.1f},{a:.1f},{b_val:.1f}) → RGB({r2},{g2},{b2})")
        match = abs(r-r2)<=1 and abs(g-g2)<=1 and abs(b-b2)<=1
        print(f"  Doğrulama: {'✓' if match else '✗'}\n")
