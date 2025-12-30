"""
XYZ ↔ LAB Renk Uzayı Dönüşümleri
================================
Bu modül CIE XYZ ve CIE LAB renk uzayları arasında dönüşüm fonksiyonlarını içerir.

CIE LAB:
- Algısal olarak uniform renk uzayı
- L*: Parlaklık (0-100)
- a*: Yeşil(-) ↔ Kırmızı(+) ekseni
- b*: Mavi(-) ↔ Sarı(+) ekseni

CIE Standart Sabitleri:
- δ = 6/29 ≈ 0.206896
- δ³ = 216/24389 ≈ 0.008856
- κ = 24389/27 ≈ 903.296
"""

import numpy as np

# CIE Standart Sabitleri
DELTA = 6 / 29                    # ≈ 0.206896551724
DELTA_CUBE = 216 / 24389          # ≈ 0.008856451679
KAPPA = 24389 / 27                # ≈ 903.296296296

# D65 Beyaz Nokta Referansı
D65_XN = 95.047
D65_YN = 100.000
D65_ZN = 108.883


def f(t: float) -> float:
    """
    LAB dönüşümü için f(t) fonksiyonu.
    
    Formül:
    - t^(1/3),                eğer t > δ³
    - (κ×t + 16) / 116,       eğer t ≤ δ³
    
    Args:
        t: Normalize XYZ değeri
    
    Returns:
        float: f(t) sonucu
    """
    if t > DELTA_CUBE:
        return t ** (1/3)
    else:
        return (KAPPA * t + 16) / 116


def f_inverse(t: float) -> float:
    """
    LAB ters dönüşümü için f⁻¹(t) fonksiyonu.
    
    Formül:
    - t³,                     eğer t > δ
    - (116×t - 16) / κ,       eğer t ≤ δ
    
    Args:
        t: f(t) değeri
    
    Returns:
        float: Orijinal t değeri
    """
    if t > DELTA:
        return t ** 3
    else:
        return (116 * t - 16) / KAPPA


def xyz_to_lab(x: float, y: float, z: float,
               xn: float = D65_XN, yn: float = D65_YN, zn: float = D65_ZN) -> tuple:
    """
    XYZ'den LAB'a dönüşüm.
    
    Adımlar:
    1. XYZ'yi beyaz noktaya göre normalize et
    2. f(t) fonksiyonunu uygula
    3. L*, a*, b* hesapla
    
    Formüller:
    - L* = 116 × f(Y/Yn) - 16
    - a* = 500 × (f(X/Xn) - f(Y/Yn))
    - b* = 200 × (f(Y/Yn) - f(Z/Zn))
    
    Args:
        x, y, z: XYZ değerleri
        xn, yn, zn: Beyaz nokta referansı (varsayılan D65)
    
    Returns:
        tuple: (L*, a*, b*)
    """
    # Normalize et
    xr = x / xn
    yr = y / yn
    zr = z / zn
    
    # f(t) uygula
    fx = f(xr)
    fy = f(yr)
    fz = f(zr)
    
    # LAB hesapla
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return (L, a, b)


def lab_to_xyz(L: float, a: float, b: float,
               xn: float = D65_XN, yn: float = D65_YN, zn: float = D65_ZN) -> tuple:
    """
    LAB'dan XYZ'ye dönüşüm.
    
    Adımlar:
    1. fy, fx, fz hesapla
    2. f⁻¹(t) fonksiyonunu uygula
    3. Beyaz nokta ile çarp
    
    Formüller:
    - fy = (L* + 16) / 116
    - fx = a*/500 + fy
    - fz = fy - b*/200
    
    Args:
        L, a, b: LAB değerleri
        xn, yn, zn: Beyaz nokta referansı (varsayılan D65)
    
    Returns:
        tuple: (X, Y, Z)
    """
    # Ara değerler
    fy = (L + 16) / 116
    fx = (a / 500) + fy
    fz = fy - (b / 200)
    
    # f⁻¹(t) uygula ve beyaz nokta ile çarp
    X = xn * f_inverse(fx)
    Y = yn * f_inverse(fy)
    Z = zn * f_inverse(fz)
    
    return (X, Y, Z)


def xyz_image_to_lab(xyz_image: np.ndarray) -> np.ndarray:
    """XYZ görüntüyü LAB'a dönüştür."""
    lab_image = np.zeros_like(xyz_image, dtype=np.float32)
    
    for i in range(xyz_image.shape[0]):
        for j in range(xyz_image.shape[1]):
            x, y, z = xyz_image[i, j]
            L, a, b = xyz_to_lab(x, y, z)
            lab_image[i, j] = [L, a, b]
    
    return lab_image


def lab_image_to_xyz(lab_image: np.ndarray) -> np.ndarray:
    """LAB görüntüyü XYZ'ye dönüştür."""
    xyz_image = np.zeros_like(lab_image, dtype=np.float32)
    
    for i in range(lab_image.shape[0]):
        for j in range(lab_image.shape[1]):
            L, a, b = lab_image[i, j]
            x, y, z = lab_to_xyz(L, a, b)
            xyz_image[i, j] = [x, y, z]
    
    return xyz_image


# Test
if __name__ == "__main__":
    print("=== XYZ ↔ LAB Dönüşüm Testi ===\n")
    print(f"Sabitler: δ={DELTA:.6f}, δ³={DELTA_CUBE:.6f}, κ={KAPPA:.3f}\n")
    
    # XYZ test değerleri (tipik renkler)
    test_colors = [
        ("Beyaz (D65)", D65_XN, D65_YN, D65_ZN),
        ("Siyah", 0, 0, 0),
        ("Orta Gri", 20.52, 21.59, 23.52),
        ("Kırmızı", 41.24, 21.26, 1.93),
        ("Yeşil", 35.76, 71.52, 11.92),
        ("Mavi", 18.05, 7.22, 95.03),
    ]
    
    for name, x, y, z in test_colors:
        L, a, b = xyz_to_lab(x, y, z)
        x2, y2, z2 = lab_to_xyz(L, a, b)
        print(f"{name}:")
        print(f"  XYZ({x:.2f}, {y:.2f}, {z:.2f}) → LAB({L:.2f}, {a:.2f}, {b:.2f})")
        print(f"  LAB → XYZ({x2:.2f}, {y2:.2f}, {z2:.2f})")
        match = abs(x-x2) < 0.01 and abs(y-y2) < 0.01 and abs(z-z2) < 0.01
        print(f"  Doğrulama: {'✓' if match else '✗'}\n")
