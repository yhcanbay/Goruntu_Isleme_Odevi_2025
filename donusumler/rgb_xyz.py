"""
RGB ↔ XYZ Renk Uzayı Dönüşümleri
================================
Bu modül RGB ve CIE XYZ renk uzayları arasında dönüşüm fonksiyonlarını içerir.

CIE XYZ:
- Cihazdan bağımsız referans renk uzayı
- Tüm görünür renkleri kapsar
- LAB dönüşümü için ara adım

Önemli Kavramlar:
- sRGB Gamma düzeltmesi (γ = 2.4)
- D65 Beyaz nokta referansı
"""

import numpy as np

# D65 Beyaz Nokta Referansı
D65_XN = 95.047
D65_YN = 100.000
D65_ZN = 108.883

# sRGB → XYZ Dönüşüm Matrisi (D65)
RGB_TO_XYZ_MATRIX = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

# XYZ → sRGB Ters Dönüşüm Matrisi (D65)
XYZ_TO_RGB_MATRIX = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
])


def gamma_expand(c: float) -> float:
    """
    sRGB → Linear RGB (Gamma Açma)
    
    Formül:
    - c / 12.92,                    eğer c ≤ 0.04045
    - ((c + 0.055) / 1.055)^2.4,   eğer c > 0.04045
    
    Args:
        c: sRGB değeri (0-1)
    
    Returns:
        float: Linear RGB değeri (0-1)
    """
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4


def gamma_compress(c: float) -> float:
    """
    Linear RGB → sRGB (Gamma Uygulama)
    
    Formül:
    - 12.92 × c,                    eğer c ≤ 0.0031308
    - 1.055 × c^(1/2.4) - 0.055,   eğer c > 0.0031308
    
    Args:
        c: Linear RGB değeri (0-1)
    
    Returns:
        float: sRGB değeri (0-1)
    """
    if c <= 0.0031308:
        return 12.92 * c
    else:
        return 1.055 * (c ** (1 / 2.4)) - 0.055


def rgb_to_xyz(r: float, g: float, b: float) -> tuple:
    """
    RGB'den XYZ'ye dönüşüm.
    
    Adımlar:
    1. RGB'yi 0-1 aralığına normalize et
    2. Gamma düzeltmesi uygula (sRGB → Linear)
    3. Dönüşüm matrisi ile çarp
    
    Args:
        r, g, b: RGB değerleri (0-255 veya 0-1)
    
    Returns:
        tuple: (X, Y, Z) - Tipik aralık: X(0-95), Y(0-100), Z(0-109)
    """
    # Normalize et
    if max(r, g, b) > 1:
        r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    # Gamma açma (sRGB → Linear)
    r_lin = gamma_expand(r)
    g_lin = gamma_expand(g)
    b_lin = gamma_expand(b)
    
    # Matris çarpımı
    rgb_lin = np.array([r_lin, g_lin, b_lin])
    xyz = RGB_TO_XYZ_MATRIX @ rgb_lin * 100
    
    return tuple(xyz)


def xyz_to_rgb(x: float, y: float, z: float) -> tuple:
    """
    XYZ'den RGB'ye dönüşüm.
    
    Adımlar:
    1. Ters matris ile çarp
    2. Gamma sıkıştırma uygula (Linear → sRGB)
    3. 0-255 aralığına ölçekle
    
    Args:
        x, y, z: XYZ değerleri
    
    Returns:
        tuple: (R, G, B) - Her biri 0-255 aralığında
    """
    # XYZ'yi 0-1 aralığına normalize et
    xyz = np.array([x / 100.0, y / 100.0, z / 100.0])
    
    # Ters matris çarpımı
    rgb_lin = XYZ_TO_RGB_MATRIX @ xyz
    
    # Gamma sıkıştırma ve clipping
    r = gamma_compress(max(0, min(1, rgb_lin[0])))
    g = gamma_compress(max(0, min(1, rgb_lin[1])))
    b = gamma_compress(max(0, min(1, rgb_lin[2])))
    
    # 0-255'e ölçekle
    r = int(round(r * 255))
    g = int(round(g * 255))
    b = int(round(b * 255))
    
    # Final clipping
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    return (r, g, b)


def rgb_image_to_xyz(image: np.ndarray) -> np.ndarray:
    """RGB görüntüyü XYZ'ye dönüştür."""
    xyz_image = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            x, y, z = rgb_to_xyz(r, g, b)
            xyz_image[i, j] = [x, y, z]
    
    return xyz_image


def xyz_image_to_rgb(xyz_image: np.ndarray) -> np.ndarray:
    """XYZ görüntüyü RGB'ye dönüştür."""
    rgb_image = np.zeros((xyz_image.shape[0], xyz_image.shape[1], 3), dtype=np.uint8)
    
    for i in range(xyz_image.shape[0]):
        for j in range(xyz_image.shape[1]):
            x, y, z = xyz_image[i, j]
            r, g, b = xyz_to_rgb(x, y, z)
            rgb_image[i, j] = [r, g, b]
    
    return rgb_image


# Test
if __name__ == "__main__":
    print("=== RGB ↔ XYZ Dönüşüm Testi ===\n")
    
    test_colors = [
        ("Beyaz", 255, 255, 255),
        ("Siyah", 0, 0, 0),
        ("Kırmızı", 255, 0, 0),
        ("Yeşil", 0, 255, 0),
        ("Mavi", 0, 0, 255),
        ("Gri", 128, 128, 128),
    ]
    
    for name, r, g, b in test_colors:
        x, y, z = rgb_to_xyz(r, g, b)
        r2, g2, b2 = xyz_to_rgb(x, y, z)
        print(f"{name}:")
        print(f"  RGB({r}, {g}, {b}) → XYZ({x:.2f}, {y:.2f}, {z:.2f})")
        print(f"  XYZ → RGB({r2}, {g2}, {b2})")
        match = abs(r-r2) <= 1 and abs(g-g2) <= 1 and abs(b-b2) <= 1
        print(f"  Doğrulama: {'✓' if match else '✗'}\n")
