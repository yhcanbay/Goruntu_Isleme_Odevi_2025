"""
RGB ↔ HSL Renk Uzayı Dönüşümleri
================================
Bu modül RGB ve HSL renk uzayları arasında dönüşüm fonksiyonlarını içerir.

HSL Bileşenleri:
- H (Hue): Renk tonu, 0-360 derece
- S (Saturation): Doygunluk, 0-1
- L (Lightness): Açıklık, 0-1 (0=siyah, 0.5=en canlı, 1=beyaz)

HSV vs HSL Farkı:
- HSV'de V=1 → En parlak renkler
- HSL'de L=1 → Beyaz, L=0.5 → En canlı renkler
"""

import numpy as np


def rgb_to_hsl(r: float, g: float, b: float) -> tuple:
    """
    RGB'den HSL'ye dönüşüm.
    
    Formüller:
    - L = (max + min) / 2
    - S = Δ / (1 - |2L - 1|), eğer Δ ≠ 0
    - H, HSV ile aynı formül
    
    Args:
        r: Kırmızı değeri (0-255 veya 0-1)
        g: Yeşil değeri (0-255 veya 0-1)
        b: Mavi değeri (0-255 veya 0-1)
    
    Returns:
        tuple: (H, S, L) - H: 0-360, S: 0-1, L: 0-1
    """
    # Normalize et
    if max(r, g, b) > 1:
        r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min
    
    # Lightness
    l = (c_max + c_min) / 2.0
    
    # Saturation
    if delta == 0:
        s = 0
    else:
        s = delta / (1 - abs(2 * l - 1))
    
    # Hue (HSV ile aynı)
    if delta == 0:
        h = 0
    elif c_max == r:
        h = 60 * (((g - b) / delta) % 6)
    elif c_max == g:
        h = 60 * (((b - r) / delta) + 2)
    else:  # c_max == b
        h = 60 * (((r - g) / delta) + 4)
    
    if h < 0:
        h += 360
    
    return (h, s, l)


def hsl_to_rgb(h: float, s: float, l: float) -> tuple:
    """
    HSL'den RGB'ye dönüşüm.
    
    Formüller:
    - C = (1 - |2L - 1|) × S
    - X = C × (1 - |H/60 mod 2 - 1|)
    - m = L - C/2
    
    Args:
        h: Hue değeri (0-360)
        s: Saturation değeri (0-1)
        l: Lightness değeri (0-1)
    
    Returns:
        tuple: (R, G, B) - Her biri 0-255 aralığında
    """
    c = (1 - abs(2 * l - 1)) * s  # Chroma
    h_prime = h / 60.0
    x = c * (1 - abs(h_prime % 2 - 1))
    m = l - c / 2.0
    
    if 0 <= h_prime < 1:
        r_prime, g_prime, b_prime = c, x, 0
    elif 1 <= h_prime < 2:
        r_prime, g_prime, b_prime = x, c, 0
    elif 2 <= h_prime < 3:
        r_prime, g_prime, b_prime = 0, c, x
    elif 3 <= h_prime < 4:
        r_prime, g_prime, b_prime = 0, x, c
    elif 4 <= h_prime < 5:
        r_prime, g_prime, b_prime = x, 0, c
    else:  # 5 <= h_prime < 6
        r_prime, g_prime, b_prime = c, 0, x
    
    r = int(round((r_prime + m) * 255))
    g = int(round((g_prime + m) * 255))
    b = int(round((b_prime + m) * 255))
    
    # Clipping
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    return (r, g, b)


def rgb_image_to_hsl(image: np.ndarray) -> np.ndarray:
    """RGB görüntüyü HSL'ye dönüştür."""
    hsl_image = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            h, s, l = rgb_to_hsl(r, g, b)
            hsl_image[i, j] = [h, s * 255, l * 255]
    
    return hsl_image


def hsl_image_to_rgb(hsl_image: np.ndarray) -> np.ndarray:
    """HSL görüntüyü RGB'ye dönüştür."""
    rgb_image = np.zeros((hsl_image.shape[0], hsl_image.shape[1], 3), dtype=np.uint8)
    
    for i in range(hsl_image.shape[0]):
        for j in range(hsl_image.shape[1]):
            h, s, l = hsl_image[i, j]
            s = s / 255.0 if s > 1 else s
            l = l / 255.0 if l > 1 else l
            r, g, b = hsl_to_rgb(h, s, l)
            rgb_image[i, j] = [r, g, b]
    
    return rgb_image


# Test
if __name__ == "__main__":
    print("=== RGB ↔ HSL Dönüşüm Testi ===\n")
    
    test_colors = [
        ("Kırmızı", 255, 0, 0),
        ("Yeşil", 0, 255, 0),
        ("Mavi", 0, 0, 255),
        ("Beyaz", 255, 255, 255),
        ("Gri", 128, 128, 128),
        ("Koyu Kırmızı", 128, 0, 0),
    ]
    
    for name, r, g, b in test_colors:
        h, s, l = rgb_to_hsl(r, g, b)
        r2, g2, b2 = hsl_to_rgb(h, s, l)
        print(f"{name}:")
        print(f"  RGB({r}, {g}, {b}) → HSL({h:.1f}°, {s:.2f}, {l:.2f})")
        print(f"  HSL → RGB({r2}, {g2}, {b2})")
        match = abs(r-r2) <= 1 and abs(g-g2) <= 1 and abs(b-b2) <= 1
        print(f"  Doğrulama: {'✓' if match else '✗'}\n")
