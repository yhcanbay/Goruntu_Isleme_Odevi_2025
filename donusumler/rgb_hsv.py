"""
RGB ↔ HSV Renk Uzayı Dönüşümleri
================================
Bu modül RGB ve HSV renk uzayları arasında dönüşüm fonksiyonlarını içerir.

HSV Bileşenleri:
- H (Hue): Renk tonu, 0-360 derece
- S (Saturation): Doygunluk, 0-1
- V (Value): Parlaklık, 0-1
"""

import numpy as np


def rgb_to_hsv(r: float, g: float, b: float) -> tuple:
    """
    RGB'den HSV'ye dönüşüm.
    
    Formüller:
    - V = max(R, G, B)
    - S = (max - min) / max, eğer max ≠ 0
    - H = 60° × [(G-B)/(max-min) mod 6], eğer max = R
          60° × [(B-R)/(max-min) + 2], eğer max = G
          60° × [(R-G)/(max-min) + 4], eğer max = B
    
    Args:
        r: Kırmızı değeri (0-255 veya 0-1)
        g: Yeşil değeri (0-255 veya 0-1)
        b: Mavi değeri (0-255 veya 0-1)
    
    Returns:
        tuple: (H, S, V) - H: 0-360, S: 0-1, V: 0-1
    """
    # Normalize et (0-255 ise 0-1'e çevir)
    if max(r, g, b) > 1:
        r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min
    
    # Value
    v = c_max
    
    # Saturation
    if c_max == 0:
        s = 0
    else:
        s = delta / c_max
    
    # Hue
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
    
    return (h, s, v)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    """
    HSV'den RGB'ye dönüşüm.
    
    Formüller:
    - C = V × S (Chroma)
    - X = C × (1 - |H/60 mod 2 - 1|)
    - m = V - C
    - (R', G', B') H değerine göre belirlenir
    - (R, G, B) = (R' + m, G' + m, B' + m)
    
    Args:
        h: Hue değeri (0-360)
        s: Saturation değeri (0-1)
        v: Value değeri (0-1)
    
    Returns:
        tuple: (R, G, B) - Her biri 0-255 aralığında
    """
    c = v * s  # Chroma
    h_prime = h / 60.0
    x = c * (1 - abs(h_prime % 2 - 1))
    m = v - c
    
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
    
    return (r, g, b)


def rgb_image_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    RGB görüntüyü HSV'ye dönüştür.
    
    Args:
        image: RGB görüntü (H, W, 3), değerler 0-255
    
    Returns:
        np.ndarray: HSV görüntü (H, W, 3)
    """
    hsv_image = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            h, s, v = rgb_to_hsv(r, g, b)
            hsv_image[i, j] = [h, s * 255, v * 255]
    
    return hsv_image


def hsv_image_to_rgb(hsv_image: np.ndarray) -> np.ndarray:
    """
    HSV görüntüyü RGB'ye dönüştür.
    
    Args:
        hsv_image: HSV görüntü (H, W, 3)
    
    Returns:
        np.ndarray: RGB görüntü (H, W, 3), değerler 0-255
    """
    rgb_image = np.zeros_like(hsv_image, dtype=np.uint8)
    
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            h, s, v = hsv_image[i, j]
            s = s / 255.0 if s > 1 else s
            v = v / 255.0 if v > 1 else v
            r, g, b = hsv_to_rgb(h, s, v)
            rgb_image[i, j] = [r, g, b]
    
    return rgb_image


# Test
if __name__ == "__main__":
    print("=== RGB ↔ HSV Dönüşüm Testi ===\n")
    
    test_colors = [
        ("Kırmızı", 255, 0, 0),
        ("Yeşil", 0, 255, 0),
        ("Mavi", 0, 0, 255),
        ("Sarı", 255, 255, 0),
        ("Turuncu", 255, 128, 0),
        ("Mor", 128, 0, 255),
    ]
    
    for name, r, g, b in test_colors:
        h, s, v = rgb_to_hsv(r, g, b)
        r2, g2, b2 = hsv_to_rgb(h, s, v)
        print(f"{name}:")
        print(f"  RGB({r}, {g}, {b}) → HSV({h:.1f}°, {s:.2f}, {v:.2f})")
        print(f"  HSV → RGB({r2}, {g2}, {b2})")
        print(f"  Doğrulama: {'✓' if (r, g, b) == (r2, g2, b2) else '✗'}\n")
