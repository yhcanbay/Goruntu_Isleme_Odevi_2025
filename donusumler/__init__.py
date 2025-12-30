"""
Renk Uzayı Dönüşümleri - Ana Modül
==================================
Tüm dönüşüm fonksiyonlarını tek bir modülde toplar.

Kullanım:
    from donusumler import rgb_to_hsv, hsv_to_rgb, rgb_to_lab, lab_to_rgb
"""

# RGB ↔ HSV
from rgb_hsv import rgb_to_hsv, hsv_to_rgb, rgb_image_to_hsv, hsv_image_to_rgb

# RGB ↔ HSL
from rgb_hsl import rgb_to_hsl, hsl_to_rgb, rgb_image_to_hsl, hsl_image_to_rgb

# RGB ↔ XYZ
from rgb_xyz import (rgb_to_xyz, xyz_to_rgb, rgb_image_to_xyz, xyz_image_to_rgb,
                     gamma_expand, gamma_compress)

# XYZ ↔ LAB
from xyz_lab import xyz_to_lab, lab_to_xyz, xyz_image_to_lab, lab_image_to_xyz, f, f_inverse

# LAB ↔ LCH
from lab_lch import lab_to_lch, lch_to_lab, lab_image_to_lch, lch_image_to_lab

# RGB ↔ LAB (Tam Zincir)
from rgb_lab import rgb_to_lab, lab_to_rgb, rgb_image_to_lab, lab_image_to_rgb

# Delta E
from delta_e import delta_e_cie76, delta_e_cie94, delta_e_ciede2000, interpret_delta_e


__all__ = [
    # HSV
    'rgb_to_hsv', 'hsv_to_rgb', 'rgb_image_to_hsv', 'hsv_image_to_rgb',
    # HSL
    'rgb_to_hsl', 'hsl_to_rgb', 'rgb_image_to_hsl', 'hsl_image_to_rgb',
    # XYZ
    'rgb_to_xyz', 'xyz_to_rgb', 'rgb_image_to_xyz', 'xyz_image_to_rgb',
    'gamma_expand', 'gamma_compress',
    # LAB
    'xyz_to_lab', 'lab_to_xyz', 'xyz_image_to_lab', 'lab_image_to_xyz',
    'rgb_to_lab', 'lab_to_rgb', 'rgb_image_to_lab', 'lab_image_to_rgb',
    # LCH
    'lab_to_lch', 'lch_to_lab', 'lab_image_to_lch', 'lch_image_to_lab',
    # Delta E
    'delta_e_cie76', 'delta_e_cie94', 'delta_e_ciede2000', 'interpret_delta_e',
]


if __name__ == "__main__":
    print("=== Renk Uzayı Dönüşümleri - Kapsamlı Test ===\n")
    
    # Test RGB değeri
    r, g, b = 180, 100, 60
    print(f"Orijinal RGB: ({r}, {g}, {b})\n")
    
    # HSV
    h, s, v = rgb_to_hsv(r, g, b)
    r2, g2, b2 = hsv_to_rgb(h, s, v)
    print(f"HSV: ({h:.1f}°, {s:.2f}, {v:.2f}) → RGB: ({r2}, {g2}, {b2})")
    
    # HSL
    h, s, l = rgb_to_hsl(r, g, b)
    r2, g2, b2 = hsl_to_rgb(h, s, l)
    print(f"HSL: ({h:.1f}°, {s:.2f}, {l:.2f}) → RGB: ({r2}, {g2}, {b2})")
    
    # XYZ
    x, y, z = rgb_to_xyz(r, g, b)
    r2, g2, b2 = xyz_to_rgb(x, y, z)
    print(f"XYZ: ({x:.2f}, {y:.2f}, {z:.2f}) → RGB: ({r2}, {g2}, {b2})")
    
    # LAB
    L, a, b_val = rgb_to_lab(r, g, b)
    r2, g2, b2 = lab_to_rgb(L, a, b_val)
    print(f"LAB: ({L:.2f}, {a:.2f}, {b_val:.2f}) → RGB: ({r2}, {g2}, {b2})")
    
    # LCH
    L2, C, h_deg = lab_to_lch(L, a, b_val)
    L3, a2, b2 = lch_to_lab(L2, C, h_deg)
    print(f"LCH: ({L2:.2f}, {C:.2f}, {h_deg:.1f}°)")
    
    # Delta E
    lab1 = (50, 20, 30)
    lab2 = (55, 25, 28)
    de = delta_e_cie76(lab1, lab2)
    print(f"\nDelta E (CIE76) between {lab1} and {lab2}: {de:.2f}")
    print(f"Yorum: {interpret_delta_e(de)}")
