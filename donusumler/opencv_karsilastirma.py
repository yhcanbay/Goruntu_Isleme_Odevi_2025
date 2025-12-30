"""
OpenCV Karşılaştırma
====================
Kendi dönüşüm fonksiyonlarımızı OpenCV ile karşılaştırır.
"""

import numpy as np
import cv2
from rgb_hsv import rgb_to_hsv, hsv_to_rgb
from rgb_lab import rgb_to_lab, lab_to_rgb


def compare_hsv():
    """RGB→HSV dönüşümünü OpenCV ile karşılaştır."""
    print("=== HSV Karşılaştırma ===\n")
    
    test_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 64, 32)]
    
    for r, g, b in test_colors:
        # Bizim fonksiyonumuz
        h, s, v = rgb_to_hsv(r, g, b)
        
        # OpenCV (BGR sırası!)
        bgr = np.uint8([[[b, g, r]]])
        hsv_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        
        # OpenCV'de H: 0-180, S: 0-255, V: 0-255
        h_cv = hsv_cv[0] * 2  # 0-360'a çevir
        s_cv = hsv_cv[1] / 255
        v_cv = hsv_cv[2] / 255
        
        print(f"RGB({r}, {g}, {b}):")
        print(f"  Bizim:  H={h:.1f}° S={s:.2f} V={v:.2f}")
        print(f"  OpenCV: H={h_cv:.1f}° S={s_cv:.2f} V={v_cv:.2f}")
        match = abs(h-h_cv) < 2 and abs(s-s_cv) < 0.02 and abs(v-v_cv) < 0.02
        print(f"  Eşleşme: {'✓' if match else '✗'}\n")


def compare_lab():
    """RGB→LAB dönüşümünü OpenCV ile karşılaştır."""
    print("=== LAB Karşılaştırma ===\n")
    
    test_colors = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (128, 128, 128)]
    
    for r, g, b in test_colors:
        # Bizim fonksiyonumuz
        L, a, b_val = rgb_to_lab(r, g, b)
        
        # OpenCV (BGR sırası!)
        bgr = np.uint8([[[b, g, r]]])
        lab_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]
        
        # OpenCV'de L: 0-255, a: 0-255, b: 0-255 (128 offset)
        L_cv = lab_cv[0] * 100 / 255
        a_cv = lab_cv[1] - 128
        b_cv = lab_cv[2] - 128
        
        print(f"RGB({r}, {g}, {b}):")
        print(f"  Bizim:  L={L:.1f} a={a:.1f} b={b_val:.1f}")
        print(f"  OpenCV: L={L_cv:.1f} a={a_cv:.1f} b={b_cv:.1f}")
        match = abs(L-L_cv) < 2 and abs(a-a_cv) < 3 and abs(b_val-b_cv) < 3
        print(f"  Eşleşme: {'✓' if match else '≈'}\n")


if __name__ == "__main__":
    compare_hsv()
    compare_lab()
    print("Not: Küçük farklar yuvarlama ve gamma düzeltme farklılıklarından kaynaklanır.")
