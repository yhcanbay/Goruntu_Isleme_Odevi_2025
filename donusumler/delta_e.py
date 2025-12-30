"""
Delta E Renk Farkı Metrikleri
=============================
CIE76, CIE94 ve CIEDE2000 formülleri
"""

import math


def delta_e_cie76(lab1: tuple, lab2: tuple) -> float:
    """CIE76 Delta E: ΔE = √[(ΔL)² + (Δa)² + (Δb)²]"""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return math.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2)


def delta_e_cie94(lab1: tuple, lab2: tuple, textile: bool = False) -> float:
    """CIE94 Delta E: Endüstriyel uygulamalar için."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    kL, K1, K2 = (2, 0.048, 0.014) if textile else (1, 0.045, 0.015)
    
    C1 = math.sqrt(a1**2 + b1**2)
    delta_L = L1 - L2
    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_C = C1 - math.sqrt(a2**2 + b2**2)
    delta_H = math.sqrt(max(0, delta_a**2 + delta_b**2 - delta_C**2))
    
    SL, SC, SH = 1, 1 + K1*C1, 1 + K2*C1
    
    return math.sqrt((delta_L/(kL*SL))**2 + (delta_C/SC)**2 + (delta_H/SH)**2)


def delta_e_ciede2000(lab1: tuple, lab2: tuple) -> float:
    """CIEDE2000 Delta E: En doğru algısal metrik."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    C1, C2 = math.sqrt(a1**2+b1**2), math.sqrt(a2**2+b2**2)
    C_bar = (C1+C2)/2
    G = 0.5*(1-math.sqrt(C_bar**7/(C_bar**7+25**7)))
    
    a1p, a2p = a1*(1+G), a2*(1+G)
    C1p, C2p = math.sqrt(a1p**2+b1**2), math.sqrt(a2p**2+b2**2)
    h1p = math.degrees(math.atan2(b1,a1p))%360
    h2p = math.degrees(math.atan2(b2,a2p))%360
    
    dLp, dCp = L2-L1, C2p-C1p
    dhp = h2p-h1p
    if abs(dhp)>180: dhp -= 360*math.copysign(1,dhp)
    dHp = 2*math.sqrt(C1p*C2p)*math.sin(math.radians(dhp/2))
    
    Lbp, Cbp = (L1+L2)/2, (C1p+C2p)/2
    hbp = (h1p+h2p)/2
    if abs(h1p-h2p)>180: hbp += 180
    
    T = 1-0.17*math.cos(math.radians(hbp-30))+0.24*math.cos(math.radians(2*hbp))
    T += 0.32*math.cos(math.radians(3*hbp+6))-0.20*math.cos(math.radians(4*hbp-63))
    
    SL = 1+(0.015*(Lbp-50)**2)/math.sqrt(20+(Lbp-50)**2)
    SC, SH = 1+0.045*Cbp, 1+0.015*Cbp*T
    
    dth = 30*math.exp(-((hbp-275)/25)**2)
    RC = 2*math.sqrt(Cbp**7/(Cbp**7+25**7))
    RT = -math.sin(math.radians(2*dth))*RC
    
    return math.sqrt((dLp/SL)**2+(dCp/SC)**2+(dHp/SH)**2+RT*(dCp/SC)*(dHp/SH))


def interpret_delta_e(de: float) -> str:
    """Delta E yorumla."""
    if de < 1: return "Algılanamaz"
    if de < 2: return "Eğitimli göz gerekir"
    if de < 3.5: return "Yakından fark edilir"
    if de < 5: return "Belirgin fark"
    if de < 10: return "Açıkça farklı"
    return "Tamamen farklı"


if __name__ == "__main__":
    print("=== Delta E Testi ===\n")
    pairs = [
        ("Aynı", (50,20,30), (50,20,30)),
        ("Benzer", (50,20,30), (52,22,28)),
        ("Farklı", (50,20,30), (80,-20,-30)),
    ]
    for name, l1, l2 in pairs:
        de76 = delta_e_cie76(l1, l2)
        print(f"{name}: ΔE76={de76:.2f} → {interpret_delta_e(de76)}")
