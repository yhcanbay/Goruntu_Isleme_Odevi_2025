# Renk Uzayları ve Dönüşüm Formülleri - Kapsamlı Referans

Bu belge, görüntü işlemede kullanılan tüm temel renk uzaylarını ve aralarındaki matematiksel dönüşüm formüllerini eksiksiz şekilde açıklamaktadır.

---

## 📚 İçindekiler
1. [Temel Sabitler ve Tanımlar](#1-temel-sabitler-ve-tanımlar)
2. [RGB Renk Uzayı](#2-rgb-renk-uzayı)
3. [HSV Renk Uzayı](#3-hsv-renk-uzayı)
4. [HSL Renk Uzayı](#4-hsl-renk-uzayı)
5. [CIE XYZ Renk Uzayı](#5-cie-xyz-renk-uzayı)
6. [CIE LAB Renk Uzayı](#6-cie-lab-renk-uzayı)
7. [CIE LCH Renk Uzayı](#7-cie-lch-renk-uzayı)
8. [Delta E Formülleri](#8-delta-e-formülleri)
9. [Sayısal Hesaplama Örnekleri](#9-sayısal-hesaplama-örnekleri)
10. [OpenCV Özel Notları](#10-opencv-özel-notları)

---

## 1. Temel Sabitler ve Tanımlar

### 1.1 CIE Standart Sabitleri
```
δ (delta) = 6/29 = 0.206896551724
δ² = 36/841 = 0.042806183278
δ³ = 216/24389 = 0.008856451679
κ (kappa) = 24389/27 = 903.296296296
κ × δ³ = 8

3δ² = 108/841 = 0.128418549835
4/29 = 0.137931034483
16/116 = 0.137931034483
```

### 1.2 D65 Beyaz Nokta Referansı (Standart Gün Işığı)
```
Xn = 95.047
Yn = 100.000
Zn = 108.883
```

### 1.3 D50 Beyaz Nokta Referansı (Baskı için)
```
Xn = 96.422
Yn = 100.000
Zn = 82.521
```

### 1.4 sRGB Gamma Sabitleri
```
γ = 2.4
a = 0.055
Eşik değeri = 0.04045
Lineer eşik = 0.0031308
```

---

## 2. RGB Renk Uzayı

### 2.1 Tanım
RGB (Red, Green, Blue), eklemeli (additive) bir renk modelidir. Üç ana rengin farklı yoğunluklarda karıştırılmasıyla oluşur.

### 2.2 Değer Aralıkları
| Format | R | G | B |
|--------|---|---|---|
| 8-bit (uint8) | 0-255 | 0-255 | 0-255 |
| Normalize (float) | 0.0-1.0 | 0.0-1.0 | 0.0-1.0 |
| 16-bit | 0-65535 | 0-65535 | 0-65535 |

### 2.3 Normalizasyon Formülleri
```
8-bit → Normalize:
R' = R / 255
G' = G / 255
B' = B / 255

Normalize → 8-bit:
R = round(R' × 255)
G = round(G' × 255)
B = round(B' × 255)
```

### 2.4 sRGB Gamma Düzeltmesi

**sRGB → Linear (Gamma Açma):**
```
         ⎧ Csrgb / 12.92,                      eğer Csrgb ≤ 0.04045
Clinear = ⎨
         ⎩ ((Csrgb + 0.055) / 1.055)^2.4,     eğer Csrgb > 0.04045
```

**Linear → sRGB (Gamma Uygulama):**
```
        ⎧ 12.92 × Clinear,                     eğer Clinear ≤ 0.0031308
Csrgb = ⎨
        ⎩ 1.055 × Clinear^(1/2.4) - 0.055,    eğer Clinear > 0.0031308
```

---

## 3. HSV Renk Uzayı

### 3.1 Tanım
| Bileşen | Ad | Açıklama | Aralık |
|---------|-----|----------|--------|
| H | Hue (Ton) | Rengin türü | 0° - 360° |
| S | Saturation (Doygunluk) | Rengin canlılığı | 0 - 1 |
| V | Value (Değer) | Parlaklık | 0 - 1 |

### 3.2 RGB → HSV Dönüşümü

**Girdiler:** R, G, B ∈ [0, 1]

**Adım 1: Yardımcı Değerler**
```
Cmax = max(R, G, B)
Cmin = min(R, G, B)
Δ = Cmax - Cmin
```

**Adım 2: Value (V)**
```
V = Cmax
```

**Adım 3: Saturation (S)**
```
    ⎧ 0,           eğer Cmax = 0
S = ⎨
    ⎩ Δ / Cmax,    eğer Cmax ≠ 0
```

**Adım 4: Hue (H)**
```
       ⎧ tanımsız (0),                         eğer Δ = 0
       ⎪
       ⎪ 60° × [(G - B) / Δ mod 6],           eğer Cmax = R
H =    ⎨
       ⎪ 60° × [(B - R) / Δ + 2],             eğer Cmax = G
       ⎪
       ⎩ 60° × [(R - G) / Δ + 4],             eğer Cmax = B

Eğer H < 0 ise: H = H + 360°
```

### 3.3 HSV → RGB Dönüşümü

**Girdiler:** H ∈ [0°, 360°], S ∈ [0, 1], V ∈ [0, 1]

**Adım 1: Yardımcı Değerler**
```
C = V × S                        (Chroma)
H' = H / 60°
X = C × (1 - |H' mod 2 - 1|)
m = V - C
```

**Adım 2: Ara RGB Değerleri (H' değerine göre)**
```
H' aralığı      (R', G', B')
─────────────────────────────
0 ≤ H' < 1      (C, X, 0)
1 ≤ H' < 2      (X, C, 0)
2 ≤ H' < 3      (0, C, X)
3 ≤ H' < 4      (0, X, C)
4 ≤ H' < 5      (X, 0, C)
5 ≤ H' < 6      (C, 0, X)
```

**Adım 3: Son RGB Değerleri**
```
R = R' + m
G = G' + m
B = B' + m
```

---

## 4. HSL Renk Uzayı

### 4.1 Tanım
| Bileşen | Ad | Açıklama | Aralık |
|---------|-----|----------|--------|
| H | Hue (Ton) | Rengin türü | 0° - 360° |
| S | Saturation (Doygunluk) | Rengin canlılığı | 0 - 1 |
| L | Lightness (Açıklık) | Beyazlık miktarı | 0 - 1 |

### 4.2 HSV vs HSL Farkı
```
HSV: V=1 → En parlak renkler (beyaz değil)
HSL: L=1 → Beyaz, L=0.5 → En canlı renkler
```

### 4.3 RGB → HSL Dönüşümü

**Girdiler:** R, G, B ∈ [0, 1]

**Adım 1: Yardımcı Değerler**
```
Cmax = max(R, G, B)
Cmin = min(R, G, B)
Δ = Cmax - Cmin
```

**Adım 2: Lightness (L)**
```
L = (Cmax + Cmin) / 2
```

**Adım 3: Saturation (S)**
```
    ⎧ 0,                              eğer Δ = 0
S = ⎨
    ⎩ Δ / (1 - |2L - 1|),            eğer Δ ≠ 0
```

**Adım 4: Hue (H)**
```
H hesaplaması HSV ile aynıdır (Bölüm 3.2, Adım 4)
```

### 4.4 HSL → RGB Dönüşümü

**Girdiler:** H ∈ [0°, 360°], S ∈ [0, 1], L ∈ [0, 1]

**Adım 1: Yardımcı Değerler**
```
C = (1 - |2L - 1|) × S              (Chroma)
H' = H / 60°
X = C × (1 - |H' mod 2 - 1|)
m = L - C/2
```

**Adım 2 ve 3:** HSV → RGB ile aynı (Bölüm 3.3)

---

## 5. CIE XYZ Renk Uzayı

### 5.1 Tanım
CIE XYZ, tüm görünür renkleri kapsayan, cihazdan bağımsız bir renk uzayıdır.

| Bileşen | Açıklama |
|---------|----------|
| X | Kırmızı-yeşil karışımı |
| Y | Parlaklık (luminance) |
| Z | Mavi ağırlıklı bileşen |

### 5.2 RGB → XYZ Dönüşümü

**Adım 1: sRGB → Linear RGB**
```
Rlin = gamma_açma(R')
Glin = gamma_açma(G')
Blin = gamma_açma(B')

(Bkz. Bölüm 2.4 Gamma Düzeltmesi)
```

**Adım 2: Matris Çarpımı (D65)**
```
⎡ X ⎤   ⎡ 0.4124564  0.3575761  0.1804375 ⎤   ⎡ Rlin ⎤
⎢ Y ⎥ = ⎢ 0.2126729  0.7151522  0.0721750 ⎥ × ⎢ Glin ⎥ × 100
⎣ Z ⎦   ⎣ 0.0193339  0.1191920  0.9503041 ⎦   ⎣ Blin ⎦
```

**Açık Formüller:**
```
X = (0.4124564×Rlin + 0.3575761×Glin + 0.1804375×Blin) × 100
Y = (0.2126729×Rlin + 0.7151522×Glin + 0.0721750×Blin) × 100
Z = (0.0193339×Rlin + 0.1191920×Glin + 0.9503041×Blin) × 100
```

### 5.3 XYZ → RGB Dönüşümü

**Adım 1: Ters Matris Çarpımı (D65)**
```
⎡ Rlin ⎤   ⎡  3.2404542  -1.5371385  -0.4985314 ⎤   ⎡ X/100 ⎤
⎢ Glin ⎥ = ⎢ -0.9692660   1.8760108   0.0415560 ⎥ × ⎢ Y/100 ⎥
⎣ Blin ⎦   ⎣  0.0556434  -0.2040259   1.0572252 ⎦   ⎣ Z/100 ⎦
```

**Açık Formüller:**
```
Rlin = 3.2404542×(X/100) - 1.5371385×(Y/100) - 0.4985314×(Z/100)
Glin = -0.9692660×(X/100) + 1.8760108×(Y/100) + 0.0415560×(Z/100)
Blin = 0.0556434×(X/100) - 0.2040259×(Y/100) + 1.0572252×(Z/100)
```

**Adım 2: Linear RGB → sRGB**
```
R' = gamma_uygula(Rlin)
G' = gamma_uygula(Glin)
B' = gamma_uygula(Blin)

(Bkz. Bölüm 2.4 Gamma Düzeltmesi)
```

**Adım 3: Clipping (Gerekirse)**
```
R', G', B' değerlerini [0, 1] aralığına sınırla
```

---

## 6. CIE LAB Renk Uzayı

### 6.1 Tanım
| Bileşen | Aralık | Açıklama |
|---------|--------|----------|
| L* | 0 - 100 | Parlaklık (Siyah=0, Beyaz=100) |
| a* | ≈ -128 - +127 | Yeşil(-) ↔ Kırmızı(+) |
| b* | ≈ -128 - +127 | Mavi(-) ↔ Sarı(+) |

### 6.2 XYZ → LAB Dönüşümü

**Adım 1: Normalize Et**
```
xr = X / Xn    (Xn = 95.047)
yr = Y / Yn    (Yn = 100.000)
zr = Z / Zn    (Zn = 108.883)
```

**Adım 2: f(t) Fonksiyonu**
```
         ⎧ t^(1/3),                      eğer t > δ³ (0.008856)
f(t) =   ⎨
         ⎩ (κ×t + 16) / 116,             eğer t ≤ δ³

Burada: κ = 903.296, δ³ = 0.008856
Alternatif form: (t / 3δ²) + (4/29)
```

**Adım 3: LAB Hesaplama**
```
L* = 116 × f(yr) - 16
a* = 500 × (f(xr) - f(yr))
b* = 200 × (f(yr) - f(zr))
```

### 6.3 LAB → XYZ Dönüşümü

**Adım 1: Ara Değerler**
```
fy = (L* + 16) / 116
fx = (a* / 500) + fy
fz = fy - (b* / 200)
```

**Adım 2: f⁻¹(t) Ters Fonksiyonu**
```
           ⎧ t³,                         eğer t > δ (0.206896)
f⁻¹(t) =   ⎨
           ⎩ (116×t - 16) / κ,           eğer t ≤ δ

Alternatif form: 3δ² × (t - 4/29)
```

**Adım 3: XYZ Hesaplama**
```
X = Xn × f⁻¹(fx)    (Xn = 95.047)
Y = Yn × f⁻¹(fy)    (Yn = 100.000)
Z = Zn × f⁻¹(fz)    (Zn = 108.883)
```

---

## 7. CIE LCH Renk Uzayı

### 7.1 Tanım
LCH, LAB'ın silindirik koordinat formudur.

| Bileşen | Aralık | Açıklama |
|---------|--------|----------|
| L* | 0 - 100 | Parlaklık (LAB ile aynı) |
| C* | 0 - ≈181 | Chroma (renk yoğunluğu) |
| h° | 0° - 360° | Hue açısı |

### 7.2 LAB → LCH Dönüşümü
```
L* = L*                              (değişmez)
C* = √(a*² + b*²)
h° = atan2(b*, a*) × (180/π)

Eğer h° < 0 ise: h° = h° + 360°
```

### 7.3 LCH → LAB Dönüşümü
```
L* = L*                              (değişmez)
a* = C* × cos(h° × π/180)
b* = C* × sin(h° × π/180)
```

---

## 8. Delta E Formülleri

### 8.1 CIE76 (ΔE*ab) - Temel Öklid Mesafesi
```
ΔE*76 = √[(L₁* - L₂*)² + (a₁* - a₂*)² + (b₁* - b₂*)²]
      = √[ΔL*² + Δa*² + Δb*²]
```

### 8.2 CIE94 (ΔE*94)

**Yardımcı Hesaplamalar:**
```
ΔL* = L₁* - L₂*
C₁* = √(a₁*² + b₁*²)
C₂* = √(a₂*² + b₂*²)
ΔC*ab = C₁* - C₂*
Δa* = a₁* - a₂*
Δb* = b₁* - b₂*
ΔH*ab = √(Δa*² + Δb*² - ΔC*ab²)
```

**Ağırlık Fonksiyonları:**
```
SL = 1
SC = 1 + K₁ × C₁*
SH = 1 + K₂ × C₁*

Grafik sanatları: K₁ = 0.045, K₂ = 0.015
Tekstil: K₁ = 0.048, K₂ = 0.014
```

**Parametrik Faktörler:**
```
Grafik sanatları: kL = 1, kC = 1, kH = 1
Tekstil: kL = 2, kC = 1, kH = 1
```

**Son Formül:**
```
ΔE*94 = √[(ΔL*/(kL×SL))² + (ΔC*ab/(kC×SC))² + (ΔH*ab/(kH×SH))²]
```

### 8.3 CIEDE2000 (ΔE*00) - Tam Formül

**Adım 1: LAB Değerleri ve Ortalama L***
```
L̄ = (L₁* + L₂*) / 2
```

**Adım 2: a' Düzeltmesi**
```
C*₁ = √(a₁*² + b₁*²)
C*₂ = √(a₂*² + b₂*²)
C̄* = (C*₁ + C*₂) / 2

G = 0.5 × (1 - √(C̄*⁷ / (C̄*⁷ + 25⁷)))

a'₁ = a₁* × (1 + G)
a'₂ = a₂* × (1 + G)
```

**Adım 3: Yeni C' ve h' Hesaplama**
```
C'₁ = √(a'₁² + b₁*²)
C'₂ = √(a'₂² + b₂*²)

h'₁ = atan2(b₁*, a'₁) mod 360°
h'₂ = atan2(b₂*, a'₂) mod 360°
```

**Adım 4: Δ Değerleri**
```
ΔL' = L₂* - L₁*
ΔC' = C'₂ - C'₁

      ⎧ h'₂ - h'₁,                    eğer |h'₂ - h'₁| ≤ 180°
Δh' = ⎨ h'₂ - h'₁ + 360°,            eğer h'₂ - h'₁ < -180°
      ⎩ h'₂ - h'₁ - 360°,            eğer h'₂ - h'₁ > 180°

ΔH' = 2 × √(C'₁ × C'₂) × sin(Δh'/2 × π/180)
```

**Adım 5: Ortalama Değerler**
```
L̄' = (L₁* + L₂*) / 2
C̄' = (C'₁ + C'₂) / 2

      ⎧ (h'₁ + h'₂) / 2,                        eğer |h'₁ - h'₂| ≤ 180°
H̄' = ⎨ (h'₁ + h'₂ + 360°) / 2,                 eğer |h'₁ - h'₂| > 180° ve h'₁+h'₂ < 360°
      ⎩ (h'₁ + h'₂ - 360°) / 2,                 eğer |h'₁ - h'₂| > 180° ve h'₁+h'₂ ≥ 360°
```

**Adım 6: T Faktörü**
```
T = 1 - 0.17×cos((H̄' - 30°)×π/180)
      + 0.24×cos((2×H̄')×π/180)
      + 0.32×cos((3×H̄' + 6°)×π/180)
      - 0.20×cos((4×H̄' - 63°)×π/180)
```

**Adım 7: Ağırlık Fonksiyonları**
```
SL = 1 + (0.015 × (L̄' - 50)²) / √(20 + (L̄' - 50)²)
SC = 1 + 0.045 × C̄'
SH = 1 + 0.015 × C̄' × T
```

**Adım 8: Rotasyon Terimi**
```
Δθ = 30° × exp(-((H̄' - 275°)/25)²)
RC = 2 × √(C̄'⁷ / (C̄'⁷ + 25⁷))
RT = -sin(2×Δθ×π/180) × RC
```

**Adım 9: Son Formül**
```
ΔE*00 = √[(ΔL'/(kL×SL))² + (ΔC'/(kC×SC))² + (ΔH'/(kH×SH))² 
         + RT×(ΔC'/(kC×SC))×(ΔH'/(kH×SH))]

Genellikle: kL = kC = kH = 1
```

### 8.4 Delta E Yorumlama Tablosu
| ΔE Değeri | Algı Seviyesi |
|-----------|---------------|
| 0 - 1 | Algılanamaz fark |
| 1 - 2 | Eğitimli göz gerektirir |
| 2 - 3.5 | Yakından fark edilir |
| 3.5 - 5 | Belirgin fark |
| 5 - 10 | Açıkça farklı |
| > 10 | Farklı renkler |

---

## 9. Sayısal Hesaplama Örnekleri

### 9.1 Örnek: RGB → HSV
**Girdi:** RGB(180, 75, 40) → Normalize: (0.706, 0.294, 0.157)
```
Cmax = 0.706, Cmin = 0.157, Δ = 0.549

V = 0.706 = 70.6%
S = 0.549 / 0.706 = 0.778 = 77.8%
H = 60° × [(0.294 - 0.157) / 0.549 mod 6]
  = 60° × [0.249] = 14.97° ≈ 15°

Sonuç: HSV(15°, 77.8%, 70.6%) - Turuncu-kahverengi
```

### 9.2 Örnek: RGB → LAB
**Girdi:** RGB(100, 150, 200) → Normalize: (0.392, 0.588, 0.784)

**Adım 1: Gamma Açma**
```
Rlin = ((0.392 + 0.055)/1.055)^2.4 = 0.127
Glin = ((0.588 + 0.055)/1.055)^2.4 = 0.305
Blin = ((0.784 + 0.055)/1.055)^2.4 = 0.578
```

**Adım 2: RGB → XYZ**
```
X = (0.4125×0.127 + 0.3576×0.305 + 0.1804×0.578) × 100 = 26.66
Y = (0.2127×0.127 + 0.7152×0.305 + 0.0722×0.578) × 100 = 29.69
Z = (0.0193×0.127 + 0.1192×0.305 + 0.9503×0.578) × 100 = 58.84
```

**Adım 3: XYZ → LAB**
```
xr = 26.66/95.047 = 0.281
yr = 29.69/100.0 = 0.297
zr = 58.84/108.883 = 0.540

f(0.281) = 0.655, f(0.297) = 0.668, f(0.540) = 0.815

L* = 116 × 0.668 - 16 = 61.5
a* = 500 × (0.655 - 0.668) = -6.5
b* = 200 × (0.668 - 0.815) = -29.4

Sonuç: LAB(61.5, -6.5, -29.4) - Açık mavi
```

### 9.3 Örnek: Delta E Hesaplama
**Renk 1:** LAB(50, 20, 30)
**Renk 2:** LAB(52, 22, 28)

**CIE76:**
```
ΔE*76 = √[(50-52)² + (20-22)² + (30-28)²]
      = √[4 + 4 + 4]
      = √12 = 3.46

→ "Yakından fark edilir" seviyesi
```

---

## 10. OpenCV Özel Notları

### 10.1 OpenCV Değer Aralıkları (8-bit)
| Uzay | Kanal 1 | Kanal 2 | Kanal 3 |
|------|---------|---------|---------|
| BGR | B: 0-255 | G: 0-255 | R: 0-255 |
| HSV | H: 0-180 | S: 0-255 | V: 0-255 |
| HLS | H: 0-180 | L: 0-255 | S: 0-255 |
| LAB | L: 0-255 | a: 0-255 | b: 0-255 |

### 10.2 OpenCV Dönüşümleri
```
OpenCV HSV H değeri = Standart H / 2
OpenCV LAB a değeri = Standart a* + 128
OpenCV LAB b değeri = Standart b* + 128
OpenCV LAB L değeri = Standart L* × 255/100
```

### 10.3 HSV Renk Aralıkları (OpenCV için)
| Renk | H Alt | H Üst | S Alt | V Alt |
|------|-------|-------|-------|-------|
| Kırmızı | 0-10, 160-180 | - | 100 | 100 |
| Turuncu | 10 | 25 | 100 | 100 |
| Sarı | 25 | 35 | 100 | 100 |
| Yeşil | 35 | 85 | 100 | 100 |
| Cyan | 85 | 100 | 100 | 100 |
| Mavi | 100 | 130 | 100 | 100 |
| Mor | 130 | 160 | 100 | 100 |

### 10.4 Python/OpenCV Dönüşüm Fonksiyonları
```python
# Dönüşüm kodları
cv2.COLOR_BGR2HSV
cv2.COLOR_HSV2BGR
cv2.COLOR_BGR2LAB
cv2.COLOR_LAB2BGR
cv2.COLOR_BGR2HLS
cv2.COLOR_HLS2BGR
cv2.COLOR_BGR2XYZ
cv2.COLOR_XYZ2BGR
```

---

## 📖 Referanslar

1. CIE 15:2004 - Colorimetry (3rd Edition)
2. Gonzalez & Woods, "Digital Image Processing", Chapter 6
3. IEC 61966-2-1:1999 (sRGB Standard)
4. Sharma, Wu, Dalal (2005) - CIEDE2000 Color-Difference Formula
5. Bruce Lindbloom - Color Space Mathematics (brucelindbloom.com)
6. OpenCV Documentation - Color Space Conversions

---

*Bu belge, Renkli Görüntü İşleme ve Renk Uzayları projesi için hazırlanmıştır.*
*Son güncelleme: 29 Aralık 2024*
