"""
Projek Pengolahan Citra
=======================
Backend Flask + OpenCV.

Satu endpoint utama  POST /api/process  menerima:
  - image   : file gambar (PNG hasil upload / capture kamera)
  - method  : adjustment | noise | filtering | transformation | equalization | compression
  - params  : JSON berisi parameter metode tersebut

dan mengembalikan gambar hasil (data URL), histogram citra asli & hasil
(R, G, B, Gray - masing-masing 256 bin), serta informasi tambahan.
"""
import base64
import json
import math

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB

MAX_SIDE = 1600      # batas sisi terpanjang gambar input
OUT_MAX_SIDE = 2400  # batas sisi terpanjang gambar output (mis. saat scale besar)


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def odd(k, minimum=1):
    """Pastikan ukuran kernel ganjil dan >= minimum."""
    k = max(int(minimum), int(round(float(k))))
    return k if k % 2 == 1 else k + 1


def clip8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def limit_side(img, max_side):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / m
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def fmt_bytes(n):
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def histogram(img, mask=None):
    """Histogram 256 bin untuk kanal R, G, B, dan Gray."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def calc(ch):
        h = cv2.calcHist([ch], [0], mask, [256], [0, 256]).flatten()
        return h.astype(np.int64).tolist()

    b, g, r = cv2.split(img)
    return {"r": calc(r), "g": calc(g), "b": calc(b), "gray": calc(gray)}


def result(out, mask=None, info=None, enc=None):
    """Bungkus hasil sebuah metode.
    enc = (bytes, mime, ext) bila metode menentukan sendiri format outputnya."""
    return out, mask, (info or []), enc


# ----------------------------------------------------------------------
# 1. ADJUSTMENT  (brightness, contrast, sharpness, blur gaussian/median/box/bilateral)
# ----------------------------------------------------------------------
def m_adjustment(img, p):
    out = img
    info = []

    # --- Blur ---
    blur_type = p.get("blur_type", "none")
    k = odd(p.get("blur_k", 5))
    if blur_type == "gaussian":
        out = cv2.GaussianBlur(out, (k, k), 0)
    elif blur_type == "median":
        out = cv2.medianBlur(out, max(3, k))
    elif blur_type == "box":
        out = cv2.blur(out, (k, k))
    elif blur_type == "bilateral":
        d = min(k, 15)  # dibatasi supaya tidak lambat
        sigma = float(p.get("sigma_color", 75))
        out = cv2.bilateralFilter(out, d, sigma, sigma)
        if k > 15:
            info.append(["Catatan", "Diameter bilateral dibatasi 15"])

    # --- Sharpness (unsharp mask) ---
    amount = float(p.get("sharpness", 0))
    if amount > 0:
        blurred = cv2.GaussianBlur(out, (0, 0), 1.5)
        out = cv2.addWeighted(out, 1 + amount, blurred, -amount, 0)

    # --- Brightness & contrast (contrast terhadap abu-abu tengah 128) ---
    brightness = float(p.get("brightness", 0))
    contrast = float(p.get("contrast", 1))
    if brightness != 0 or contrast != 1:
        out = clip8((out.astype(np.float32) - 128) * contrast + 128 + brightness)

    # --- Saturation ---
    sat = float(p.get("saturation", 1))
    if sat != 1:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result(out, info=info)


# ----------------------------------------------------------------------
# 2. NOISE  (tambah noise + opsi menghilangkan noise)
# ----------------------------------------------------------------------
def m_noise(img, p):
    amount = float(p.get("amount", 30))
    ntype = p.get("noise_type", "gaussian")
    rng = np.random.default_rng(42)  # seed tetap -> hasil stabil saat slider digeser
    out = img.copy()
    info = []

    if amount > 0:
        f = img.astype(np.float32)
        if ntype == "gaussian":
            out = clip8(f + rng.normal(0, amount * 0.8, img.shape))
        elif ntype == "salt_pepper":
            prob = amount / 100 * 0.3
            r = rng.random(img.shape[:2])
            out[r < prob / 2] = 0
            out[r > 1 - prob / 2] = 255
        elif ntype == "speckle":
            out = clip8(f + f * rng.normal(0, amount / 100 * 0.8, img.shape))
        elif ntype == "uniform":
            a = amount * 1.5
            out = clip8(f + rng.uniform(-a, a, img.shape))
        elif ntype == "poisson":
            lam = 400 * (1 - amount / 100) + 4
            out = clip8(rng.poisson(f / 255 * lam) / lam * 255)

    # --- Denoise (opsional) ---
    den = p.get("denoise", "none")
    s = int(float(p.get("denoise_strength", 3)))
    if den == "median":
        out = cv2.medianBlur(out, odd(2 * s + 1, 3))
    elif den == "gaussian":
        kk = odd(2 * s + 1, 3)
        out = cv2.GaussianBlur(out, (kk, kk), 0)
    elif den == "bilateral":
        out = cv2.bilateralFilter(out, min(2 * s + 1, 15), 75, 75)
    elif den == "nlm":
        out = cv2.fastNlMeansDenoisingColored(out, None, s * 2, s * 2, 7, 21)

    if den != "none":
        mse = float(np.mean((out.astype(np.float32) - img.astype(np.float32)) ** 2))
        info.append(["PSNR vs input", "∞" if mse == 0 else f"{10 * math.log10(255 ** 2 / mse):.2f} dB"])
    return result(out, info=info)


# ----------------------------------------------------------------------
# 3. FILTERING  (gray scale, biner, negative, red, green, blue)
# ----------------------------------------------------------------------
def m_filtering(img, p):
    mode = p.get("mode", "gray")
    info = []

    if mode == "gray":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    elif mode == "binary":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if p.get("otsu"):
            t, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            info.append(["Threshold Otsu", str(int(t))])
        else:
            _, bw = cv2.threshold(g, int(float(p.get("threshold", 128))), 255, cv2.THRESH_BINARY)
        out = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    elif mode == "negative":
        out = 255 - img
    elif mode in ("red", "green", "blue"):
        idx = {"blue": 0, "green": 1, "red": 2}[mode]
        out = np.zeros_like(img)
        out[..., idx] = img[..., idx]
    else:
        out = img
    return result(out, info=info)


# ----------------------------------------------------------------------
# 4. TRANSFORMATION  (rotate, scale, translate, flip)
# ----------------------------------------------------------------------
def m_transformation(img, p):
    if p.get("flip_h"):
        img = cv2.flip(img, 1)
    if p.get("flip_v"):
        img = cv2.flip(img, 0)

    h, w = img.shape[:2]
    angle = float(p.get("angle", 0))
    scale = float(p.get("scale", 100)) / 100.0
    tx = float(p.get("shift_x", 0)) / 100.0 * w
    ty = float(p.get("shift_y", 0)) / 100.0 * h

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = h * sin + w * cos
    nh = h * cos + w * sin
    M[0, 2] += nw / 2 - w / 2 + tx
    M[1, 2] += nh / 2 - h / 2 + ty

    f = min(1.0, OUT_MAX_SIDE / max(nw, nh))  # jaga ukuran output
    M = M * f
    nw, nh = max(1, int(nw * f)), max(1, int(nh * f))

    out = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    # mask area gambar asli, agar latar hitam hasil rotasi tidak ikut histogram
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    return result(out, mask=mask, info=[["Ukuran hasil", f"{nw} × {nh} px"]])


# ----------------------------------------------------------------------
# 5. EQUALIZATION  (global HE, per-channel RGB, CLAHE, contrast stretching)
# ----------------------------------------------------------------------
def m_equalization(img, p):
    mode = p.get("mode", "global")
    info = []

    if mode == "global":
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    elif mode == "rgb":
        out = cv2.merge([cv2.equalizeHist(c) for c in cv2.split(img)])
    elif mode == "clahe":
        clip = float(p.get("clip_limit", 2))
        tile = max(2, int(float(p.get("tile", 8))))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycc[..., 0] = clahe.apply(ycc[..., 0])
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    elif mode == "stretch":
        c = float(p.get("clip_pct", 1))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lo, hi = np.percentile(gray, [c, 100 - c])
        if hi - lo < 1:
            out = img.copy()
        else:
            out = clip8((img.astype(np.float32) - lo) * 255.0 / (hi - lo))
        info.append(["Rentang asal", f"{int(lo)} – {int(hi)}"])
    else:
        out = img

    amount = float(p.get("amount", 100)) / 100.0
    if amount < 1:
        out = cv2.addWeighted(img, 1 - amount, out, amount, 0)
    return result(out, info=info)


# ----------------------------------------------------------------------
# 6. COMPRESSION  (JPEG, WebP, PNG)
# ----------------------------------------------------------------------
def m_compression(img, p):
    fmt = p.get("format", "jpeg")
    q = int(float(p.get("quality", 50)))
    level = int(float(p.get("png_level", 6)))

    if fmt == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        mime, ext = "image/jpeg", "jpg"
    elif fmt == "webp":
        ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, q])
        mime, ext = "image/webp", "webp"
    else:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, level])
        mime, ext = "image/png", "png"
    if not ok:
        raise ValueError("Gagal meng-encode gambar")

    data = buf.tobytes()
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    raw = img.shape[0] * img.shape[1] * 3
    mse = float(np.mean((decoded.astype(np.float32) - img.astype(np.float32)) ** 2))
    psnr = "∞ (lossless)" if mse == 0 else f"{10 * math.log10(255 ** 2 / mse):.2f} dB"
    info = [
        ["Ukuran mentah", fmt_bytes(raw)],
        ["Ukuran hasil", fmt_bytes(len(data))],
        ["Rasio kompresi", f"{raw / len(data):.1f} : 1"],
        ["Hemat", f"{(1 - len(data) / raw) * 100:.1f}%"],
        ["PSNR", psnr],
    ]
    return result(decoded, info=info, enc=(data, mime, ext))


METHODS = {
    "adjustment": m_adjustment,
    "noise": m_noise,
    "filtering": m_filtering,
    "transformation": m_transformation,
    "equalization": m_equalization,
    "compression": m_compression,
}


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/process")
def api_process():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="Tidak ada gambar"), 400

    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="File bukan gambar yang valid"), 400
    img = limit_side(img, MAX_SIDE)

    method = request.form.get("method", "adjustment")
    fn = METHODS.get(method)
    if fn is None:
        return jsonify(error=f"Metode '{method}' tidak dikenal"), 400
    try:
        params = json.loads(request.form.get("params", "{}"))
    except json.JSONDecodeError:
        params = {}

    # gambar referensi (mis. gambar asli saat pipeline) untuk histogram "Original"
    ref = img
    rf = request.files.get("reference")
    if rf is not None:
        r = cv2.imdecode(np.frombuffer(rf.read(), np.uint8), cv2.IMREAD_COLOR)
        if r is not None:
            ref = limit_side(r, MAX_SIDE)

    try:
        out, mask, info, enc = fn(img, params)
        if enc is None:
            ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            enc = (buf.tobytes(), "image/png", "png")
        data, mime, ext = enc
    except Exception as e:  # noqa: BLE001
        return jsonify(error=f"Proses gagal: {e}"), 500

    return jsonify(
        image=f"data:{mime};base64," + base64.b64encode(data).decode(),
        ext=ext,
        width=int(out.shape[1]),
        height=int(out.shape[0]),
        bytes=len(data),
        hist_orig=histogram(ref),
        hist_edit=histogram(out, mask),
        info=info,
    )


if __name__ == "__main__":
    app.run(debug=True)
