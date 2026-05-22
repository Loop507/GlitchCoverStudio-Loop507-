import streamlit as st
import numpy as np
import io
import random
import math
from PIL import Image, ImageDraw, ImageChops, ImageFilter, ImageEnhance
import hashlib
import time
import librosa

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")
st.title("🎵 GlitchCover Studio by Loop507 [Free App]")
st.markdown("Crea una copertina glitch unica basata sul tuo brano. Carica un file audio e genera arte ispirata al suono.")

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def get_dimensions(format_type):
    if format_type == "Quadrato (1:1)": return (800, 800)
    elif format_type == "Verticale (9:16)": return (720, 1280)
    elif format_type == "Orizzontale (16:9)": return (1280, 720)
    return (800, 800)

# ─────────────────────────────────────────────
# ANALISI AUDIO
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def analyze_audio_librosa(audio_bytes: bytes):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True, duration=30)
        try:
            bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(bpm) if bpm is not None and not np.isnan(bpm) else 120.0
        except Exception:
            bpm = 120.0

        def safe(val, fallback):
            v = float(np.mean(val))
            return v if not np.isnan(v) else fallback

        rms                = safe(librosa.feature.rms(y=y), 0.05)
        spectral_centroid  = safe(librosa.feature.spectral_centroid(y=y, sr=sr), 2000.0)
        spectral_bandwidth = safe(librosa.feature.spectral_bandwidth(y=y, sr=sr), 1000.0)
        spectral_rolloff   = safe(librosa.feature.spectral_rolloff(y=y, sr=sr), 3000.0)
        zero_crossing_rate = safe(librosa.feature.zero_crossing_rate(y), 0.1)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.nan_to_num(np.mean(mfcc, axis=1), nan=0.0, posinf=0.0, neginf=0.0)

        try:
            y_h, y_p = librosa.effects.hpss(y)
            harmonic_energy   = safe(librosa.feature.rms(y=y_h), rms * 0.6)
            percussive_energy = safe(librosa.feature.rms(y=y_p), rms * 0.4)
        except Exception:
            harmonic_energy, percussive_energy = rms * 0.6, rms * 0.4

        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_strength = len(onset_frames) / len(y) * sr if len(y) > 0 else 1.0
        except Exception:
            onset_strength = 1.0

        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            tonal_centroid = int(np.argmax(np.mean(chroma, axis=1)))
        except Exception:
            tonal_centroid = 0

        dynamic_range = abs(spectral_bandwidth - spectral_centroid)
        feature_string = f"{bpm:.2f}_{rms:.4f}_{spectral_centroid:.0f}_{len(audio_bytes)}"
        hash_obj = hashlib.sha256((feature_string + str(audio_bytes[:1000])).encode()).hexdigest()
        hash_int = int(hash_obj[:8], 16)

        if rms > 0.08 and bpm > 120 and percussive_energy > 0.03:
            emotion = "Aggressive"
        elif rms > 0.05 and bpm > 110:
            emotion = "Energetico"
        elif rms < 0.02 and bpm < 70:
            emotion = "Ambient"
        elif rms < 0.03 and bpm < 90:
            emotion = "Calmo"
        elif harmonic_energy > percussive_energy * 2:
            emotion = "Melodico"
        elif percussive_energy > harmonic_energy * 1.5:
            emotion = "Ritmico"
        elif dynamic_range > 3000:
            emotion = "Dinamico"
        else:
            emotion = "Equilibrato"

        return {
            "bpm": bpm, "rms": rms,
            "spectral_centroid": spectral_centroid, "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff, "zero_crossing_rate": zero_crossing_rate,
            "dynamic_range": dynamic_range, "harmonic_energy": harmonic_energy,
            "percussive_energy": percussive_energy, "onset_strength": onset_strength,
            "tonal_centroid": tonal_centroid, "mfcc_features": mfcc_mean,
            "emotion": emotion, "file_hash": hash_int,
            "file_size": len(audio_bytes), "audio_signature": hash_obj[:16]
        }
    except Exception as e:
        st.error(f"Errore nell'analisi audio: {str(e)}")
        return None


# ═══════════════════════════════════════════════════════════════
#  MOTORE VISIVO V3 — PALETTE + BACKGROUND + OVERLAY + EFFETTI
# ═══════════════════════════════════════════════════════════════

# ── 1. PALETTE PROCEDURALE ──────────────────────────────────────

def generate_palette(features, seed):
    """Genera una palette di 5-7 colori unica per ogni seed/brano."""
    rng = random.Random(seed)

    # Hue base derivato da più features audio combinate
    hue_base = (
        (features['tonal_centroid'] / 12.0) * 360
        + features['spectral_centroid'] / 100.0
        + features['bpm'] / 3.0
    ) % 360

    emotion_hue_shift = {
        "Aggressive": 0,    "Energetico": 30,  "Ambient": 180,
        "Calmo": 200,       "Melodico": 270,   "Ritmico": 45,
        "Dinamico": 90,     "Equilibrato": 120
    }
    hue_base = (hue_base + emotion_hue_shift.get(features['emotion'], 0)) % 360

    scheme = rng.choice(['complementare', 'analogo', 'triadico', 'split', 'tetradico'])
    saturation = min(1.0, max(0.4, features['rms'] * 6 + 0.3))
    lightness_base = min(0.7, max(0.25, 1.0 - features['rms'] * 4))

    def hsl_to_rgb(h, s, l):
        h /= 360
        if s == 0:
            v = int(l * 255)
            return (v, v, v)
        def f(n):
            k = (n + h * 12) % 12
            a = s * min(l, 1 - l)
            return l - a * max(-1, min(k - 3, 9 - k, 1))
        return (int(f(0)*255), int(f(8)*255), int(f(4)*255))

    hues = []
    if scheme == 'complementare':
        hues = [hue_base, (hue_base+180)%360, (hue_base+30)%360,
                (hue_base+210)%360, (hue_base+15)%360]
    elif scheme == 'analogo':
        hues = [hue_base, (hue_base+30)%360, (hue_base+60)%360,
                (hue_base-30)%360, (hue_base+15)%360]
    elif scheme == 'triadico':
        hues = [hue_base, (hue_base+120)%360, (hue_base+240)%360,
                (hue_base+60)%360, (hue_base+180)%360]
    elif scheme == 'split':
        hues = [hue_base, (hue_base+150)%360, (hue_base+210)%360,
                (hue_base+30)%360, (hue_base+180)%360]
    else:  # tetradico
        hues = [hue_base, (hue_base+90)%360, (hue_base+180)%360,
                (hue_base+270)%360, (hue_base+45)%360]

    palette = []
    for i, h in enumerate(hues):
        l_var = lightness_base + rng.uniform(-0.1, 0.15)
        s_var = saturation + rng.uniform(-0.1, 0.1)
        palette.append(hsl_to_rgb(h, min(1,max(0,s_var)), min(0.9,max(0.1,l_var))))

    # Aggiungi sempre un colore quasi-nero e uno quasi-bianco per contrasto
    palette.append((rng.randint(0,30), rng.randint(0,30), rng.randint(0,30)))
    palette.append((rng.randint(220,255), rng.randint(220,255), rng.randint(220,255)))
    return palette


# ── 2. BACKGROUND (10 tipi) ─────────────────────────────────────

def bg_gradient_diagonal(size, palette, seed, features):
    rng = np.random.default_rng(seed)
    w, h = size
    c1 = np.array(palette[0], dtype=np.float32)
    c2 = np.array(palette[1], dtype=np.float32)
    c3 = np.array(palette[2 % len(palette)], dtype=np.float32)
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    xv, yv = np.meshgrid(xs, ys)
    t = (xv + yv) / 2
    t2 = np.sin(t * math.pi * (1 + features['bpm'] / 120)) * 0.5 + 0.5
    arr = (c1[None,None] * (1-t[:,:,None]) + c2[None,None] * t[:,:,None]).astype(np.uint8)
    mask = t2 > 0.7
    arr[mask] = np.clip(arr[mask] * 0.5 + c3[None] * 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGB')

def bg_plasma(size, palette, seed, features):
    w, h = size
    scale = max(0.003, features['spectral_centroid'] / 200000)
    freq = features['bpm'] / 60.0
    xs = np.linspace(0, w * scale, w)
    ys = np.linspace(0, h * scale, h)
    xv, yv = np.meshgrid(xs, ys)
    plasma = (
        np.sin(xv * freq) +
        np.sin(yv * freq * 0.7) +
        np.sin((xv + yv) * freq * 0.5) +
        np.sin(np.sqrt(xv**2 + yv**2) * freq)
    )
    plasma = (plasma - plasma.min()) / (plasma.max() - plasma.min() + 1e-8)
    n = len(palette)
    idx = (plasma * (n - 1)).astype(int)
    arr = np.array(palette, dtype=np.uint8)[idx]
    return Image.fromarray(arr, 'RGB')

def bg_scanlines(size, palette, seed, features):
    rng = random.Random(seed)
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    line_height = max(2, int(features['spectral_bandwidth'] / 200))
    for y in range(0, h, line_height):
        col = palette[rng.randint(0, len(palette)-1)]
        arr[y:y+line_height, :] = col
        if rng.random() < features['zero_crossing_rate'] * 3:
            shift = rng.randint(-80, 80)
            if shift > 0: arr[y:y+line_height, shift:] = arr[y:y+line_height, :-shift]
            elif shift < 0: arr[y:y+line_height, :shift] = arr[y:y+line_height, -shift:]
    return Image.fromarray(arr, 'RGB')

def bg_voronoi(size, palette, seed, features):
    rng = random.Random(seed)
    w, h = size
    n_points = max(5, min(30, int(features['onset_strength'] * 3) + 8))
    points = [(rng.randint(0, w), rng.randint(0, h)) for _ in range(n_points)]
    colors = [palette[i % len(palette)] for i in range(n_points)]
    xs = np.arange(w); ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)
    dists = np.stack([
        (xv - px)**2 + (yv - py)**2 for px, py in points
    ], axis=0)
    nearest = np.argmin(dists, axis=0)
    arr = np.array(colors, dtype=np.uint8)[nearest]
    return Image.fromarray(arr, 'RGB')

def bg_concentric_shapes(size, palette, seed, features):
    rng = random.Random(seed)
    w, h = size
    img = Image.new('RGB', size, palette[0])
    draw = ImageDraw.Draw(img)
    shape = rng.choice(['ellipse', 'rect', 'diamond'])
    step = max(10, int(features['spectral_bandwidth'] / 80))
    cx, cy = w // 2, h // 2
    max_r = max(w, h)
    for r in range(max_r, 0, -step):
        col = palette[r // step % len(palette)]
        if shape == 'ellipse':
            draw.ellipse([cx-r, cy-int(r*0.7), cx+r, cy+int(r*0.7)], outline=col, width=max(2, step//3))
        elif shape == 'rect':
            draw.rectangle([cx-r, cy-int(r*0.75), cx+r, cy+int(r*0.75)], outline=col, width=max(2, step//3))
        else:  # diamond
            pts = [(cx, cy-r), (cx+r, cy), (cx, cy+r), (cx-r, cy)]
            draw.polygon(pts, outline=col)
    return img

def bg_interference_waves(size, palette, seed, features):
    w, h = size
    rng = random.Random(seed)
    f1 = features['bpm'] / 120.0
    f2 = features['spectral_centroid'] / 5000.0
    xs = np.linspace(0, 2*math.pi, w)
    ys = np.linspace(0, 2*math.pi, h)
    xv, yv = np.meshgrid(xs, ys)
    angle = rng.uniform(0, 2*math.pi)
    wave = (
        np.sin(xv * f1 + angle) *
        np.cos(yv * f2) +
        np.sin(np.sqrt((xv-math.pi)**2 + (yv-math.pi)**2) * f1 * 0.5)
    )
    wave = (wave - wave.min()) / (wave.max() - wave.min() + 1e-8)
    n = len(palette)
    idx = (wave * (n-1)).astype(int)
    arr = np.array(palette, dtype=np.uint8)[idx]
    return Image.fromarray(arr, 'RGB')

def bg_tiled_blocks(size, palette, seed, features):
    rng = random.Random(seed)
    w, h = size
    tile_size = max(20, int(features['spectral_bandwidth'] / 50) * 10)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for ty in range(0, h, tile_size):
        for tx in range(0, w, tile_size):
            col = palette[rng.randint(0, len(palette)-1)]
            if rng.random() < 0.3:
                arr[ty:ty+tile_size, tx:tx+tile_size] = col
            else:
                sub_size = tile_size // 2
                for sy in range(2):
                    for sx in range(2):
                        sub_col = palette[rng.randint(0, len(palette)-1)]
                        y0 = ty + sy*sub_size; x0 = tx + sx*sub_size
                        arr[y0:y0+sub_size, x0:x0+sub_size] = sub_col
    return Image.fromarray(arr, 'RGB')

def bg_diagonal_stripes(size, palette, seed, features):
    rng = random.Random(seed)
    w, h = size
    stripe_w = max(8, int(features['spectral_bandwidth'] / 150))
    angle = rng.choice([30, 45, 60, -30, -45])
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    xs = np.arange(w); ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)
    proj = xv * math.cos(math.radians(angle)) + yv * math.sin(math.radians(angle))
    idx = (proj / stripe_w).astype(int) % len(palette)
    arr = np.array(palette, dtype=np.uint8)[idx]
    return Image.fromarray(arr, 'RGB')

def bg_fractal_noise(size, palette, seed, features):
    """Simplex-like noise multi-ottava con NumPy puro."""
    rng = np.random.default_rng(seed)
    w, h = size
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    frequency = max(0.002, features['spectral_centroid'] / 500000)
    for _ in range(5):
        # Genera noise a questa ottava con interpolazione bilineare
        sh = max(2, int(h * frequency)); sw = max(2, int(w * frequency))
        raw = rng.random((sh, sw)).astype(np.float32)
        # Upsample con resize PIL
        layer = np.array(Image.fromarray((raw * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        noise += layer * amplitude
        amplitude *= 0.5
        frequency *= 2
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    n = len(palette)
    idx = (noise * (n-1)).astype(int)
    arr = np.array(palette, dtype=np.uint8)[idx]
    return Image.fromarray(arr, 'RGB')

def bg_radial_burst(size, palette, seed, features):
    rng = random.Random(seed)
    w, h = size
    cx = w * rng.uniform(0.3, 0.7)
    cy = h * rng.uniform(0.3, 0.7)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    angle = np.arctan2(yv - cy, xv - cx)
    dist  = np.sqrt((xv - cx)**2 + (yv - cy)**2)
    n_rays = max(4, int(features['bpm'] / 15))
    ray_idx = ((angle + math.pi) / (2 * math.pi) * n_rays).astype(int) % len(palette)
    dist_idx = (dist / max(w, h) * len(palette)).astype(int) % len(palette)
    combined = (ray_idx + dist_idx) % len(palette)
    arr = np.array(palette, dtype=np.uint8)[combined]
    return Image.fromarray(arr, 'RGB')

def create_base_image(size, palette, seed, features):
    """Sceglie uno dei 10 background in modo dipendente da seed + audio."""
    rng = random.Random(seed)
    # Peso i background in base alle features per più varietà
    bg_funcs = [
        bg_gradient_diagonal, bg_plasma, bg_scanlines, bg_voronoi,
        bg_concentric_shapes, bg_interference_waves, bg_tiled_blocks,
        bg_diagonal_stripes, bg_fractal_noise, bg_radial_burst
    ]
    # Il tipo cambia con il seed (regen) E con le features del brano
    bg_idx = (seed // 7 + int(features['spectral_centroid']) // 100) % len(bg_funcs)
    # Con probabilità 0.3 scegliamo random puro per più sorprese
    if rng.random() < 0.3:
        bg_idx = rng.randint(0, len(bg_funcs)-1)
    return bg_funcs[bg_idx](size, palette, seed, features)


# ── 3. OVERLAY (aggiuntivi sopra il background) ─────────────────

def overlay_glitch_bars(img, features, seed, palette):
    rng = random.Random(seed + 9000)
    arr = np.array(img)
    h, w = arr.shape[:2]
    n_bars = rng.randint(3, 12)
    for _ in range(n_bars):
        y = rng.randint(0, h-1)
        bar_h = rng.randint(1, max(2, int(features['rms'] * 60)))
        col = list(palette[rng.randint(0, len(palette)-1)])
        alpha = rng.uniform(0.3, 0.9)
        region = arr[y:min(y+bar_h, h), :].astype(np.float32)
        arr[y:min(y+bar_h, h), :] = np.clip(region * (1-alpha) + np.array(col) * alpha, 0, 255).astype(np.uint8)
        # shift orizzontale del bar
        shift = rng.randint(-120, 120)
        if shift > 0 and shift < w: arr[y:min(y+bar_h,h), shift:] = arr[y:min(y+bar_h,h), :-shift].copy()
        elif shift < 0 and abs(shift) < w: arr[y:min(y+bar_h,h), :shift] = arr[y:min(y+bar_h,h), -shift:].copy()
    return Image.fromarray(arr, 'RGB')

def overlay_frequency_bars(img, features, seed, palette):
    """Barre di frequenza simulate (tipo spettro audio) nella parte bassa."""
    rng = random.Random(seed + 8000)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    n_bars = rng.randint(20, 60)
    bar_w = w // n_bars
    max_bar_h = int(h * rng.uniform(0.15, 0.4))
    for i in range(n_bars):
        # altezza barra guidata dalle mfcc e dal seed
        mfcc_idx = i % len(features['mfcc_features'])
        mfcc_val = abs(features['mfcc_features'][mfcc_idx])
        bar_h = int(min(max_bar_h, max(4, mfcc_val * rng.uniform(0.8, 2.5))))
        x0 = i * bar_w
        y0 = h - bar_h
        col = palette[i % len(palette)]
        draw.rectangle([x0, y0, x0 + bar_w - 1, h], fill=col)
    return img

def overlay_grid(img, features, seed, palette):
    rng = random.Random(seed + 7000)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    grid_step = max(20, int(features['spectral_bandwidth'] / 60))
    col = palette[rng.randint(0, len(palette)-1)]
    alpha_img = Image.new('RGBA', img.size, (0,0,0,0))
    d = ImageDraw.Draw(alpha_img)
    line_alpha = rng.randint(40, 120)
    for x in range(0, w, grid_step):
        d.line([(x,0),(x,h)], fill=(*col, line_alpha), width=1)
    for y in range(0, h, grid_step):
        d.line([(0,y),(w,y)], fill=(*col, line_alpha), width=1)
    img_rgba = img.convert('RGBA')
    img_rgba = Image.alpha_composite(img_rgba, alpha_img)
    return img_rgba.convert('RGB')

def overlay_glitch_text(img, features, seed, palette):
    """Testo glitch stilizzato con dati audio."""
    rng = random.Random(seed + 6000)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    texts = [
        f"BPM:{features['bpm']:.0f}",
        f"RMS:{features['rms']:.3f}",
        f"{features['emotion'].upper()}",
        f"SC:{features['spectral_centroid']:.0f}",
        f"ZCR:{features['zero_crossing_rate']:.2f}",
        "GLITCH", "ERROR", "LOOP507",
        f"#{features['audio_signature'][:6]}",
    ]
    n_texts = rng.randint(3, 8)
    for _ in range(n_texts):
        text = rng.choice(texts)
        x = rng.randint(0, max(1, w - 100))
        y = rng.randint(0, max(1, h - 30))
        size = rng.randint(10, 28)
        col = palette[rng.randint(0, len(palette)-1)]
        # Effetto glitch: duplica con offset e colore diverso
        if rng.random() < 0.6:
            col2 = palette[rng.randint(0, len(palette)-1)]
            off = rng.randint(2, 8)
            draw.text((x+off, y), text, fill=col2)
        draw.text((x, y), text, fill=col)
    return img

def overlay_broken_geometry(img, features, seed, palette):
    rng = random.Random(seed + 5000)
    img_rgba = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    n_shapes = rng.randint(2, 8)
    for _ in range(n_shapes):
        col = palette[rng.randint(0, len(palette)-1)]
        alpha = rng.randint(60, 180)
        shape_type = rng.choice(['triangle', 'line_group', 'broken_rect'])
        if shape_type == 'triangle':
            pts = [(rng.randint(0,w), rng.randint(0,h)) for _ in range(3)]
            draw.polygon(pts, outline=(*col, alpha), fill=(*col, alpha//4))
        elif shape_type == 'line_group':
            for _ in range(rng.randint(3, 8)):
                x1,y1 = rng.randint(0,w), rng.randint(0,h)
                x2,y2 = x1+rng.randint(-200,200), y1+rng.randint(-200,200)
                draw.line([(x1,y1),(x2,y2)], fill=(*col, alpha), width=rng.randint(1,4))
        else:
            x0,y0 = rng.randint(0,w-10), rng.randint(0,h-10)
            x1,y1 = x0+rng.randint(20,200), y0+rng.randint(20,150)
            draw.rectangle([x0,y0,x1,y1], outline=(*col, alpha), width=rng.randint(1,3))
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    return img_rgba.convert('RGB')

def overlay_databend_stripes(img, features, seed):
    """Simula databending copiando strisce da posizioni random."""
    rng = random.Random(seed + 4000)
    arr = np.array(img)
    h, w = arr.shape[:2]
    n = rng.randint(2, 8)
    for _ in range(n):
        src_y = rng.randint(0, h-1)
        dst_y = rng.randint(0, h-1)
        stripe_h = rng.randint(1, max(2, int(features['percussive_energy'] * 80)))
        src_end = min(src_y + stripe_h, h)
        dst_end = min(dst_y + stripe_h, h)
        copy_h = min(src_end - src_y, dst_end - dst_y)
        if copy_h > 0:
            src_slice = arr[src_y:src_y+copy_h].copy()
            # channel swap casuale
            ch_perm = rng.choice([[0,1,2],[2,0,1],[1,2,0],[0,2,1]])
            src_slice = src_slice[:, :, ch_perm]
            arr[dst_y:dst_y+copy_h] = src_slice
    return Image.fromarray(arr, 'RGB')

def apply_overlays(img, features, seed, palette, regen_seed):
    """Applica un sottoinsieme casuale di overlay, diverso ad ogni regen."""
    rng = random.Random(regen_seed)
    all_overlays = [
        lambda i: overlay_glitch_bars(i, features, seed, palette),
        lambda i: overlay_frequency_bars(i, features, seed, palette),
        lambda i: overlay_grid(i, features, seed, palette),
        lambda i: overlay_glitch_text(i, features, seed, palette),
        lambda i: overlay_broken_geometry(i, features, seed, palette),
        lambda i: overlay_databend_stripes(i, features, seed),
    ]
    # Sempre almeno 2, al massimo 4
    n = rng.randint(2, 4)
    chosen = rng.sample(all_overlays, n)
    for fn in chosen:
        img = fn(img)
    return img


# ── 4. EFFETTI GLITCH (invariati + migliorati) ───────────────────

def apply_pixel_sorting(img, features, seed):
    rng = random.Random(seed)
    width, height = img.size
    pixels = np.array(img)
    sort_intensity = max(0.1, min(0.9, features['onset_strength']/10 + features['percussive_energy']*5))
    if rng.random() < 0.5:
        for y in range(0, height, rng.randint(2,8)):
            if rng.random() < sort_intensity:
                row = pixels[y].copy()
                idx = np.argsort(np.mean(row, axis=1) if rng.random()<0.5 else row[:,rng.randint(0,2)])
                pixels[y] = row[idx]
    else:
        for x in range(0, width, rng.randint(2,8)):
            if rng.random() < sort_intensity:
                col = pixels[:,x].copy()
                idx = np.argsort(np.mean(col, axis=1) if rng.random()<0.5 else col[:,rng.randint(0,2)])
                pixels[:,x] = col[idx]
    return Image.fromarray(pixels)

def apply_digital_corruption(img, features, seed):
    rng = random.Random(seed)
    np.random.seed(seed % 2147483647)
    width, height = img.size
    pixels = np.array(img)
    corruption_intensity = max(0.1, min(0.8, features['zero_crossing_rate']*5 + features['rms']*3))
    num_corruptions = int(corruption_intensity * 50)
    for _ in range(num_corruptions):
        ct = rng.randint(0,4)
        if ct == 0:
            bw = rng.randint(10, min(100,width//4,height//4)); bh = rng.randint(10, min(100,width//4,height//4))
            x = rng.randint(0, max(1,width-bw)); y = rng.randint(0, max(1,height-bh))
            pixels[y:min(y+bh,height), x:min(x+bw,width)] = 255 - pixels[y:min(y+bh,height), x:min(x+bw,width)]
        elif ct == 1:
            y = rng.randint(0, height-1); lh = rng.randint(1, min(5,height-y)); shift = rng.randint(-50,50)
            if shift > 0 and shift < width: pixels[y:y+lh, shift:] = pixels[y:y+lh, :-shift]
            elif shift < 0 and abs(shift) < width: pixels[y:y+lh, :shift] = pixels[y:y+lh, -shift:]
        elif ct == 2:
            bw = rng.randint(5, min(30,width//8)); bh = rng.randint(5, min(30,height//8))
            x = rng.randint(0, max(1,width-bw)); y = rng.randint(0, max(1,height-bh))
            pixels[y:min(y+bh,height), x:min(x+bw,width)] = np.random.randint(0,255,(min(bh,height-y),min(bw,width-x),3))
        elif ct == 3:
            if rng.random() < 0.5:
                sh = rng.randint(1,min(8,height//10)); y = rng.randint(0,max(1,height-sh))
                ty2 = rng.randint(0,max(1,height-sh)); pixels[ty2:ty2+sh] = pixels[y:y+sh]
            else:
                sw = rng.randint(1,min(8,width//10)); x = rng.randint(0,max(1,width-sw))
                tx2 = rng.randint(0,max(1,width-sw)); pixels[:,tx2:tx2+sw] = pixels[:,x:x+sw]
        elif ct == 4:
            offset = rng.randint(5,min(30,width//10)); ch = rng.randint(0,2)
            d = rng.choice([(offset,0),(-offset,0),(0,offset),(0,-offset)])
            if d[0]>0 and d[0]<width: pixels[:,offset:,ch] = pixels[:,:-offset,ch]
            elif d[0]<0 and abs(d[0])<width: pixels[:,:offset,ch] = pixels[:,-offset:,ch]
            elif d[1]>0 and d[1]<height: pixels[offset:,:,ch] = pixels[:-offset,:,ch]
            elif d[1]<0 and abs(d[1])<height: pixels[:offset,:,ch] = pixels[-offset:,:,ch]
    return Image.fromarray(pixels)

def apply_advanced_chromatic_aberration(img, features, seed):
    rng = random.Random(seed)
    base_offset = max(3, min(30, int(features['spectral_bandwidth']/100)))
    rms_mul = max(2, min(15, int(features['rms']*150)))
    r, g, b = img.split()
    r = ImageChops.offset(r, base_offset+rng.randint(-rms_mul,rms_mul), rng.randint(-5,5))
    g = ImageChops.offset(g, rng.randint(-base_offset//2,base_offset//2), rng.randint(-3,3))
    b = ImageChops.offset(b, -base_offset+rng.randint(-rms_mul//2,rms_mul//2), rng.randint(-5,5))
    return Image.merge('RGB', (r, g, b))

def apply_black_and_white(img, intensity=0.5):
    return Image.blend(img, img.convert('L').convert('RGB'), intensity)

def apply_distorted_lines(img, features, seed):
    rng = random.Random(seed)
    width, height = img.size
    draw = ImageDraw.Draw(img)
    num_lines = int(features['percussive_energy'] * 200) + 5
    for _ in range(num_lines):
        color = (rng.randint(0,255), rng.randint(0,255), rng.randint(0,255))
        sx = rng.randint(0, width); sy = rng.randint(0, height)
        points = []
        for i in range(0, width, 10):
            y_off = int(np.sin(i * 0.01 + seed) * features['zero_crossing_rate'] * 100)
            points.append((sx+i, sy+y_off))
        if len(points) > 1:
            draw.line(points, fill=color, width=rng.randint(1,5))
    return img

def apply_vhs_effect(img, features, seed):
    rng = random.Random(seed)
    width, height = img.size
    pixels = np.array(img)
    noise_intensity = max(0.1, min(0.5, features['rms'] * 5))
    for y in range(height):
        if rng.random() < noise_intensity:
            nw = rng.randint(5,50); ns = rng.randint(0, max(1,width-nw))
            pixels[y, ns:ns+nw] = [rng.randint(0,255) for _ in range(3)]
    r, g, b = Image.fromarray(pixels).split()
    r = ImageChops.offset(r, 2, 0); b = ImageChops.offset(b, -2, 0)
    return ImageEnhance.Color(Image.merge('RGB', (r, g, b))).enhance(0.7)

def apply_posterize(img, features, seed):
    rng = random.Random(seed)
    bits = rng.randint(2, 5)
    from PIL import ImageOps
    return ImageOps.posterize(img, bits)

def apply_random_distortion(img, features, seed):
    rng = random.Random(seed)
    width, height = img.size
    pixels = np.array(img)
    dt = rng.randint(0,3)
    if dt == 0:
        buf = io.BytesIO(); img.save(buf, format='JPEG', quality=rng.randint(5,40)); buf.seek(0); return Image.open(buf)
    elif dt == 1: return img.quantize(colors=rng.randint(4,48)).convert('RGB')
    elif dt == 2:
        scale = rng.uniform(0.2,0.7)
        return img.resize((max(1,int(width*scale)),max(1,int(height*scale))), Image.NEAREST).resize((width,height), Image.NEAREST)
    elif dt == 3:
        for y in range(height):
            wo = int(np.sin(y*0.1+seed)*20)
            if wo>0 and wo<width: pixels[y,wo:] = pixels[y,:-wo]
            elif wo<0 and abs(wo)<width: pixels[y,:wo] = pixels[y,-wo:]
        return Image.fromarray(pixels)
    return img


# ── 5. PIPELINE PRINCIPALE ───────────────────────────────────────

def generate_glitch_image(features, seed, regen_count, size=(800,800), glitch_controls=None):
    """
    seed       = hash del file (stabile per il brano)
    regen_count = cambia ad ogni rigenerazione → varia tutto
    """
    regen_seed = seed ^ (regen_count * 2654435761)  # hash mixing robusto

    rng = random.Random(regen_seed)
    np.random.seed(regen_seed % 2147483647)

    if glitch_controls is None:
        glitch_controls = {
            'pixel_sorting':True,'digital_corruption':True,'chromatic_aberration':True,
            'black_and_white':False,'distorted_lines':False,'vhs_effect':False,
            'posterize':False,'bw_intensity':0.5
        }

    # 1. Palette procedurale (varia con regen_seed)
    palette = generate_palette(features, regen_seed)

    # 2. Background (varia con regen_seed + features)
    img = create_base_image(size, palette, regen_seed, features)

    # 3. Overlay grafici (sottoinsieme random ad ogni regen)
    img = apply_overlays(img, features, seed, palette, regen_seed)

    # 4. Effetti glitch — ordine RANDOMIZZATO ad ogni regen
    all_effects = []
    if glitch_controls.get('pixel_sorting', True):
        all_effects.append(lambda i, s=regen_seed: apply_pixel_sorting(i, features, s))
    if glitch_controls.get('digital_corruption', True):
        all_effects.append(lambda i, s=regen_seed+100: apply_digital_corruption(i, features, s))
    if glitch_controls.get('chromatic_aberration', True):
        all_effects.append(lambda i, s=regen_seed+200: apply_advanced_chromatic_aberration(i, features, s))
    if glitch_controls.get('distorted_lines', False):
        all_effects.append(lambda i, s=regen_seed+400: apply_distorted_lines(i, features, s))
    if glitch_controls.get('vhs_effect', False):
        all_effects.append(lambda i, s=regen_seed+500: apply_vhs_effect(i, features, s))
    if glitch_controls.get('posterize', False):
        all_effects.append(lambda i, s=regen_seed+600: apply_posterize(i, features, s))

    # Ordine casuale degli effetti
    rng.shuffle(all_effects)

    # Distorsione random SEMPRE (varia con regen)
    all_effects.insert(rng.randint(0, max(0,len(all_effects))),
                       lambda i, s=regen_seed+300: apply_random_distortion(i, features, s))

    for fn in all_effects:
        img = fn(img)

    # 5. Finalizzazione emozione
    if features['emotion'] == 'Aggressive':
        img = apply_digital_corruption(img, features, regen_seed+5000)
        img = apply_pixel_sorting(img, features, regen_seed+6000)
    elif features['emotion'] == 'Energetico':
        img = apply_advanced_chromatic_aberration(img, features, regen_seed+7000)
        img = apply_random_distortion(img, features, regen_seed+8000)
    elif features['emotion'] == 'Ambient':
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.0)))
        img = apply_advanced_chromatic_aberration(img, features, regen_seed+9000)
    elif features['emotion'] == 'Ritmico':
        img = apply_pixel_sorting(img, features, regen_seed+10000)

    if glitch_controls.get('black_and_white', False):
        img = apply_black_and_white(img, glitch_controls.get('bw_intensity', 0.5))

    # 6. Post-processing
    if features['harmonic_energy'] > 0.03:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(1.1, 1.6))
    if features['rms'] > 0.04:
        img = ImageEnhance.Color(img).enhance(rng.uniform(1.1, 2.0))
    if rng.random() < 0.4:
        img = ImageEnhance.Sharpness(img).enhance(rng.uniform(1.5, 3.0))

    descrizione = [
        f"🎵 BPM: {features['bpm']:.1f} → Velocità pattern e corruzioni",
        f"🔊 RMS: {features['rms']:.3f} → Intensità effetti glitch",
        f"🎼 Centro spettrale: {features['spectral_centroid']:.0f} Hz → Tipo di distorsione",
        f"⚡ Energia percussiva: {features['percussive_energy']:.3f} → Pixel sorting",
        f"🎹 Energia armonica: {features['harmonic_energy']:.3f} → Aberrazione cromatica",
        f"🌊 Zero crossing: {features['zero_crossing_rate']:.3f} → Corruzioni digitali",
        f"😊 Emozione: {features['emotion']} → Palette e stile",
        f"📊 Onset strength: {features['onset_strength']:.1f} → Complessità overlay",
    ]
    return img, descrizione


# ─────────────────────────────────────────────
# REPORT TESTUALE STILIZZATO
# ─────────────────────────────────────────────

def generate_report(features, audio_filename, regen_count, glitch_controls, glitch_intensity):
    emotion_descriptions = {
        "Aggressive": "brutal e disturbante, con corruzioni digitali intense e pixel sorting estremo.",
        "Energetico": "vibrante e dinamico, con aberrazione cromatica vivida e colori esplosivi.",
        "Ambient": "etereo e sospeso, con glitch morbidi e sfumature cromatiche rarefatte.",
        "Calmo": "meditativo e fluido, con distorsioni gentili che si dissolvono nello spazio visivo.",
        "Melodico": "armonico e ipnotico, con pattern di distorsione che seguono le curve melodiche.",
        "Ritmico": "pulsante e geometrico, con corruzioni sincronizzate al battito.",
        "Dinamico": "contrastato e imprevedibile, dove silenzi ed esplosioni sonore si alternano.",
        "Equilibrato": "bilanciato e stratificato, con un mix di effetti multidimensionali."
    }
    bpm_feel = "lento e pesante" if features['bpm']<80 else ("moderato" if features['bpm']<120 else ("veloce" if features['bpm']<160 else "frenetico"))
    rms_feel = "sussurrato" if features['rms']<0.02 else ("delicato" if features['rms']<0.05 else ("potente" if features['rms']<0.08 else "esplosivo"))
    active_effects = [k.replace("_"," ") for k,v in glitch_controls.items() if isinstance(v,bool) and v and k!='black_and_white']
    effects_str = ", ".join(active_effects) if active_effects else "standard"
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    dominant_note = notes[features['tonal_centroid']%12]
    track_name = audio_filename.rsplit('.',1)[0]

    return f"""╔══════════════════════════════════════════════════════════════╗
║         🎵 GLITCHCOVER STUDIO — REPORT AUDIO/VISIVO          ║
║                      by Loop507                              ║
╚══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE:      {audio_filename}
VARIANTE:  #{regen_count}
SIGNATURE: {features['audio_signature']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IDENTITA SONORA
  Emozione rilevata   →  {features['emotion'].upper()}
  Descrizione visiva  →  Un artwork {emotion_descriptions.get(features['emotion'], "unico.")}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATI AUDIO ESTRATTI (librosa)

  BPM                  →  {features['bpm']:.1f}   [{bpm_feel}]
  RMS Energy           →  {features['rms']:.4f}  [{rms_feel}]
  Centroide Spettrale  →  {features['spectral_centroid']:.0f} Hz
  Banda Spettrale      →  {features['spectral_bandwidth']:.0f} Hz
  Spectral Rolloff     →  {features['spectral_rolloff']:.0f} Hz
  Zero Crossing Rate   →  {features['zero_crossing_rate']:.4f}
  Energia Percussiva   →  {features['percussive_energy']:.4f}
  Energia Armonica     →  {features['harmonic_energy']:.4f}
  Onset Strength       →  {features['onset_strength']:.2f}
  Tonal Centroid       →  {features['tonal_centroid']}  (nota dominante: {dominant_note})
  Dimensione file      →  {features['file_size']/1024:.1f} KB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PARAMETRI DI GENERAZIONE

  Intensita Glitch     →  {glitch_intensity:.1f}x
  Effetti applicati    →  {effects_str}
  B&N                  →  {"ON (" + str(glitch_controls.get('bw_intensity',0)) + ")" if glitch_controls.get('black_and_white') else "OFF"}
  VHS                  →  {"ON" if glitch_controls.get('vhs_effect') else "OFF"}
  Posterize            →  {"ON" if glitch_controls.get('posterize') else "OFF"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DESCRIZIONE PER SOCIAL / YOUTUBE

🎵 "{track_name}" — Artwork generativo · GlitchCover Studio

Questa cover e nata dall'analisi algoritmica del brano.
Ogni pixel riflette un dato sonoro reale:

→ Il BPM ({features['bpm']:.0f}) ha guidato la velocita dei pattern glitch
→ L'energia RMS ({features['rms']:.3f}) ha determinato l'intensita delle corruzioni
→ Il centroide spettrale ({features['spectral_centroid']:.0f} Hz) ha definito la distorsione
→ L'energia percussiva ({features['percussive_energy']:.3f}) ha generato il pixel sorting
→ La nota dominante ({dominant_note}) ha influenzato le sfumature cromatiche
→ L'emozione rilevata — {features['emotion']} — ha scelto palette, overlay e stile

🤖 Generato da #GlitchCoverStudio · Powered by #Loop507
🎨 Arte algoritmica su analisi audio reale · No AI generativa

#glitchart #coverart #algorithmicart #musiccover #audiovisual
#glitch #digitalart #{features['emotion'].lower()} #bpm{int(features['bpm'])}
#loop507 #generativeart #musicproduction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loop507 Studio · Generato il: {time.strftime('%Y-%m-%d %H:%M')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

for key, default in [
    ('regen_count', 0), ('img_bytes', None), ('img_format_saved', 'PNG'),
    ('img_emotion', ''), ('report_text', None), ('audio_filename', ''),
    ('descrizione', []), ('features_saved', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3, WAV, OGG)", type=["mp3","wav","ogg"])

col1, col2, col3 = st.columns(3)
with col1: img_format = st.selectbox("Formato immagine:", ["PNG","JPEG"])
with col2: aspect_ratio = st.selectbox("Formato dimensioni:", ["Quadrato (1:1)","Verticale (9:16)","Orizzontale (16:9)"])
with col3: show_analysis = st.checkbox("Mostra analisi avanzata", value=True)

st.sidebar.markdown("### 🎛️ Controlli Glitch")
glitch_intensity = st.sidebar.slider("Intensita Glitch", 0.1, 2.0, 1.0, 0.1)

st.sidebar.markdown("#### Effetti Base")
use_pixel_sorting      = st.sidebar.checkbox("Pixel Sorting", value=True)
use_digital_corruption = st.sidebar.checkbox("Corruzioni Digitali", value=True)
use_chromatic          = st.sidebar.checkbox("Aberrazione Cromatica", value=True)

st.sidebar.markdown("#### Effetti Extra")
use_bw = st.sidebar.checkbox("Bianco e Nero", value=False)
bw_intensity = st.sidebar.slider("Intensita B&N", 0.1, 1.0, 0.5, 0.1) if use_bw else 0.5
use_distorted_lines = st.sidebar.checkbox("Linee Distorte", value=False)
use_vhs    = st.sidebar.checkbox("Effetto VHS", value=False)
use_poster = st.sidebar.checkbox("Posterize", value=False)

glitch_controls = {
    'pixel_sorting': use_pixel_sorting, 'digital_corruption': use_digital_corruption,
    'chromatic_aberration': use_chromatic, 'black_and_white': use_bw,
    'distorted_lines': use_distorted_lines, 'vhs_effect': use_vhs,
    'posterize': use_poster, 'bw_intensity': bw_intensity,
}

if audio_file:
    audio_bytes = audio_file.read()

    with st.spinner("🔍 Analizzando caratteristiche audio..."):
        features = analyze_audio_librosa(audio_bytes)

    if features:
        col_main, col_side = st.columns([2, 1])

        with col_main:
            if st.button("🎨 Genera Copertina Glitch", key="generate"):
                st.session_state.regen_count += 1

                modified_features = features.copy()
                modified_features['rms']              *= glitch_intensity
                modified_features['percussive_energy'] *= glitch_intensity
                modified_features['onset_strength']   *= glitch_intensity

                dimensions = get_dimensions(aspect_ratio)

                with st.spinner("🎨 Generando copertina glitch..."):
                    start = time.time()
                    glitch_img, descrizione = generate_glitch_image(
                        modified_features,
                        features['file_hash'],
                        st.session_state.regen_count,
                        dimensions,
                        glitch_controls
                    )
                    gen_time = time.time() - start

                buf = io.BytesIO()
                if img_format == "PNG":
                    glitch_img.save(buf, format='PNG')
                else:
                    glitch_img.save(buf, format='JPEG', quality=95)
                buf.seek(0)

                st.session_state.img_bytes        = buf.getvalue()
                st.session_state.img_format_saved = img_format
                st.session_state.img_emotion      = features['emotion']
                st.session_state.audio_filename   = audio_file.name
                st.session_state.descrizione      = descrizione
                st.session_state.features_saved   = features
                st.session_state.report_text = generate_report(
                    features, audio_file.name,
                    st.session_state.regen_count,
                    glitch_controls, glitch_intensity
                )
                st.success(f"✅ Copertina generata in {gen_time:.2f}s — variante #{st.session_state.regen_count}")

        with col_side:
            st.markdown("### 🎯 Rigenerazioni")
            st.markdown(f"**Variante:** #{st.session_state.regen_count}")
            if st.button("🔄 Nuova Variante"):
                st.session_state.regen_count += 1
                st.rerun()
            if st.button("🔄 Reset"):
                st.session_state.regen_count = 0
                st.session_state.img_bytes = None
                st.session_state.report_text = None
                st.rerun()
            st.markdown("### ℹ️ Motore V3")
            st.markdown("""
- **10 background** diversi
- **Palette procedurale** per ogni seed
- **6 overlay** grafici random
- **Ordine effetti** randomizzato
- **Hash mixing** robusto tra varianti
""")

        if show_analysis and st.session_state.descrizione:
            st.markdown("### 🔍 Analisi Audio")
            for desc in st.session_state.descrizione:
                st.markdown(f"- {desc}")
            f = st.session_state.features_saved or features
            ca, cb = st.columns(2)
            with ca:
                st.metric("🎵 BPM", f"{f['bpm']:.1f}")
                st.metric("🔊 RMS", f"{f['rms']:.3f}")
                st.metric("🎼 Centroide", f"{f['spectral_centroid']:.0f} Hz")
                st.metric("😊 Emozione", f['emotion'])
            with cb:
                st.metric("⚡ Percussiva", f"{f['percussive_energy']:.3f}")
                st.metric("🎹 Armonica", f"{f['harmonic_energy']:.3f}")
                st.metric("🌊 ZCR", f"{f['zero_crossing_rate']:.3f}")
                st.metric("📊 Onset", f"{f['onset_strength']:.1f}")

        # ── DOWNLOAD ZONE (fuori da button, usa session_state) ────────────
        if st.session_state.img_bytes is not None:
            st.divider()
            st.markdown("### 💾 Anteprima & Download")
            st.image(st.session_state.img_bytes,
                     caption=f"Variante #{st.session_state.regen_count} — {st.session_state.img_emotion}")

            ext  = "png" if st.session_state.img_format_saved=="PNG" else "jpg"
            mime = "image/png" if st.session_state.img_format_saved=="PNG" else "image/jpeg"
            fname = f"glitch_{st.session_state.regen_count}_{st.session_state.img_emotion.lower()}"

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label=f"🖼️ Scarica ({st.session_state.img_format_saved})",
                    data=st.session_state.img_bytes,
                    file_name=f"{fname}.{ext}", mime=mime,
                    use_container_width=True, key="dl_img"
                )
            with dl2:
                if st.session_state.report_text:
                    st.download_button(
                        label="📄 Scarica Report .txt",
                        data=st.session_state.report_text.encode("utf-8"),
                        file_name=f"report_{fname}.txt", mime="text/plain",
                        use_container_width=True, key="dl_report"
                    )
    else:
        st.error("❌ Impossibile analizzare il file audio.")
else:
    st.info("👆 Carica un file audio per iniziare!")
    st.markdown("""
**GlitchCover Studio V3** — Motore visivo completamente rinnovato:
- 🎨 **Palette procedurale** generata dai dati audio e dal seed
- 🖼️ **10 tipi di background**: plasma, voronoi, interference waves, fractal noise, radial burst...
- 🔲 **6 overlay grafici**: barre di frequenza, griglia, testo glitch, geometrie spezzate, databend...
- 🔀 **Ordine effetti randomizzato** ad ogni rigenerazione
- 🔑 **Hash mixing robusto**: ogni variante è visivamente distinta
""")

st.markdown("---")
st.markdown("**🔧 Loop507 Studio** — GlitchCover V3")
