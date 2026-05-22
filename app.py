import streamlit as st
import numpy as np
import io
import random
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
    if format_type == "Quadrato (1:1)":
        return (800, 800)
    elif format_type == "Verticale (9:16)":
        return (720, 1280)
    elif format_type == "Orizzontale (16:9)":
        return (1280, 720)
    return (800, 800)

# ─────────────────────────────────────────────
# ANALISI AUDIO — con cache su bytes
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

# ─────────────────────────────────────────────
# GENERAZIONE IMMAGINE
# ─────────────────────────────────────────────

def create_random_base_image(size, features, seed):
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    width, height = size

    emotion_palettes = {
        "Aggressive": [(255,0,0),(255,100,0),(200,0,100),(255,255,0),(255,0,255)],
        "Energetico": [(255,140,0),(255,0,255),(0,255,255),(255,255,0),(255,100,100)],
        "Ambient":    [(0,100,255),(100,0,255),(0,255,200),(100,255,100),(200,100,255)],
        "Calmo":      [(0,150,255),(100,200,255),(0,255,150),(150,255,200),(200,150,255)],
        "Melodico":   [(200,0,255),(255,0,200),(100,100,255),(255,100,200),(200,100,255)],
        "Ritmico":    [(255,0,255),(0,255,255),(255,255,0),(255,100,0),(100,255,100)],
        "Dinamico":   [(255,50,150),(150,255,50),(50,150,255),(255,150,50),(150,50,255)],
        "Equilibrato":[(100,255,100),(255,100,100),(100,100,255),(255,255,100),(255,100,255)]
    }
    colors = emotion_palettes.get(features["emotion"], emotion_palettes["Equilibrato"])
    colors_arr = np.array(colors, dtype=np.uint8)
    img = Image.new('RGB', size, (0,0,0))
    draw = ImageDraw.Draw(img)
    bg_type = seed % 4

    if bg_type == 0:
        # Gradiente diagonale — vettorizzato con NumPy
        xs = np.arange(width)
        ys = np.arange(height)
        xv, yv = np.meshgrid(xs, ys)
        idx = ((xv + yv) / (width + height) * len(colors)).astype(int) % len(colors)
        img = Image.fromarray(colors_arr[idx], 'RGB')
    elif bg_type == 1:
        band_width = max(10, int(features['spectral_bandwidth'] / 100))
        for i in range(0, width, band_width):
            color = colors[i // band_width % len(colors)]
            draw.rectangle([i, 0, i + band_width, height], fill=color)
    elif bg_type == 2:
        center_x, center_y = width // 2, height // 2
        max_radius = max(width, height) // 2
        for radius in range(0, max_radius, 20):
            color = colors[radius // 20 % len(colors)]
            draw.ellipse([center_x-radius, center_y-radius,
                          center_x+radius, center_y+radius], outline=color, width=10)
    else:
        # Pattern rumore — vettorizzato
        noise_array = np.random.randint(0, len(colors), (height, width))
        img = Image.fromarray(colors_arr[noise_array], 'RGB')

    return img

def apply_black_and_white(img, intensity=0.5):
    gray = img.convert('L').convert('RGB')
    return Image.blend(img, gray, intensity)

def apply_distorted_lines(img, features, seed):
    random.seed(seed)
    width, height = img.size
    draw = ImageDraw.Draw(img)
    num_lines = int(features['percussive_energy'] * 200) + 5
    for _ in range(num_lines):
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        start_x = random.randint(0, width)
        start_y = random.randint(0, height)
        points = []
        for i in range(0, width, 10):
            y_offset = int(np.sin(i * 0.01 + seed) * features['zero_crossing_rate'] * 100)
            points.append((start_x + i, start_y + y_offset))
        if len(points) > 1:
            draw.line(points, fill=color, width=random.randint(1, 5))
    return img

def apply_vhs_effect(img, features, seed):
    random.seed(seed)
    width, height = img.size
    pixels = np.array(img)
    noise_intensity = max(0.1, min(0.5, features['rms'] * 5))
    for y in range(height):
        if random.random() < noise_intensity:
            noise_width = random.randint(5, 50)
            noise_start = random.randint(0, max(1, width - noise_width))
            pixels[y, noise_start:noise_start+noise_width] = [random.randint(0,255) for _ in range(3)]
    r, g, b = Image.fromarray(pixels).split()
    r = ImageChops.offset(r, 2, 0)
    b = ImageChops.offset(b, -2, 0)
    vhs_img = Image.merge('RGB', (r, g, b))
    return ImageEnhance.Color(vhs_img).enhance(0.7)

def apply_pixel_sorting(img, features, seed):
    random.seed(seed)
    width, height = img.size
    pixels = np.array(img)
    sort_intensity = max(0.1, min(0.9, features['onset_strength']/10 + features['percussive_energy']*5))
    if random.random() < 0.5:
        for y in range(0, height, random.randint(2,8)):
            if random.random() < sort_intensity:
                row = pixels[y].copy()
                idx = np.argsort(np.mean(row, axis=1) if random.random()<0.5 else row[:,random.randint(0,2)])
                pixels[y] = row[idx]
    else:
        for x in range(0, width, random.randint(2,8)):
            if random.random() < sort_intensity:
                col = pixels[:,x].copy()
                idx = np.argsort(np.mean(col, axis=1) if random.random()<0.5 else col[:,random.randint(0,2)])
                pixels[:,x] = col[idx]
    return Image.fromarray(pixels)

def apply_digital_corruption(img, features, seed):
    random.seed(seed)
    width, height = img.size
    pixels = np.array(img)
    corruption_intensity = max(0.1, min(0.8, features['zero_crossing_rate']*5 + features['rms']*3))
    num_corruptions = int(corruption_intensity * 50)
    for _ in range(num_corruptions):
        ct = random.randint(0,4)
        if ct == 0:
            bw = random.randint(10, min(100,width//4,height//4))
            bh = random.randint(10, min(100,width//4,height//4))
            x = random.randint(0, max(1,width-bw)); y = random.randint(0, max(1,height-bh))
            pixels[y:min(y+bh,height), x:min(x+bw,width)] = 255 - pixels[y:min(y+bh,height), x:min(x+bw,width)]
        elif ct == 1:
            y = random.randint(0, height-1); lh = random.randint(1, min(5,height-y))
            shift = random.randint(-50,50)
            if shift > 0 and shift < width: pixels[y:y+lh, shift:] = pixels[y:y+lh, :-shift]
            elif shift < 0 and abs(shift) < width: pixels[y:y+lh, :shift] = pixels[y:y+lh, -shift:]
        elif ct == 2:
            bw = random.randint(5, min(30,width//8)); bh = random.randint(5, min(30,height//8))
            x = random.randint(0, max(1,width-bw)); y = random.randint(0, max(1,height-bh))
            pixels[y:min(y+bh,height), x:min(x+bw,width)] = np.random.randint(0,255,(min(bh,height-y),min(bw,width-x),3))
        elif ct == 3:
            if random.random() < 0.5:
                sh = random.randint(1, min(8,height//10)); y = random.randint(0, max(1,height-sh))
                ty = random.randint(0, max(1,height-sh)); pixels[ty:ty+sh] = pixels[y:y+sh]
            else:
                sw = random.randint(1, min(8,width//10)); x = random.randint(0, max(1,width-sw))
                tx = random.randint(0, max(1,width-sw)); pixels[:,tx:tx+sw] = pixels[:,x:x+sw]
        elif ct == 4:
            offset = random.randint(5, min(30,width//10)); ch = random.randint(0,2)
            d = random.choice([(offset,0),(-offset,0),(0,offset),(0,-offset)])
            if d[0]>0 and d[0]<width: pixels[:,offset:,ch] = pixels[:,:-offset,ch]
            elif d[0]<0 and abs(d[0])<width: pixels[:,:offset,ch] = pixels[:,-offset:,ch]
            elif d[1]>0 and d[1]<height: pixels[offset:,:,ch] = pixels[:-offset,:,ch]
            elif d[1]<0 and abs(d[1])<height: pixels[:offset,:,ch] = pixels[-offset:,:,ch]
    return Image.fromarray(pixels)

def apply_random_distortion(img, features, seed):
    random.seed(seed)
    width, height = img.size
    pixels = np.array(img)
    dt = random.randint(0,3)
    if dt == 0:
        buf = io.BytesIO(); img.save(buf, format='JPEG', quality=random.randint(10,50)); buf.seek(0); return Image.open(buf)
    elif dt == 1: return img.quantize(colors=random.randint(8,64)).convert('RGB')
    elif dt == 2:
        scale = random.uniform(0.3,0.8)
        return img.resize((int(width*scale),int(height*scale)), Image.NEAREST).resize((width,height), Image.NEAREST)
    elif dt == 3:
        for y in range(height):
            wo = int(np.sin(y*0.1+seed)*20)
            if wo>0 and wo<width: pixels[y,wo:] = pixels[y,:-wo]
            elif wo<0 and abs(wo)<width: pixels[y,:wo] = pixels[y,-wo:]
        return Image.fromarray(pixels)
    return img

def apply_advanced_chromatic_aberration(img, features, seed):
    random.seed(seed)
    base_offset = max(3, min(30, int(features['spectral_bandwidth']/100)))
    rms_mul = max(2, min(15, int(features['rms']*150)))
    r, g, b = img.split()
    r = ImageChops.offset(r, base_offset+random.randint(-rms_mul,rms_mul), random.randint(-5,5))
    g = ImageChops.offset(g, random.randint(-base_offset//2,base_offset//2), random.randint(-3,3))
    b = ImageChops.offset(b, -base_offset+random.randint(-rms_mul//2,rms_mul//2), random.randint(-5,5))
    return Image.merge('RGB', (r, g, b))

def apply_recursive_glitch(img, features, seed, iterations=3, glitch_controls=None):
    if glitch_controls is None:
        glitch_controls = {
            'pixel_sorting':True,'digital_corruption':True,'chromatic_aberration':True,
            'black_and_white':False,'distorted_lines':False,'vhs_effect':False,
            'bw_intensity':0.5,'distorted_lines_intensity':1.0,'vhs_intensity':1.0
        }
    current_img = img.copy()
    for i in range(iterations):
        s = seed + i*1000
        if glitch_controls['pixel_sorting'] and features['percussive_energy'] > 0.03:
            current_img = apply_pixel_sorting(current_img, features, s)
        if glitch_controls['digital_corruption'] and features['onset_strength'] > 2:
            current_img = apply_digital_corruption(current_img, features, s+100)
        if glitch_controls['chromatic_aberration'] and features['spectral_bandwidth'] > 1500:
            current_img = apply_advanced_chromatic_aberration(current_img, features, s+200)
        if glitch_controls['black_and_white']:
            current_img = apply_black_and_white(current_img, glitch_controls['bw_intensity'])
        if glitch_controls['distorted_lines']:
            current_img = apply_distorted_lines(current_img, features, s+400)
        if glitch_controls['vhs_effect']:
            current_img = apply_vhs_effect(current_img, features, s+500)
        if random.random() < 0.7:
            current_img = apply_random_distortion(current_img, features, s+300)
    return current_img

def generate_advanced_glitch_image(features, seed, size=(800,800), glitch_controls=None):
    random.seed(seed); np.random.seed(seed % 2147483647)
    img = create_random_base_image(size, features, seed)
    img = apply_recursive_glitch(img, features, seed, glitch_controls=glitch_controls)
    if features['emotion'] == 'Aggressive':
        img = apply_digital_corruption(img, features, seed+5000)
        img = apply_pixel_sorting(img, features, seed+6000)
    elif features['emotion'] == 'Energetico':
        img = apply_advanced_chromatic_aberration(img, features, seed+7000)
        img = apply_random_distortion(img, features, seed+8000)
    elif features['emotion'] == 'Ambient':
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        img = apply_advanced_chromatic_aberration(img, features, seed+9000)
    elif features['emotion'] == 'Ritmico':
        img = apply_pixel_sorting(img, features, seed+10000)
        img = apply_digital_corruption(img, features, seed+11000)
    else:
        effects = [
            lambda: apply_pixel_sorting(img, features, seed+12000),
            lambda: apply_digital_corruption(img, features, seed+13000),
            lambda: apply_random_distortion(img, features, seed+14000)
        ]
        for effect in random.sample(effects, random.randint(1,3)):
            img = effect()
    if features['harmonic_energy'] > 0.03:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.1,1.5))
    if features['rms'] > 0.05:
        img = ImageEnhance.Color(img).enhance(random.uniform(1.1,1.8))

    descrizione = [
        f"🎵 BPM: {features['bpm']:.1f} → Velocità pattern e corruzioni",
        f"🔊 RMS: {features['rms']:.3f} → Intensità effetti glitch",
        f"🎼 Centro spettrale: {features['spectral_centroid']:.0f} Hz → Tipo di distorsione",
        f"⚡ Energia percussiva: {features['percussive_energy']:.3f} → Pixel sorting",
        f"🎹 Energia armonica: {features['harmonic_energy']:.3f} → Aberrazione cromatica",
        f"🌊 Zero crossing: {features['zero_crossing_rate']:.3f} → Corruzioni digitali",
        f"😊 Emozione: {features['emotion']} → Stile glitch generale",
        f"📊 Onset strength: {features['onset_strength']:.1f} → Ricorsività effetti"
    ]
    return img, descrizione

# ─────────────────────────────────────────────
# REPORT TESTUALE STILIZZATO
# ─────────────────────────────────────────────

def generate_report(features, audio_filename, regen_count, glitch_controls, glitch_intensity):
    emotion_descriptions = {
        "Aggressive": "brutal e disturbante, con corruzioni digitali intense e pixel sorting estremo che riflettono l'energia distruttiva del brano.",
        "Energetico": "vibrante e dinamico, con aberrazione cromatica vivida e colori esplosivi che catturano la potenza della traccia.",
        "Ambient": "etereo e sospeso, con glitch morbidi e sfumature cromatiche che evocano l'atmosfera rarefatta del suono.",
        "Calmo": "meditativo e fluido, con distorsioni gentili che si dissolvono nello spazio visivo come note nell'aria.",
        "Melodico": "armonico e ipnotico, con pattern di distorsione che seguono le curve melodiche del brano.",
        "Ritmico": "pulsante e geometrico, con corruzioni sincronizzate al battito che creano una danza visiva frenetica.",
        "Dinamico": "contrastato e imprevedibile, dove silenzi e esplosioni sonore si traducono in tensioni visive opposte.",
        "Equilibrato": "bilanciato e stratificato, con un mix di effetti che rispecchiano la complessita multidimensionale del suono."
    }

    bpm_feel = "lento e pesante" if features['bpm'] < 80 else (
        "moderato" if features['bpm'] < 120 else (
        "veloce" if features['bpm'] < 160 else "frenetico"))
    rms_feel = "sussurrato" if features['rms'] < 0.02 else (
        "delicato" if features['rms'] < 0.05 else (
        "potente" if features['rms'] < 0.08 else "esplosivo"))

    active_effects = [k.replace("_"," ") for k, v in glitch_controls.items()
                      if isinstance(v, bool) and v and k not in ('black_and_white',)]
    effects_str = ", ".join(active_effects) if active_effects else "standard"
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    dominant_note = notes[features['tonal_centroid'] % 12]
    track_name = audio_filename.rsplit('.', 1)[0]

    report = f"""╔══════════════════════════════════════════════════════════════╗
║         🎵 GLITCHCOVER STUDIO — REPORT AUDIO/VISIVO          ║
║                      by Loop507                              ║
╚══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 FILE:      {audio_filename}
🎨 VARIANTE:  #{regen_count}
🧬 SIGNATURE: {features['audio_signature']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎭  IDENTITA SONORA
    Emozione rilevata   →  {features['emotion'].upper()}
    Descrizione visiva  →  Un artwork {emotion_descriptions.get(features['emotion'], "unico e irripetibile.")}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊  DATI AUDIO ESTRATTI (librosa analysis)

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
    Dimensione file      →  {features['file_size'] / 1024:.1f} KB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎛️  PARAMETRI DI GENERAZIONE

    Intensita Glitch     →  {glitch_intensity:.1f}x
    Effetti applicati    →  {effects_str}
    Bianco e Nero        →  {"ON (intensita " + str(glitch_controls.get('bw_intensity',0)) + ")" if glitch_controls.get('black_and_white') else "OFF"}
    Effetto VHS          →  {"ON" if glitch_controls.get('vhs_effect') else "OFF"}
    Linee Distorte       →  {"ON" if glitch_controls.get('distorted_lines') else "OFF"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📱  DESCRIZIONE PER SOCIAL / YOUTUBE

🎵 "{track_name}" — Artwork generativo creato da GlitchCover Studio

Questa cover e nata dall'analisi algoritmica del brano.
Ogni pixel e il riflesso di un dato sonoro reale:

→ Il BPM ({features['bpm']:.0f}) ha guidato la velocita dei pattern glitch
→ L'energia RMS ({features['rms']:.3f}) ha determinato l'intensita delle corruzioni
→ Il centroide spettrale ({features['spectral_centroid']:.0f} Hz) ha definito il tipo di distorsione
→ Le {features['percussive_energy']:.3f} unita di energia percussiva hanno generato il pixel sorting
→ La nota dominante ({dominant_note}) ha influenzato le sfumature cromatiche
→ L'emozione rilevata — {features['emotion']} — ha scelto la palette e lo stile generale

🤖 Generato da #GlitchCoverStudio · Powered by #Loop507
🎨 Arte algoritmica basata su analisi audio reale · No AI generativa
🔁 Ogni brano crea un artwork unico e irripetibile

#glitchart #coverart #algorithmicart #musiccover #audiovisual
#glitch #digitalart #{features['emotion'].lower()} #bpm{int(features['bpm'])}
#loop507 #generativeart #musicproduction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 Loop507 Studio · glitchcover.loop507.com
Generato il: {time.strftime('%Y-%m-%d %H:%M')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return report

# ─────────────────────────────────────────────
# SESSION STATE — inizializzazione
# ─────────────────────────────────────────────

for key, default in [
    ('regen_count', 0),
    ('img_bytes', None),
    ('img_format_saved', 'PNG'),
    ('img_emotion', ''),
    ('report_text', None),
    ('audio_filename', ''),
    ('descrizione', []),
    ('features_saved', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# UI — UPLOAD & CONTROLLI
# ─────────────────────────────────────────────

audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3, WAV, OGG)", type=["mp3","wav","ogg"])

col1, col2, col3 = st.columns(3)
with col1:
    img_format = st.selectbox("Formato immagine:", ["PNG","JPEG"])
with col2:
    aspect_ratio = st.selectbox("Formato dimensioni:", ["Quadrato (1:1)","Verticale (9:16)","Orizzontale (16:9)"])
with col3:
    show_analysis = st.checkbox("Mostra analisi avanzata", value=True)

st.sidebar.markdown("### 🎛️ Controlli Glitch")
glitch_intensity = st.sidebar.slider("Intensita Glitch", 0.1, 2.0, 1.0, 0.1)
st.sidebar.markdown("#### Effetti Base")
use_pixel_sorting      = st.sidebar.checkbox("Pixel Sorting", value=True)
use_digital_corruption = st.sidebar.checkbox("Corruzioni Digitali", value=True)
use_chromatic          = st.sidebar.checkbox("Aberrazione Cromatica", value=True)
st.sidebar.markdown("#### Nuovi Effetti")
use_bw = st.sidebar.checkbox("Bianco e Nero", value=False)
bw_intensity = st.sidebar.slider("Intensita B&N", 0.1, 1.0, 0.5, 0.1) if use_bw else 0.5
use_distorted_lines = st.sidebar.checkbox("Linee Distorte", value=False)
use_vhs = st.sidebar.checkbox("Effetto VHS", value=False)

glitch_controls = {
    'pixel_sorting': use_pixel_sorting,
    'digital_corruption': use_digital_corruption,
    'chromatic_aberration': use_chromatic,
    'black_and_white': use_bw,
    'distorted_lines': use_distorted_lines,
    'vhs_effect': use_vhs,
    'bw_intensity': bw_intensity,
    'distorted_lines_intensity': 1.0,
    'vhs_intensity': 1.0
}

# ─────────────────────────────────────────────
# UI — LOGICA PRINCIPALE
# ─────────────────────────────────────────────

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
                seed = features['file_hash'] + st.session_state.regen_count * 12345

                with st.spinner("🎨 Generando copertina glitch..."):
                    start = time.time()
                    glitch_img, descrizione = generate_advanced_glitch_image(
                        modified_features, seed, dimensions, glitch_controls
                    )
                    gen_time = time.time() - start

                # Serializza immagine → session_state (evita rerun al download)
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

                # Genera report e salvalo in session_state
                st.session_state.report_text = generate_report(
                    features, audio_file.name,
                    st.session_state.regen_count,
                    glitch_controls, glitch_intensity
                )

                st.success(f"✅ Copertina generata in {gen_time:.2f} secondi!")

        with col_side:
            st.markdown("### 🎯 Rigenerazioni")
            st.markdown(f"**Numero:** {st.session_state.regen_count}")
            if st.button("🔄 Nuova Variante"):
                st.session_state.regen_count += 1
                st.rerun()
            if st.button("🔄 Reset"):
                st.session_state.regen_count = 0
                st.session_state.img_bytes = None
                st.session_state.report_text = None
                st.rerun()
            st.markdown("### ℹ️ Come Funziona")
            st.markdown("""
- **BPM** → Velocità pattern
- **RMS** → Intensità glitch
- **Spettro** → Tipo corruzioni
- **Percussioni** → Pixel sorting
- **Armonie** → Aberrazione cromatica
- **Zero Crossing** → Corruzioni digitali
""")

        # ── ANALISI (mostrata sotto, fuori dal button block) ──────────────
        if show_analysis and st.session_state.descrizione:
            st.markdown("### 🔍 Analisi Audio Dettagliata")
            for desc in st.session_state.descrizione:
                st.markdown(f"- {desc}")
            f = st.session_state.features_saved or features
            ca, cb = st.columns(2)
            with ca:
                st.metric("🎵 BPM", f"{f['bpm']:.1f}")
                st.metric("🔊 RMS Energy", f"{f['rms']:.3f}")
                st.metric("🎼 Spectral Centroid", f"{f['spectral_centroid']:.0f} Hz")
                st.metric("😊 Emozione", f['emotion'])
            with cb:
                st.metric("⚡ Energia Percussiva", f"{f['percussive_energy']:.3f}")
                st.metric("🎹 Energia Armonica", f"{f['harmonic_energy']:.3f}")
                st.metric("🌊 Zero Crossing Rate", f"{f['zero_crossing_rate']:.3f}")
                st.metric("📊 Onset Strength", f"{f['onset_strength']:.1f}")

        # ── DOWNLOAD ZONE ─────────────────────────────────────────────────
        # Renderizzati FUORI da qualsiasi if button: non causano rerun.
        # Persistono tra rerun grazie a session_state.
        # ─────────────────────────────────────────────────────────────────

        if st.session_state.img_bytes is not None:
            st.divider()
            st.markdown("### 💾 Anteprima & Download")

            st.image(
                st.session_state.img_bytes,
                caption=f"Copertina #{st.session_state.regen_count} — {st.session_state.img_emotion}"
            )

            ext  = "png" if st.session_state.img_format_saved == "PNG" else "jpg"
            mime = "image/png" if st.session_state.img_format_saved == "PNG" else "image/jpeg"
            fname_base = f"glitch_{st.session_state.regen_count}_{st.session_state.img_emotion.lower()}"

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label=f"🖼️ Scarica Copertina ({st.session_state.img_format_saved})",
                    data=st.session_state.img_bytes,
                    file_name=f"{fname_base}.{ext}",
                    mime=mime,
                    use_container_width=True,
                    key="dl_img"
                )
            with dl2:
                if st.session_state.report_text:
                    st.download_button(
                        label="📄 Scarica Report .txt",
                        data=st.session_state.report_text.encode("utf-8"),
                        file_name=f"report_{fname_base}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="dl_report"
                    )

    else:
        st.error("❌ Impossibile analizzare il file audio. Verifica che sia un formato supportato.")

else:
    st.info("👆 Carica un file audio per iniziare a generare la tua copertina glitch!")
    st.markdown("### 🎨 Esempi di Copertine Glitch")
    st.markdown("""
**GlitchCover Studio** analizza le caratteristiche del tuo brano e genera arte glitch unica:

- 🎵 **Musica Aggressiva** → Corruzioni digitali intense, pixel sorting estremo
- 🎶 **Musica Energetica** → Aberrazione cromatica vivida, colori vibranti
- 🎼 **Musica Ambient** → Glitch morbidi, effetti sfumati
- 🥁 **Musica Ritmica** → Pattern geometrici, corruzioni sincronizzate
- 🎹 **Musica Melodica** → Distorsioni armoniose, palette cromatiche
""")

st.markdown("---")
st.markdown("**🔧 Loop507 Studio** — Generatore di copertine glitch basato su analisi audio avanzata")
st.markdown("*Ogni copertina è unica e basata sulle caratteristiche sonore del tuo brano*")
