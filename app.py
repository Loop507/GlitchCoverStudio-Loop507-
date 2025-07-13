import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageChops
import hashlib
import time
import librosa

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("🎧 GlitchCover Studio by Loop507 [Free App]")
st.markdown("Crea una copertina glitch unica basata sul tuo brano. Carica un file audio e genera arte ispirata al suono.")

def get_dimensions(format_type):
    if format_type == "Quadrato (1:1)":
        return (800, 800)
    elif format_type == "Verticale (9:16)":
        return (720, 1280)
    elif format_type == "Orizzontale (16:9)":
        return (1280, 720)
    else:
        return (800, 800)

def analyze_audio_librosa(file):
    try:
        file.seek(0)
        audio_bytes = file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True, duration=30)

        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(bpm) if bpm is not None else 60.0

        rms = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        dynamic_range = spectral_bandwidth - spectral_centroid

        hash_obj = hashlib.sha256(audio_bytes[:10000]).hexdigest()
        hash_int = int(hash_obj[:8], 16)

        if rms > 0.05 and bpm > 110:
            emotion = "Energetico"
        elif rms < 0.02 and bpm < 80:
            emotion = "Calmo"
        elif dynamic_range > 2000:
            emotion = "Dinamico"
        else:
            emotion = "Equilibrato"

        return {
            "bpm": bpm,
            "rms": rms,
            "spectral_centroid": spectral_centroid,
            "dynamic_range": dynamic_range,
            "emotion": emotion,
            "file_hash": hash_int,
            "file_size": len(audio_bytes)
        }
    except Exception as e:
        st.error(f"Errore nell'analisi audio con Librosa: {str(e)}")
        return None

def apply_tv_glitch_effects(img, features, seed):
    """Applica effetti glitch realistici tipo TV distorta"""
    random.seed(seed)
    width, height = img.size
    
    # 1. DATAMOSHING - Sposta blocchi di pixel
    for _ in range(random.randint(10, 30)):
        # Seleziona un blocco casuale
        block_w = random.randint(50, 200)
        block_h = random.randint(20, 100)
        x1 = random.randint(0, width - block_w)
        y1 = random.randint(0, height - block_h)
        
        # Copia il blocco
        block = img.crop((x1, y1, x1 + block_w, y1 + block_h))
        
        # Incollalo in una posizione casuale
        x2 = random.randint(0, width - block_w)
        y2 = random.randint(0, height - block_h)
        img.paste(block, (x2, y2))
    
    # 2. CHANNEL SHIFT - Sposta i canali RGB
    r, g, b = img.split()
    shift_intensity = int(features["dynamic_range"] / 100) + random.randint(5, 25)
    
    r = ImageChops.offset(r, random.randint(-shift_intensity, shift_intensity), 0)
    g = ImageChops.offset(g, random.randint(-shift_intensity, shift_intensity), 0)
    b = ImageChops.offset(b, random.randint(-shift_intensity, shift_intensity), 0)
    
    img = Image.merge("RGB", (r, g, b))
    
    # 3. SCANLINES - Linee di scansione TV
    scanlines = create_scanlines(width, height, random.randint(2, 4), 0.4)
    img = Image.alpha_composite(img.convert("RGBA"), scanlines).convert("RGB")
    
    # 4. HORIZONTAL TEARING - Strappi orizzontali
    tear_count = int(features["rms"] * 20) + random.randint(3, 8)
    for _ in range(tear_count):
        y = random.randint(0, height - 20)
        tear_height = random.randint(5, 20)
        shift = random.randint(-50, 50)
        
        # Estrai la sezione
        section = img.crop((0, y, width, y + tear_height))
        
        # Cancella la sezione originale
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, y, width, y + tear_height], fill="black")
        
        # Rimetti la sezione spostata
        new_x = max(0, min(width - section.width, shift))
        img.paste(section, (new_x, y))
    
    # 5. VERTICAL BARS - Barre verticali di disturbo
    bar_count = int(features["bpm"] / 20) + random.randint(2, 6)
    draw = ImageDraw.Draw(img)
    for _ in range(bar_count):
        x = random.randint(0, width - 10)
        bar_width = random.randint(2, 8)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.rectangle([x, 0, x + bar_width, height], fill=color)
    
    # 6. STATIC NOISE - Rumore statico
    static_intensity = min(0.5, features["rms"] * 2)
    static = create_tv_static(width, height, static_intensity)
    img = Image.blend(img, static, 0.3)
    
    # 7. INTERLACING - Effetto interlacciamento
    for y in range(1, height, 2):
        # Rendi le linee pari più scure
        draw = ImageDraw.Draw(img)
        overlay = Image.new("RGB", (width, 1), (0, 0, 0))
        img.paste(overlay, (0, y), overlay)
    
    # 8. COMPRESSION ARTIFACTS - Artefatti di compressione
    # Riduci e ingrandisci per creare artefatti
    compression_factor = random.randint(5, 15)
    compressed = img.resize((width // compression_factor, height // compression_factor), Image.NEAREST)
    img = compressed.resize((width, height), Image.NEAREST)
    
    # 9. CHROMATIC ABERRATION - Aberrazione cromatica
    if random.random() < 0.7:
        r, g, b = img.split()
        aberration = random.randint(2, 8)
        r = ImageChops.offset(r, aberration, 0)
        b = ImageChops.offset(b, -aberration, 0)
        img = Image.merge("RGB", (r, g, b))
    
    # 10. GLITCH BARS - Barre di disturbo casuali
    glitch_bars = create_glitch_bars(width, height, features["rms"])
    img = Image.blend(img, glitch_bars, 0.2)
    
    # 11. PIXEL DRIFT - Deriva dei pixel
    if random.random() < 0.5:
        drift_img = img.copy()
        pixels = drift_img.load()
        
        drift_count = int(width * height * 0.01)  # 1% dei pixel
        for _ in range(drift_count):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            # Sposta il pixel casualmente
            new_x = max(0, min(width - 1, x + random.randint(-3, 3)))
            new_y = max(0, min(height - 1, y + random.randint(-3, 3)))
            
            if (x, y) != (new_x, new_y):
                pixels[new_x, new_y] = pixels[x, y]
        
        img = drift_img
    
    return img

def create_tv_static(width, height, intensity=0.3):
    """Crea rumore TV statico"""
    static = Image.new("RGB", (width, height), "black")
    pixels = static.load()
    
    for x in range(width):
        for y in range(height):
            if random.random() < intensity:
                val = random.randint(0, 255)
                pixels[x, y] = (val, val, val)
    
    return static

def create_scanlines(width, height, line_height=2, opacity=0.3):
    """Crea linee di scansione TV"""
    scanlines = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(scanlines)
    
    for y in range(0, height, line_height * 2):
        alpha = int(255 * opacity)
        draw.rectangle([0, y, width, y + line_height], fill=(0, 0, 0, alpha))
    
    return scanlines

def create_glitch_bars(width, height, intensity):
    """Crea barre di disturbo orizzontali"""
    bars = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(bars)
    
    bar_count = int(intensity * 50)
    for _ in range(bar_count):
        y = random.randint(0, height)
        bar_height = random.randint(1, 10)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.rectangle([0, y, width, y + bar_height], fill=color)
    
    return bars

def generate_glitch_image(features, seed, size=(800, 800)):
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    
    width, height = size
    
    # BASE: Genera un'immagine base più varia
    base_pattern = random.choice(['gradient', 'noise', 'waves', 'geometric'])
    
    if base_pattern == 'gradient':
        # Gradiente basato sull'emozione
        img = Image.new("RGB", size, "black")
        draw = ImageDraw.Draw(img)
        
        palettes = {
            "Energetico": [(255, 0, 0), (255, 255, 0), (255, 100, 0)],
            "Calmo": [(0, 100, 200), (100, 200, 255), (200, 255, 255)],
            "Dinamico": [(255, 0, 255), (0, 255, 0), (255, 255, 0)],
            "Equilibrato": [(100, 0, 200), (200, 100, 255), (255, 200, 100)]
        }
        colors = palettes.get(features["emotion"], palettes["Equilibrato"])
        
        # Gradiente verticale con variazioni
        for y in range(height):
            progress = y / height
            color_idx = int(progress * (len(colors) - 1))
            color = colors[color_idx]
            
            # Aggiungi variazione basata su BPM
            variation = int(features["bpm"] * np.sin(y * 0.1) * 0.3)
            final_color = tuple(max(0, min(255, c + variation)) for c in color)
            
            draw.line([(0, y), (width, y)], fill=final_color)
    
    elif base_pattern == 'noise':
        # Pattern di rumore basato su RMS
        img = Image.new("RGB", size, "black")
        pixels = img.load()
        
        noise_intensity = min(1.0, features["rms"] * 10)
        for x in range(width):
            for y in range(height):
                if random.random() < noise_intensity:
                    intensity = random.randint(50, 255)
                    pixels[x, y] = (intensity, intensity//2, intensity//3)
    
    elif base_pattern == 'waves':
        # Pattern a onde basato su frequenza
        img = Image.new("RGB", size, "black")
        draw = ImageDraw.Draw(img)
        
        wave_freq = features["spectral_centroid"] / 1000
        wave_count = int(features["bpm"] / 10)
        
        for i in range(wave_count):
            y_base = (height // wave_count) * i
            for x in range(width):
                y_offset = int(30 * np.sin(x * wave_freq * 0.01))
                y = y_base + y_offset
                if 0 <= y < height:
                    color = (random.randint(100, 255), random.randint(0, 200), random.randint(0, 150))
                    draw.point((x, y), fill=color)
    
    else:  # geometric
        # Pattern geometrico
        img = Image.new("RGB", size, "black")
        draw = ImageDraw.Draw(img)
        
        shape_count = int(features["dynamic_range"] / 200) + 5
        for _ in range(shape_count):
            x, y = random.randint(0, width), random.randint(0, height)
            size_val = random.randint(20, 100)
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            
            if random.random() < 0.5:
                draw.ellipse([x, y, x + size_val, y + size_val], fill=color)
            else:
                draw.rectangle([x, y, x + size_val, y + size_val], fill=color)
    
    # APPLICA EFFETTI GLITCH TV
    img = apply_tv_glitch_effects(img, features, seed)
    
    descrizione = [
        f"🎵 BPM stimato: {features['bpm']:.1f}",
        f"🔊 RMS (energia media): {features['rms']:.3f}",
        f"📡 Centro spettrale: {features['spectral_centroid']:.0f} Hz",
        f"🎭 Stile emotivo: {features['emotion']}",
        f"🎨 Pattern base: {base_pattern}"
    ]

    return img, descrizione

# --- UI ---

if 'regen_count' not in st.session_state:
    st.session_state.regen_count = 0

audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])

col1, col2, col3 = st.columns(3)
with col1:
    img_format = st.selectbox("Formato immagine:", ["PNG", "JPEG"])
with col2:
    aspect_ratio = st.selectbox("Formato dimensioni:", ["Quadrato (1:1)", "Verticale (9:16)", "Orizzontale (16:9)"])
with col3:
    show_analysis = st.checkbox("Mostra analisi", value=True)

if audio_file:
    features = analyze_audio_librosa(audio_file)
    if features:
        # MOSTRA L'ANALISI SUBITO DOPO L'ELABORAZIONE
        if show_analysis:
            st.markdown("### 🎨 Analisi Audio")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🎵 BPM", f"{features['bpm']:.1f}")
                st.metric("🔊 RMS", f"{features['rms']:.3f}")
            with col_b:
                st.metric("📡 Centro spettrale", f"{features['spectral_centroid']:.0f} Hz")
                st.metric("🎭 Emozione", features['emotion'])
        
        dimensions = get_dimensions(aspect_ratio)
        
        # MIGLIORA IL SEED PER VARIAZIONI REALI
        base_seed = features['file_hash']
        
        if st.button("🔄 Rigenera Copertina"):
            st.session_state.regen_count += 1
            st.rerun()  # Forza il refresh della pagina

        # CREA UN SEED COMPLETAMENTE DIVERSO AD OGNI RIGENERAZIONE
        timestamp_component = int(time.time() * 1000000) % 1000000
        random_component = random.randint(1, 999999)
        regen_component = st.session_state.regen_count * 54321
        
        if st.session_state.regen_count == 0:
            seed = base_seed  # Prima volta usa il seed del file
        else:
            # Rigenerazioni: combina tutti i componenti per massima variazione
            seed = (timestamp_component + random_component + regen_component + base_seed) % 2147483647

        with st.spinner("🎨 Creazione copertina glitch..."):
            img, descrizione = generate_glitch_image(features, seed, size=dimensions)

        st.image(img, caption=f"Copertina glitch generata - {aspect_ratio} (Versione #{st.session_state.regen_count + 1})", use_container_width=True)

        # MOSTRA I DETTAGLI TECNICI DELLA GENERAZIONE
        if show_analysis:
            with st.expander("🔧 Dettagli Tecnici Generazione"):
                st.write(f"**Seed utilizzato:** {seed}")
                st.write(f"**Numero rigenerazioni:** {st.session_state.regen_count + 1}")
                st.write("**Parametri utilizzati:**")
                for d in descrizione:
                    st.markdown(f"- {d}")

        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()

        filename = f"glitch_cover_{features['emotion'].lower()}_{int(time.time())}.{img_format.lower()}"
        st.download_button(
            label=f"⬇️ Scarica {img_format} ({dimensions[0]}x{dimensions[1]})",
            data=byte_im,
            file_name=filename,
            mime=f"image/{img_format.lower()}"
        )
else:
    st.session_state.regen_count = 0
    st.info("👆 Carica un file audio per iniziare!")
