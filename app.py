import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageChops, ImageFilter
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

def create_tv_static_noise(size, intensity=0.3):
    """Crea rumore TV statico"""
    width, height = size
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Applica intensità
    noise = (noise * intensity).astype(np.uint8)
    
    return Image.fromarray(noise, 'RGB')

def apply_scanlines(img, spacing=2, opacity=0.15):
    """Applica linee di scansione TV"""
    width, height = img.size
    overlay = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    for y in range(0, height, spacing):
        draw.line([(0, y), (width, y)], fill=(0, 0, 0), width=1)
    
    # Blend con l'immagine originale
    overlay = overlay.convert('RGBA')
    img = img.convert('RGBA')
    
    # Applica trasparenza
    alpha = int(255 * opacity)
    overlay.putalpha(alpha)
    
    result = Image.alpha_composite(img, overlay)
    return result.convert('RGB')

def apply_chromatic_aberration(img, offset_r=5, offset_g=0, offset_b=-5):
    """Applica aberrazione cromatica"""
    r, g, b = img.split()
    
    # Sposta i canali
    r = ImageChops.offset(r, offset_r, 0)
    g = ImageChops.offset(g, offset_g, 0)
    b = ImageChops.offset(b, offset_b, 0)
    
    return Image.merge('RGB', (r, g, b))

def apply_data_moshing(img, features, seed):
    """Applica effetti di datamoshing"""
    random.seed(seed)
    width, height = img.size
    
    # Intensità basata sul BPM
    intensity = min(int(features['bpm'] / 10), 15)
    
    # Crea disturbi orizzontali (glitch linee)
    for _ in range(random.randint(5, intensity)):
        y_pos = random.randint(0, height - 1)
        line_height = random.randint(1, 8)
        
        # Estrai una striscia
        if y_pos + line_height < height:
            strip = img.crop((0, y_pos, width, y_pos + line_height))
            
            # Sposta la striscia
            offset = random.randint(-50, 50)
            new_y = max(0, min(height - line_height, y_pos + random.randint(-20, 20)))
            
            # Duplica o sposta la striscia
            img.paste(strip, (offset, new_y))
    
    return img

def apply_rgb_shift(img, features, seed):
    """Applica shift RGB basato sull'audio"""
    random.seed(seed)
    
    # Intensità basata sull'RMS
    shift_intensity = int(features['rms'] * 100) + random.randint(2, 15)
    
    r, g, b = img.split()
    
    # Shift diversi per ogni canale
    r = ImageChops.offset(r, random.randint(-shift_intensity, shift_intensity), 
                         random.randint(-2, 2))
    g = ImageChops.offset(g, random.randint(-shift_intensity//2, shift_intensity//2), 
                         random.randint(-2, 2))
    b = ImageChops.offset(b, random.randint(-shift_intensity, shift_intensity), 
                         random.randint(-2, 2))
    
    return Image.merge('RGB', (r, g, b))

def apply_digital_noise(img, features, seed):
    """Applica rumore digitale"""
    random.seed(seed)
    width, height = img.size
    
    # Intensità basata sul dynamic range
    noise_intensity = min(features['dynamic_range'] / 5000, 0.4)
    
    # Crea rumore
    noise = create_tv_static_noise((width, height), noise_intensity)
    
    # Blend con l'immagine originale
    img = img.convert('RGBA')
    noise = noise.convert('RGBA')
    
    # Applica il rumore solo in alcune zone
    for _ in range(random.randint(3, 8)):
        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height // 2)
        x2 = x1 + random.randint(100, width // 2)
        y2 = y1 + random.randint(100, height // 2)
        
        # Ritaglia zona di rumore
        if x2 < width and y2 < height:
            noise_section = noise.crop((x1, y1, x2, y2))
            img.paste(noise_section, (x1, y1), noise_section)
    
    return img.convert('RGB')

def create_waveform_visual(features, size, seed):
    """Crea visualizzazione basata sulla forma d'onda"""
    random.seed(seed)
    width, height = size
    
    # Palette colori basata sull'emozione
    palettes = {
        "Energetico": [(255, 0, 100), (255, 100, 0), (100, 255, 0)],
        "Calmo": [(0, 100, 255), (100, 0, 255), (0, 255, 200)],
        "Dinamico": [(255, 0, 255), (0, 255, 255), (255, 255, 0)],
        "Equilibrato": [(150, 0, 255), (255, 0, 150), (0, 255, 150)]
    }
    
    colors = palettes.get(features["emotion"], palettes["Equilibrato"])
    
    # Crea base scura
    img = Image.new('RGB', size, (5, 5, 15))
    draw = ImageDraw.Draw(img)
    
    # Genera pattern basati sull'audio
    frequency_bands = int(features['spectral_centroid'] / 100)
    amplitude = features['rms'] * 1000
    
    # Disegna forme d'onda multiple
    for band in range(min(frequency_bands, 8)):
        y_center = height // (band + 2)
        wave_color = random.choice(colors)
        
        # Genera punti della forma d'onda
        points = []
        for x in range(0, width, 4):
            # Forma d'onda basata su BPM e frequenza
            wave_height = amplitude * np.sin(x * features['bpm'] / 1000 + band) * (band + 1)
            wave_height += random.randint(-10, 10)  # Rumore
            
            y = int(y_center + wave_height)
            y = max(0, min(height - 1, y))
            points.append((x, y))
        
        # Disegna la forma d'onda
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=wave_color, width=random.randint(1, 3))
    
    # Aggiungi elementi di frequenza
    for _ in range(int(features['bpm'] // 10)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(5, 30)
        color = random.choice(colors)
        
        # Cerchi con trasparenza
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                    outline=color, width=random.randint(1, 3))
    
    return img

def generate_glitch_image(features, seed, size=(800, 800)):
    """Genera immagine glitch basata su audio"""
    try:
        random.seed(seed)
        np.random.seed(seed % 2147483647)
        
        # 1. Crea visualizzazione base dalla forma d'onda
        img = create_waveform_visual(features, size, seed)
        
        # 2. Applica effetti glitch in sequenza
        
        # Aberrazione cromatica
        chromatic_offset = int(features['rms'] * 50) + random.randint(2, 10)
        img = apply_chromatic_aberration(img, 
                                       offset_r=chromatic_offset,
                                       offset_b=-chromatic_offset)
        
        # Shift RGB
        img = apply_rgb_shift(img, features, seed + 1000)
        
        # Datamoshing
        img = apply_data_moshing(img, features, seed + 2000)
        
        # Rumore digitale
        img = apply_digital_noise(img, features, seed + 3000)
        
        # Linee di scansione
        scanline_spacing = max(2, int(6 - features['rms'] * 20))
        img = apply_scanlines(img, spacing=scanline_spacing, opacity=0.2)
        
        # Effetto finale di disturbo
        if random.random() < 0.7:  # 70% probabilità
            img = apply_chromatic_aberration(img, 
                                           offset_r=random.randint(1, 5),
                                           offset_b=random.randint(-5, -1))
        
        descrizione = [
            f"🎵 BPM: {features['bpm']:.1f} → Intensità glitch",
            f"🔊 RMS: {features['rms']:.3f} → Shift cromatico",
            f"📡 Freq: {features['spectral_centroid']:.0f} Hz → Pattern forma d'onda",
            f"🎭 Emozione: {features['emotion']} → Palette colori"
        ]
        
        return img, descrizione
        
    except Exception as e:
        st.error(f"Errore nella generazione: {str(e)}")
        # Immagine di fallback con effetto glitch minimo
        fallback = Image.new('RGB', size, (20, 20, 40))
        return fallback, ["Errore nella generazione"]

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
        # Analisi audio
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
        
        # Sistema di seed migliorato
        base_seed = features['file_hash']
        
        if st.button("🔄 Rigenera Copertina"):
            st.session_state.regen_count += 1
            st.rerun()

        # Seed unico per ogni rigenerazione
        if st.session_state.regen_count == 0:
            seed = base_seed
        else:
            # Combina hash file, timestamp e counter per unicità
            timestamp_seed = int(time.time() * 1000000) % 1000000
            seed = (base_seed + timestamp_seed + st.session_state.regen_count * 12345) % 2147483647

        with st.spinner("🎨 Generando effetti glitch TV..."):
            img, descrizione = generate_glitch_image(features, seed, size=dimensions)

        st.image(img, 
                caption=f"Copertina Glitch TV - {aspect_ratio} (Versione #{st.session_state.regen_count + 1})", 
                use_container_width=True)

        # Dettagli tecnici
        if show_analysis:
            with st.expander("🔧 Dettagli Effetti Glitch"):
                st.write(f"**Seed utilizzato:** {seed}")
                st.write(f"**Rigenerazioni:** {st.session_state.regen_count + 1}")
                st.write("**Effetti applicati basati sull'audio:**")
                for d in descrizione:
                    st.markdown(f"- {d}")

        # Download
        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()

        filename = f"glitch_tv_{features['emotion'].lower()}_{int(time.time())}.{img_format.lower()}"
        st.download_button(
            label=f"⬇️ Scarica Glitch TV {img_format} ({dimensions[0]}x{dimensions[1]})",
            data=byte_im,
            file_name=filename,
            mime=f"image/{img_format.lower()}"
        )
        
        # Info sui risultati
        st.success("✅ **Effetti Glitch TV Applicati:**")
        st.write("• Forma d'onda basata sull'audio")
        st.write("• Aberrazione cromatica")
        st.write("• Shift RGB dinamico")
        st.write("• Datamoshing")
        st.write("• Rumore digitale")
        st.write("• Linee di scansione TV")
        
else:
    st.session_state.regen_count = 0
    st.info("👆 Carica un file audio per iniziare!")
    
    # Mostra anteprima degli effetti
    st.markdown("### 🎯 Effetti Glitch TV Che Otterrai:")
    st.write("❌ **Niente più quadratini ripetitivi**")
    st.write("✅ **Veri effetti glitch TV**")
    st.write("✅ **Ogni brano genera pattern diversi**")
    st.write("✅ **Ogni rigenerazione è unica**")
    st.write("✅ **Effetti basati sulle caratteristiche audio**")
