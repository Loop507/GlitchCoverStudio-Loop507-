import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageChops, ImageFilter, ImageEnhance
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

        # Analisi più dettagliata
        bpm, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(bpm) if bpm is not None else 60.0

        # Caratteristiche audio più ricche
        rms = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Analisi armonica/percussiva
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.mean(librosa.feature.rms(y=y_harmonic))
        percussive_energy = np.mean(librosa.feature.rms(y=y_percussive))
        
        # Onset detection per ritmo
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_strength = len(onset_frames) / len(y) * sr
        
        # Analisi tonale
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tonal_centroid = np.argmax(np.mean(chroma, axis=1))
        
        dynamic_range = spectral_bandwidth - spectral_centroid
        
        # Hash più robusto basato su più caratteristiche
        feature_string = f"{bpm:.2f}_{rms:.4f}_{spectral_centroid:.0f}_{len(audio_bytes)}"
        hash_obj = hashlib.sha256((feature_string + str(audio_bytes[:10000])).encode()).hexdigest()
        hash_int = int(hash_obj[:8], 16)

        # Classificazione emotiva più accurata
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
            "bpm": bpm,
            "rms": rms,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "zero_crossing_rate": zero_crossing_rate,
            "dynamic_range": dynamic_range,
            "harmonic_energy": harmonic_energy,
            "percussive_energy": percussive_energy,
            "onset_strength": onset_strength,
            "tonal_centroid": tonal_centroid,
            "mfcc_features": mfcc_mean,
            "emotion": emotion,
            "file_hash": hash_int,
            "file_size": len(audio_bytes),
            "audio_signature": hash_obj[:16]  # Firma audio unica
        }
    except Exception as e:
        st.error(f"Errore nell'analisi audio con Librosa: {str(e)}")
        return None

def create_advanced_noise_pattern(size, features, seed):
    """Crea pattern di rumore avanzato basato sull'audio"""
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    width, height = size
    
    # Crea pattern di base influenzato dall'audio
    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Pattern basato su MFCC
    for i, mfcc_val in enumerate(features['mfcc_features'][:8]):
        intensity = int(abs(mfcc_val) * 50) + 10
        frequency = int(features['spectral_centroid'] / 100) + i
        
        # Genera onde sinusoidali colorate
        for y in range(height):
            for x in range(width):
                wave_val = np.sin(x * frequency / 100 + y * intensity / 200 + mfcc_val)
                color_intensity = int((wave_val + 1) * 127)
                
                # Colore basato sull'indice MFCC
                if i % 3 == 0:
                    pattern[y, x, 0] = min(255, pattern[y, x, 0] + color_intensity)
                elif i % 3 == 1:
                    pattern[y, x, 1] = min(255, pattern[y, x, 1] + color_intensity)
                else:
                    pattern[y, x, 2] = min(255, pattern[y, x, 2] + color_intensity)
    
    return Image.fromarray(pattern, 'RGB')

def create_frequency_visualization(features, size, seed):
    """Crea visualizzazione basata sulle frequenze"""
    random.seed(seed)
    width, height = size
    
    # Palette colori dinamica basata sull'emozione e caratteristiche audio
    base_hue = int(features['tonal_centroid'] * 30) % 360
    
    emotion_palettes = {
        "Aggressive": [(255, 0, 0), (255, 100, 0), (200, 0, 100)],
        "Energetico": [(255, 0, 150), (0, 255, 100), (100, 0, 255)],
        "Ambient": [(0, 150, 255), (100, 0, 200), (0, 200, 150)],
        "Calmo": [(0, 100, 255), (150, 0, 255), (0, 255, 200)],
        "Melodico": [(200, 100, 255), (255, 150, 0), (100, 255, 150)],
        "Ritmico": [(255, 0, 255), (0, 255, 255), (255, 255, 0)],
        "Dinamico": [(255, 0, 255), (0, 255, 255), (255, 255, 0)],
        "Equilibrato": [(150, 0, 255), (255, 0, 150), (0, 255, 150)]
    }
    
    colors = emotion_palettes.get(features["emotion"], emotion_palettes["Equilibrato"])
    
    # Modifica colori in base alle caratteristiche audio
    modified_colors = []
    for color in colors:
        r, g, b = color
        r = int(r * (1 + features['harmonic_energy'] * 2)) % 256
        g = int(g * (1 + features['percussive_energy'] * 3)) % 256
        b = int(b * (1 + features['rms'] * 5)) % 256
        modified_colors.append((r, g, b))
    
    # Crea base con gradiente
    img = Image.new('RGB', size, (5, 5, 15))
    draw = ImageDraw.Draw(img)
    
    # Disegna bande di frequenza
    num_bands = min(int(features['spectral_rolloff'] / 1000), 12)
    band_height = height // (num_bands + 1)
    
    for band in range(num_bands):
        y_start = band * band_height
        y_end = (band + 1) * band_height
        
        # Intensità basata su caratteristiche specifiche
        intensity = features['mfcc_features'][band % len(features['mfcc_features'])]
        wave_amplitude = int(abs(intensity) * 100) + 20
        
        color = modified_colors[band % len(modified_colors)]
        
        # Disegna forma d'onda per questa banda
        points = []
        for x in range(0, width, 2):
            # Forma d'onda complessa basata su multiple caratteristiche
            wave1 = np.sin(x * features['bpm'] / 500 + band * np.pi / 4)
            wave2 = np.cos(x * features['onset_strength'] / 50 + intensity)
            wave3 = np.sin(x * features['zero_crossing_rate'] * 1000)
            
            combined_wave = (wave1 + wave2 * 0.5 + wave3 * 0.3) * wave_amplitude
            
            y = int(y_start + band_height // 2 + combined_wave)
            y = max(y_start, min(y_end - 1, y))
            points.append((x, y))
        
        # Disegna linea spessa per la forma d'onda
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=random.randint(2, 5))
    
    # Aggiungi elementi geometrici basati sull'audio
    num_elements = int(features['onset_strength'] * 2) + 3
    for _ in range(num_elements):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size_elem = int(features['rms'] * 200) + random.randint(10, 50)
        
        color = random.choice(modified_colors)
        
        # Forme diverse basate sull'energia percussiva
        if features['percussive_energy'] > 0.02:
            # Rettangoli per musica percussiva
            draw.rectangle([x - size_elem//2, y - size_elem//2, 
                          x + size_elem//2, y + size_elem//2], 
                         outline=color, width=random.randint(2, 6))
        else:
            # Cerchi per musica melodica
            draw.ellipse([x - size_elem//2, y - size_elem//2, 
                        x + size_elem//2, y + size_elem//2], 
                       outline=color, width=random.randint(2, 6))
    
    return img

def apply_advanced_chromatic_aberration(img, features, seed):
    """Aberrazione cromatica avanzata basata sull'audio"""
    random.seed(seed)
    
    # Offset basati sulle caratteristiche audio
    base_offset = int(features['spectral_bandwidth'] / 100)
    rms_multiplier = int(features['rms'] * 200)
    
    offset_r = base_offset + random.randint(-rms_multiplier, rms_multiplier)
    offset_g = random.randint(-base_offset//2, base_offset//2)
    offset_b = -base_offset + random.randint(-rms_multiplier//2, rms_multiplier//2)
    
    # Aberrazione verticale per musica più percussiva
    if features['percussive_energy'] > features['harmonic_energy']:
        offset_r_y = random.randint(-5, 5)
        offset_b_y = random.randint(-5, 5)
    else:
        offset_r_y = 0
        offset_b_y = 0
    
    r, g, b = img.split()
    
    # Applica offset con componente verticale
    r = ImageChops.offset(r, offset_r, offset_r_y)
    g = ImageChops.offset(g, offset_g, 0)
    b = ImageChops.offset(b, offset_b, offset_b_y)
    
    return Image.merge('RGB', (r, g, b))

def apply_audio_driven_datamoshing(img, features, seed):
    """Datamoshing avanzato guidato dall'audio"""
    random.seed(seed)
    width, height = img.size
    
    # Intensità basata su onset e percussività
    glitch_intensity = int(features['onset_strength'] * 50) + int(features['percussive_energy'] * 100)
    glitch_intensity = min(glitch_intensity, 25)  # Limita per evitare effetti troppo estremi
    
    # Tipo di glitch basato sull'energia
    if features['percussive_energy'] > 0.03:
        # Glitch verticali per musica percussiva
        for _ in range(random.randint(3, glitch_intensity)):
            x_pos = random.randint(0, width - 1)
            strip_width = random.randint(1, 10)
            
            if x_pos + strip_width < width:
                strip = img.crop((x_pos, 0, x_pos + strip_width, height))
                
                # Sposta la striscia
                new_x = max(0, min(width - strip_width, x_pos + random.randint(-100, 100)))
                img.paste(strip, (new_x, 0))
    else:
        # Glitch orizzontali per musica melodica
        for _ in range(random.randint(3, glitch_intensity)):
            y_pos = random.randint(0, height - 1)
            strip_height = random.randint(1, 8)
            
            if y_pos + strip_height < height:
                strip = img.crop((0, y_pos, width, y_pos + strip_height))
                
                # Sposta la striscia
                new_y = max(0, min(height - strip_height, y_pos + random.randint(-50, 50)))
                img.paste(strip, (0, new_y))
    
    # Aggiungi "pixel sorting" basato su BPM
    if features['bpm'] > 100:
        num_sorts = int(features['bpm'] / 20)
        for _ in range(min(num_sorts, 8)):
            x1 = random.randint(0, width // 2)
            y1 = random.randint(0, height // 2)
            x2 = x1 + random.randint(50, 200)
            y2 = y1 + random.randint(10, 50)
            
            if x2 < width and y2 < height:
                section = img.crop((x1, y1, x2, y2))
                # Simula pixel sorting con spostamento
                section = ImageChops.offset(section, random.randint(-10, 10), 0)
                img.paste(section, (x1, y1))
    
    return img

def apply_spectral_noise(img, features, seed):
    """Rumore basato sullo spettro audio"""
    random.seed(seed)
    width, height = img.size
    
    # Crea pattern di rumore basato sulle caratteristiche spettrali
    noise_pattern = create_advanced_noise_pattern((width, height), features, seed + 1000)
    
    # Intensità del rumore basata su zero crossing rate
    noise_intensity = min(features['zero_crossing_rate'] * 2, 0.6)
    
    # Blend con modalità diverse basate sull'emozione
    blend_modes = {
        "Aggressive": "multiply",
        "Energetico": "screen",
        "Ambient": "overlay",
        "Calmo": "soft_light",
        "Melodico": "overlay",
        "Ritmico": "difference",
        "Dinamico": "hard_light",
        "Equilibrato": "normal"
    }
    
    # Applica il rumore con intensità variabile
    img_array = np.array(img)
    noise_array = np.array(noise_pattern)
    
    # Mixing basato sull'intensità
    mixed = img_array * (1 - noise_intensity) + noise_array * noise_intensity
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)
    
    return Image.fromarray(mixed)

def apply_dynamic_scanlines(img, features, seed):
    """Linee di scansione dinamiche"""
    random.seed(seed)
    width, height = img.size
    
    # Spaziatura basata su BPM
    base_spacing = max(2, int(8 - features['bpm'] / 20))
    
    # Crea pattern di scanline variabile
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Linee principali
    for y in range(0, height, base_spacing):
        opacity = int(100 + features['rms'] * 500) % 150
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, opacity), width=1)
    
    # Linee secondarie basate sull'energia percussiva
    if features['percussive_energy'] > 0.02:
        secondary_spacing = base_spacing * 2
        for y in range(base_spacing, height, secondary_spacing):
            opacity = int(50 + features['percussive_energy'] * 1000) % 100
            draw.line([(0, y), (width, y)], fill=(0, 0, 0, opacity), width=1)
    
    # Linee verticali per musica con alto zero crossing rate
    if features['zero_crossing_rate'] > 0.1:
        vertical_spacing = int(width / 20)
        for x in range(0, width, vertical_spacing):
            opacity = int(30 + features['zero_crossing_rate'] * 500) % 80
            draw.line([(x, 0), (x, height)], fill=(0, 0, 0, opacity), width=1)
    
    # Blend con l'immagine
    img = img.convert('RGBA')
    result = Image.alpha_composite(img, overlay)
    return result.convert('RGB')

def generate_advanced_glitch_image(features, seed, size=(800, 800)):
    """Genera immagine glitch avanzata basata su audio"""
    try:
        # Seed unico per ogni componente
        random.seed(seed)
        np.random.seed(seed % 2147483647)
        
        # 1. Crea base dalla visualizzazione delle frequenze
        img = create_frequency_visualization(features, size, seed)
        
        # 2. Applica filtri di miglioramento basati sull'audio
        if features['harmonic_energy'] > 0.03:
            # Aumenta contrasto per musica armonica
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
        
        if features['rms'] > 0.05:
            # Aumenta saturazione per musica ad alta energia
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.2)
        
        # 3. Applica effetti glitch in sequenza con seed diversi
        
        # Aberrazione cromatica avanzata
        img = apply_advanced_chromatic_aberration(img, features, seed + 1000)
        
        # Datamoshing guidato dall'audio
        img = apply_audio_driven_datamoshing(img, features, seed + 2000)
        
        # Rumore spettrale
        img = apply_spectral_noise(img, features, seed + 3000)
        
        # Scanline dinamiche
        img = apply_dynamic_scanlines(img, features, seed + 4000)
        
        # Effetto finale basato sull'emozione
        if features['emotion'] == 'Aggressive':
            # Shift RGB aggressivo
            img = apply_advanced_chromatic_aberration(img, features, seed + 5000)
        elif features['emotion'] == 'Ambient':
            # Blur leggero per musica ambient
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        elif features['emotion'] == 'Ritmico':
            # Effetto "posterize" per musica ritmica
            img = img.quantize(colors=16).convert('RGB')
        
        # Genera descrizione dettagliata
        descrizione = [
            f"🎵 BPM: {features['bpm']:.1f} → Intensità e velocità glitch",
            f"🔊 RMS: {features['rms']:.3f} → Saturazione e contrasto",
            f"📡 Centro spettrale: {features['spectral_centroid']:.0f} Hz → Pattern frequenze",
            f"⚡ Energia percussiva: {features['percussive_energy']:.3f} → Tipo di glitch",
            f"🎼 Energia armonica: {features['harmonic_energy']:.3f} → Forme e colori",
            f"🎭 Emozione: {features['emotion']} → Palette e modalità blend",
            f"🔄 Zero crossing: {features['zero_crossing_rate']:.3f} → Dettagli rumore",
            f"🎯 Onset strength: {features['onset_strength']:.1f} → Elementi geometrici"
        ]
        
        return img, descrizione
        
    except Exception as e:
        st.error(f"Errore nella generazione avanzata: {str(e)}")
        # Fallback più creativo
        fallback = Image.new('RGB', size, (random.randint(20, 80), 
                                          random.randint(20, 80), 
                                          random.randint(20, 80)))
        return fallback, ["Errore nella generazione avanzata"]

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
    show_analysis = st.checkbox("Mostra analisi avanzata", value=True)

if audio_file:
    with st.spinner("🎵 Analizzando caratteristiche audio avanzate..."):
        features = analyze_audio_librosa(audio_file)
    
    if features:
        # Analisi audio dettagliata
        if show_analysis:
            st.markdown("### 🎨 Analisi Audio Avanzata")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("🎵 BPM", f"{features['bpm']:.1f}")
                st.metric("🔊 RMS", f"{features['rms']:.3f}")
                st.metric("🎭 Emozione", features['emotion'])
            
            with col_b:
                st.metric("📡 Centro spettrale", f"{features['spectral_centroid']:.0f} Hz")
                st.metric("⚡ Energia percussiva", f"{features['percussive_energy']:.3f}")
                st.metric("🎼 Energia armonica", f"{features['harmonic_energy']:.3f}")
            
            with col_c:
                st.metric("🔄 Zero crossing", f"{features['zero_crossing_rate']:.3f}")
                st.metric("🎯 Onset strength", f"{features['onset_strength']:.1f}")
                st.metric("🎹 Centro tonale", f"{features['tonal_centroid']}")
        
        dimensions = get_dimensions(aspect_ratio)
        
        # Sistema di seed migliorato con più entropia
        base_seed = features['file_hash']
        audio_signature = int(features['audio_signature'], 16)
        
        if st.button("🔄 Rigenera Copertina Glitch"):
            st.session_state.regen_count += 1
            st.rerun()

        # Seed unico che incorpora più variabili
        if st.session_state.regen_count == 0:
            seed = base_seed
        else:
            # Combina hash, timestamp, caratteristiche audio e counter
            timestamp_seed = int(time.time() * 1000000) % 1000000
            audio_variations = int(features['rms'] * 1000000) + int(features['spectral_centroid'])
            seed = (base_seed + audio_signature + timestamp_seed + 
                   audio_variations + st.session_state.regen_count * 54321) % 2147483647

        with st.spinner("🎨 Generando effetti glitch avanzati..."):
            img, descrizione = generate_advanced_glitch_image(features, seed, size=dimensions)

        st.image(img, 
                caption=f"Glitch Cover - {features['emotion']} Style - {aspect_ratio} (Gen #{st.session_state.regen_count + 1})", 
                use_container_width=True)

        # Dettagli tecnici avanzati
        if show_analysis:
            with st.expander("🔧 Dettagli Tecnici Avanzati"):
                st.write(f"**Seed utilizzato:** {seed}")
                st.write(f"**Firma audio:** {features['audio_signature']}")
                st.write(f"**Rigenerazioni:** {st.session_state.regen_count + 1}")
                st.write("**Mappatura audio → effetti visivi:**")
                for d in descrizione:
                    st.markdown(f"- {d}")
                
                st.write("**Caratteristiche MFCC utilizzate:**")
                for i, mfcc in enumerate(features['mfcc_features'][:6]):
                    st.write(f"MFCC {i+1}: {mfcc:.3f}")

        # Download
        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()

        filename = f"glitch_cover_{features['emotion'].lower()}_{features['audio_signature'][:8]}_{int(time.time())}.{img_format.lower()}"
        st.download_button(
            label=f"⬇️ Scarica Glitch Cover {img_format} ({dimensions[0]}x{dimensions[1]})",
            data=byte_im,
            file_name=filename,
            mime=f"image/{img_format.lower()}"
        )
        
        # Info migliorata
        st.success("✅ **Effetti Glitch Avanzati Applicati:**")
        st.write("• Visualizzazione frequenze basata su MFCC")
        st.write("• Aberrazione cromatica guidata dallo spettro")
        st.write("• Datamoshing adattivo (orizzontale/verticale)")
        st.write("• Rumore spettrale dinamico")
        st.write("• Scanline responsive al ritmo")
        st.write("• Palette colori emotiva")
        st.write("• Miglioramenti contrast/saturazione")
        st.write("• Effetti finali basati sul genere musicale")
        
else:
    st.session_state.regen_count = 0
    st.info("👆 Carica un file audio per iniziare!")
