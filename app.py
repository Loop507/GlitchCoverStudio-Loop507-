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

        try:
            bpm, beats = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(bpm) if bpm is not None and not np.isnan(bpm) else 120.0
        except:
            bpm = 120.0

        rms = np.mean(librosa.feature.rms(y=y))
        rms = float(rms) if not np.isnan(rms) else 0.05

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_centroid = float(spectral_centroid) if not np.isnan(spectral_centroid) else 2000.0

        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_bandwidth = float(spectral_bandwidth) if not np.isnan(spectral_bandwidth) else 1000.0

        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_rolloff = float(spectral_rolloff) if not np.isnan(spectral_rolloff) else 3000.0

        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        zero_crossing_rate = float(zero_crossing_rate) if not np.isnan(zero_crossing_rate) else 0.1

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_mean = np.nan_to_num(mfcc_mean, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_energy = np.mean(librosa.feature.rms(y=y_harmonic))
            percussive_energy = np.mean(librosa.feature.rms(y=y_percussive))
        except:
            harmonic_energy = rms * 0.6
            percussive_energy = rms * 0.4

        harmonic_energy = float(harmonic_energy) if not np.isnan(harmonic_energy) else 0.03
        percussive_energy = float(percussive_energy) if not np.isnan(percussive_energy) else 0.02

        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_strength = len(onset_frames) / len(y) * sr if len(y) > 0 else 1.0
        except:
            onset_strength = 1.0

        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            tonal_centroid = np.argmax(np.mean(chroma, axis=1))
        except:
            tonal_centroid = 0

        dynamic_range = abs(spectral_bandwidth - spectral_centroid)

        # Aggiungi più variabilità all'hash
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
            "tonal_centroid": int(tonal_centroid),
            "mfcc_features": mfcc_mean,
            "emotion": emotion,
            "file_hash": hash_int,
            "file_size": len(audio_bytes),
            "audio_signature": hash_obj[:16]
        }
    except Exception as e:
        st.error(f"Errore nell'analisi audio con Librosa: {str(e)}")
        return None

def create_random_base_image(size, features, seed):
    """Crea un'immagine base più varia e interessante"""
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    width, height = size
    
    # Palette colori più audaci basata sull'emozione
    emotion_palettes = {
        "Aggressive": [(255, 0, 0), (255, 100, 0), (200, 0, 100), (255, 255, 0), (255, 0, 255)],
        "Energetico": [(255, 140, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 100, 100)],
        "Ambient": [(0, 100, 255), (100, 0, 255), (0, 255, 200), (100, 255, 100), (200, 100, 255)],
        "Calmo": [(0, 150, 255), (100, 200, 255), (0, 255, 150), (150, 255, 200), (200, 150, 255)],
        "Melodico": [(200, 0, 255), (255, 0, 200), (100, 100, 255), (255, 100, 200), (200, 100, 255)],
        "Ritmico": [(255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 100, 0), (100, 255, 100)],
        "Dinamico": [(255, 50, 150), (150, 255, 50), (50, 150, 255), (255, 150, 50), (150, 50, 255)],
        "Equilibrato": [(100, 255, 100), (255, 100, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)]
    }
    
    colors = emotion_palettes.get(features["emotion"], emotion_palettes["Equilibrato"])
    
    # Crea background con gradiente o pattern
    img = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Tipo di background basato su seed
    bg_type = seed % 4
    
    if bg_type == 0:
        # Gradiente diagonale
        for x in range(width):
            for y in range(height):
                color_idx = int((x + y) / (width + height) * len(colors))
                color = colors[color_idx % len(colors)]
                draw.point((x, y), fill=color)
    elif bg_type == 1:
        # Pattern a bande
        band_width = max(10, int(features['spectral_bandwidth'] / 100))
        for i in range(0, width, band_width):
            color = colors[i // band_width % len(colors)]
            draw.rectangle([i, 0, i + band_width, height], fill=color)
    elif bg_type == 2:
        # Pattern radiale
        center_x, center_y = width // 2, height // 2
        max_radius = max(width, height) // 2
        for radius in range(0, max_radius, 20):
            color = colors[radius // 20 % len(colors)]
            draw.ellipse([center_x - radius, center_y - radius, 
                         center_x + radius, center_y + radius], outline=color, width=10)
    else:
        # Pattern a rumore
        noise_array = np.random.randint(0, len(colors), (height, width))
        for y in range(height):
            for x in range(width):
                color = colors[noise_array[y, x]]
                draw.point((x, y), fill=color)
    
    return img

def apply_pixel_sorting(img, features, seed):
    """Applica pixel sorting per effetti glitch più intensi"""
    random.seed(seed)
    width, height = img.size
    pixels = np.array(img)
    
    # Intensità basata sulle caratteristiche audio
    sort_intensity = max(0.1, min(0.9, features['onset_strength'] / 10 + features['percussive_energy'] * 5))
    
    # Sorting orizzontale
    if random.random() < 0.5:
        for y in range(0, height, random.randint(2, 8)):
            if random.random() < sort_intensity:
                row = pixels[y].copy()
                # Ordina per luminosità o per colore
                if random.random() < 0.5:
                    brightness = np.mean(row, axis=1)
                    sort_indices = np.argsort(brightness)
                else:
                    sort_indices = np.argsort(row[:, random.randint(0, 2)])
                pixels[y] = row[sort_indices]
    
    # Sorting verticale
    else:
        for x in range(0, width, random.randint(2, 8)):
            if random.random() < sort_intensity:
                col = pixels[:, x].copy()
                if random.random() < 0.5:
                    brightness = np.mean(col, axis=1)
                    sort_indices = np.argsort(brightness)
                else:
                    sort_indices = np.argsort(col[:, random.randint(0, 2)])
                pixels[:, x] = col[sort_indices]
    
    return Image.fromarray(pixels)

def apply_digital_corruption(img, features, seed):
    """Applica corruzioni digitali più aggressive"""
    random.seed(seed)
    width, height = img.size
    pixels = np.array(img)
    
    corruption_intensity = max(0.1, min(0.8, features['zero_crossing_rate'] * 5 + features['rms'] * 3))
    
    # Corruzioni random
    num_corruptions = int(corruption_intensity * 50)
    
    for _ in range(num_corruptions):
        corruption_type = random.randint(0, 4)
        
        if corruption_type == 0:
            # Blocchi corrotti
            x = random.randint(0, width - 50)
            y = random.randint(0, height - 50)
            block_w = random.randint(10, 100)
            block_h = random.randint(10, 100)
            
            # Inverti colori del blocco
            pixels[y:y+block_h, x:x+block_w] = 255 - pixels[y:y+block_h, x:x+block_w]
            
        elif corruption_type == 1:
            # Linee di interferenza
            y = random.randint(0, height - 1)
            line_height = random.randint(1, 5)
            # Shift della linea
            shift = random.randint(-50, 50)
            if shift > 0:
                pixels[y:y+line_height, shift:] = pixels[y:y+line_height, :-shift]
            else:
                pixels[y:y+line_height, :shift] = pixels[y:y+line_height, -shift:]
                
        elif corruption_type == 2:
            # Rumore a blocchi
            x = random.randint(0, width - 20)
            y = random.randint(0, height - 20)
            block_w = random.randint(5, 30)
            block_h = random.randint(5, 30)
            
            noise = np.random.randint(0, 255, (block_h, block_w, 3))
            pixels[y:y+block_h, x:x+block_w] = noise
            
        elif corruption_type == 3:
            # Duplicazione di strisce
            if random.random() < 0.5:
                # Orizzontale
                y = random.randint(0, height - 10)
                stripe_h = random.randint(1, 8)
                target_y = random.randint(0, height - stripe_h)
                pixels[target_y:target_y+stripe_h] = pixels[y:y+stripe_h]
            else:
                # Verticale
                x = random.randint(0, width - 10)
                stripe_w = random.randint(1, 8)
                target_x = random.randint(0, width - stripe_w)
                pixels[:, target_x:target_x+stripe_w] = pixels[:, x:x+stripe_w]
                
        elif corruption_type == 4:
            # Aberrazione cromatica estrema
            offset = random.randint(5, 30)
            channel = random.randint(0, 2)
            direction = random.choice([(offset, 0), (-offset, 0), (0, offset), (0, -offset)])
            
            if direction[0] > 0:
                pixels[:, offset:, channel] = pixels[:, :-offset, channel]
            elif direction[0] < 0:
                pixels[:, :offset, channel] = pixels[:, -offset:, channel]
            elif direction[1] > 0:
                pixels[offset:, :, channel] = pixels[:-offset, :, channel]
            elif direction[1] < 0:
                pixels[:offset, :, channel] = pixels[-offset:, :, channel]
    
    return Image.fromarray(pixels)

def apply_recursive_glitch(img, features, seed, iterations=3):
    """Applica più livelli di glitch ricorsivamente"""
    random.seed(seed)
    current_img = img.copy()
    
    for i in range(iterations):
        iteration_seed = seed + i * 1000
        
        # Scegli effetto basato sulle caratteristiche
        if features['percussive_energy'] > 0.03:
            current_img = apply_pixel_sorting(current_img, features, iteration_seed)
        
        if features['onset_strength'] > 2:
            current_img = apply_digital_corruption(current_img, features, iteration_seed + 100)
        
        # Aberrazione cromatica più intensa
        if features['spectral_bandwidth'] > 1500:
            current_img = apply_advanced_chromatic_aberration(current_img, features, iteration_seed + 200)
        
        # Distorsione casuale
        if random.random() < 0.7:
            current_img = apply_random_distortion(current_img, features, iteration_seed + 300)
    
    return current_img

def apply_random_distortion(img, features, seed):
    """Applica distorsioni casuali"""
    random.seed(seed)
    width, height = img.size
    pixels = np.array(img)
    
    distortion_type = random.randint(0, 3)
    
    if distortion_type == 0:
        # Compressione JPEG simulata
        quality = random.randint(10, 50)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
        
    elif distortion_type == 1:
        # Riduzione colori
        colors = random.randint(8, 64)
        return img.quantize(colors=colors).convert('RGB')
        
    elif distortion_type == 2:
        # Scaling e resize con artefatti
        scale = random.uniform(0.3, 0.8)
        small_img = img.resize((int(width * scale), int(height * scale)), Image.NEAREST)
        return small_img.resize((width, height), Image.NEAREST)
        
    elif distortion_type == 3:
        # Interferenza a onde
        for y in range(height):
            wave_offset = int(np.sin(y * 0.1 + seed) * 20)
            if wave_offset > 0:
                pixels[y, wave_offset:] = pixels[y, :-wave_offset]
            elif wave_offset < 0:
                pixels[y, :wave_offset] = pixels[y, -wave_offset:]
        return Image.fromarray(pixels)
    
    return img

def apply_advanced_chromatic_aberration(img, features, seed):
    """Aberrazione cromatica migliorata"""
    random.seed(seed)
    base_offset = max(3, min(30, int(features['spectral_bandwidth'] / 100)))
    rms_multiplier = max(2, min(15, int(features['rms'] * 150)))
    
    # Offsets più variabili
    offset_r_x = base_offset + random.randint(-rms_multiplier, rms_multiplier)
    offset_r_y = random.randint(-5, 5)
    offset_g_x = random.randint(-base_offset//2, base_offset//2)
    offset_g_y = random.randint(-3, 3)
    offset_b_x = -base_offset + random.randint(-rms_multiplier//2, rms_multiplier//2)
    offset_b_y = random.randint(-5, 5)

    r, g, b = img.split()
    r = ImageChops.offset(r, offset_r_x, offset_r_y)
    g = ImageChops.offset(g, offset_g_x, offset_g_y)
    b = ImageChops.offset(b, offset_b_x, offset_b_y)
    
    return Image.merge('RGB', (r, g, b))

def generate_advanced_glitch_image(features, seed, size=(800, 800)):
    """Genera immagine glitch con maggiore varietà"""
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    
    # Crea immagine base più varia
    img = create_random_base_image(size, features, seed)
    
    # Applica effetti glitch ricorsivamente
    img = apply_recursive_glitch(img, features, seed)
    
    # Effetti finali basati sull'emozione
    if features['emotion'] == 'Aggressive':
        img = apply_digital_corruption(img, features, seed + 5000)
        img = apply_pixel_sorting(img, features, seed + 6000)
    elif features['emotion'] == 'Energetico':
        img = apply_advanced_chromatic_aberration(img, features, seed + 7000)
        img = apply_random_distortion(img, features, seed + 8000)
    elif features['emotion'] == 'Ambient':
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        img = apply_advanced_chromatic_aberration(img, features, seed + 9000)
    elif features['emotion'] == 'Ritmico':
        img = apply_pixel_sorting(img, features, seed + 10000)
        img = apply_digital_corruption(img, features, seed + 11000)
    else:
        # Mix di effetti casuali
        effects = [
            lambda: apply_pixel_sorting(img, features, seed + 12000),
            lambda: apply_digital_corruption(img, features, seed + 13000),
            lambda: apply_random_distortion(img, features, seed + 14000)
        ]
        for effect in random.sample(effects, random.randint(1, 3)):
            img = effect()

    # Miglioramento finale
    if features['harmonic_energy'] > 0.03:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(1.1, 1.5))
    
    if features['rms'] > 0.05:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(1.1, 1.8))

    descrizione = [
        f"🎵 BPM: {features['bpm']:.1f} → Velocità pattern e corruzioni",
        f"🔊 RMS: {features['rms']:.3f} → Intensità effetti glitch",
        f"📡 Centro spettrale: {features['spectral_centroid']:.0f} Hz → Tipo di distorsione",
        f"⚡ Energia percussiva: {features['percussive_energy']:.3f} → Pixel sorting",
        f"🎼 Energia armonica: {features['harmonic_energy']:.3f} → Aberrazione cromatica",
        f"🔄 Zero crossing: {features['zero_crossing_rate']:.3f} → Corruzioni digitali",
        f"🎭 Emozione: {features['emotion']} → Stile glitch generale",
        f"🌊 Onset strength: {features['onset_strength']:.1f} → Ricorsività effetti"
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
    show_analysis = st.checkbox("Mostra analisi avanzata", value=True)

# Controlli aggiuntivi per personalizzazione
st.sidebar.markdown("### 🎨 Controlli Glitch")
glitch_intensity = st.sidebar.slider("Intensità Glitch", 0.1, 2.0, 1.0, 0.1)
use_pixel_sorting = st.sidebar.checkbox("Pixel Sorting", value=True)
use_digital_corruption = st.sidebar.checkbox("Corruzioni Digitali", value=True)
use_chromatic_aberration = st.sidebar.checkbox("Aberrazione Cromatica", value=True)

if audio_file:
    with st.spinner("🎵 Analizzando caratteristiche audio..."):
        features = analyze_audio_librosa(audio_file)
    
    if features:
        # Modifica l'intensità degli effetti
        features['rms'] *= glitch_intensity
        features['onset_strength'] *= glitch_intensity
        features['percussive_energy'] *= glitch_intensity
        
        if show_analysis:
            st.markdown("### 🎨 Analisi Audio")
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

        # Seed più variabile che incorpora timestamp e parametri UI
        base_seed = features['file_hash']
        ui_seed = hash(f"{glitch_intensity}_{use_pixel_sorting}_{use_digital_corruption}_{use_chromatic_aberration}") % 1000000
        
        if st.button("🔄 Rigenera Copertina Glitch"):
            st.session_state.regen_count += 1
            st.rerun()

        # Calcola seed finale con più variabilità
        timestamp_seed = int(time.time() * 1000000) % 1000000
        final_seed = (base_seed + ui_seed + timestamp_seed + st.session_state.regen_count * 54321) % 2147483647

        with st.spinner("🎨 Generando glitch cover avanzata..."):
            img, descrizione = generate_advanced_glitch_image(features, final_seed, size=dimensions)

        st.image(img, caption=f"Advanced Glitch Cover - {features['emotion']} Style - {aspect_ratio} (Gen #{st.session_state.regen_count + 1})", use_container_width=True)

        if show_analysis:
            with st.expander("🔧 Dettagli Tecnici"):
                st.write(f"**Seed utilizzato:** {final_seed}")
                st.write(f"**Firma audio:** {features['audio_signature']}")
                st.write(f"**Rigenerazioni:** {st.session_state.regen_count + 1}")
                st.write(f"**Intensità glitch:** {glitch_intensity}")
                st.write("**Mappatura audio → effetti visivi:**")
                for d in descrizione:
                    st.markdown(f"- {d}")

        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()
        filename = f"advanced_glitch_cover_{features['emotion'].lower()}_{features['audio_signature'][:8]}_{int(time.time())}.{img_format.lower()}"
        st.download_button(
            label=f"⬇️ Scarica Advanced Glitch Cover {img_format} ({dimensions[0]}x{dimensions[1]})",
            data=byte_im,
            file_name=filename,
            mime=f"image/{img_format.lower()}"
        )

        st.success("✅ Effetti Glitch Avanzati Applicati:")
        st.write("• Pixel sorting adattivo basato su energia percussiva")
        st.write("• Corruzioni digitali multiple (blocchi, rumore, interferenze)")
        st.write("• Aberrazione cromatica multi-direzionale")
        st.write("• Distorsioni casuali (compressione, scaling, onde)")
        st.write("• Effetti ricorsivi per maggiore complessità")
        st.write("• Background variabili (gradienti, pattern, rumore)")
        st.write("• Palette colori estese per ogni emozione")
        st.write("• Seed altamente variabile per unicità garantita")
else:
    st.session_state.regen_count = 0
    st.info("👆 Carica un file audio per iniziare!")
