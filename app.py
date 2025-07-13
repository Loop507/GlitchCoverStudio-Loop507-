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

def create_advanced_noise_pattern(size, features, seed):
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    width, height = size
    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    mfcc_features = features['mfcc_features'][:8]

    for i, mfcc_val in enumerate(mfcc_features):
        mfcc_val = max(-10, min(10, mfcc_val))
        intensity = max(10, min(100, int(abs(mfcc_val) * 50) + 10))
        frequency = max(1, int(features['spectral_centroid'] / 1000) + i + 1)

        for y in range(0, height, 2):
            for x in range(0, width, 2):
                wave_val = np.sin(x * frequency / 100.0 + y * intensity / 200.0 + mfcc_val)
                color_intensity = int((wave_val + 1) * 127)
                color_intensity = max(0, min(255, color_intensity))

                if i % 3 == 0:
                    pattern[y, x, 0] = min(255, pattern[y, x, 0] + color_intensity)
                elif i % 3 == 1:
                    pattern[y, x, 1] = min(255, pattern[y, x, 1] + color_intensity)
                else:
                    pattern[y, x, 2] = min(255, pattern[y, x, 2] + color_intensity)

    return Image.fromarray(pattern, 'RGB')

def create_frequency_visualization(features, size, seed):
    random.seed(seed)
    width, height = size

    emotion_palettes = {
        "Aggressive": [(255, 0, 0), (255, 69, 0), (220, 20, 60)],
        "Energetico": [(255, 140, 0), (255, 215, 0), (255, 69, 0)],
        "Ambient": [(0, 191, 255), (70, 130, 180), (100, 149, 237)],
        "Calmo": [(30, 144, 255), (135, 206, 235), (70, 130, 180)],
        "Melodico": [(148, 0, 211), (75, 0, 130), (238, 130, 238)],
        "Ritmico": [(255, 0, 255), (0, 255, 255), (255, 255, 0)],
        "Dinamico": [(255, 20, 147), (255, 105, 180), (255, 192, 203)],
        "Equilibrato": [(0, 255, 0), (0, 255, 255), (255, 255, 0)]
    }

    colors = emotion_palettes.get(features["emotion"], emotion_palettes["Equilibrato"])
    modified_colors = []
    for color in colors:
        r, g, b = color
        r = max(0, min(255, int(r * (1 + features['harmonic_energy'] * 2))))
        g = max(0, min(255, int(g * (1 + features['percussive_energy'] * 3))))
        b = max(0, min(255, int(b * (1 + features['rms'] * 5))))
        modified_colors.append((r, g, b))

    img = Image.new('RGB', size, (10, 10, 25))
    draw = ImageDraw.Draw(img)

    num_bands = max(3, min(int(features['spectral_rolloff'] / 1000), 12))
    band_height = height // (num_bands + 1)

    for band in range(num_bands):
        y_start = band * band_height
        y_end = min(height, (band + 1) * band_height)
        mfcc_idx = band % len(features['mfcc_features'])
        intensity = features['mfcc_features'][mfcc_idx]
        intensity = max(-10, min(10, intensity))
        wave_amplitude = max(10, min(100, int(abs(intensity) * 50) + 20))
        color = modified_colors[band % len(modified_colors)]

        points = []
        for x in range(0, width, 4):
            try:
                bpm_val = float(features.get('bpm', 120))
                wave1 = np.sin(x * bpm_val / 1000.0 + band * np.pi / 4)
                wave2 = np.cos(x * features['onset_strength'] / 100.0 + intensity)
                combined_wave = (wave1 + wave2 * 0.5) * wave_amplitude
                y = int(y_start + band_height // 2 + combined_wave)
                y = max(y_start, min(y_end - 1, y))
                points.append((x, y))
            except:
                continue

        if len(points) > 1:
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=color, width=max(1, random.randint(2, 4)))

    num_elements = max(3, min(int(features['onset_strength'] * 2) + 3, 15))
    for _ in range(num_elements):
        try:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            size_elem = max(10, min(80, int(features['rms'] * 200) + random.randint(10, 50)))
            color = random.choice(modified_colors)

            x1 = max(0, x - size_elem // 2)
            y1 = max(0, y - size_elem // 2)
            x2 = min(width, x + size_elem // 2)
            y2 = min(height, y + size_elem // 2)

            if features['percussive_energy'] > 0.02:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=random.randint(2, 4))
            else:
                draw.ellipse([x1, y1, x2, y2], outline=color, width=random.randint(2, 4))
        except:
            continue

    return img

def apply_advanced_chromatic_aberration(img, features, seed):
    random.seed(seed)
    base_offset = max(1, min(20, int(features['spectral_bandwidth'] / 200)))
    rms_multiplier = max(1, min(10, int(features['rms'] * 100)))
    offset_r = base_offset + random.randint(-rms_multiplier, rms_multiplier)
    offset_g = random.randint(-base_offset//2, base_offset//2)
    offset_b = -base_offset + random.randint(-rms_multiplier//2, rms_multiplier//2)

    if features['percussive_energy'] > features['harmonic_energy']:
        offset_r_y = random.randint(-3, 3)
        offset_b_y = random.randint(-3, 3)
    else:
        offset_r_y = 0
        offset_b_y = 0

    r, g, b = img.split()
    r = ImageChops.offset(r, offset_r, offset_r_y)
    g = ImageChops.offset(g, offset_g, 0)
    b = ImageChops.offset(b, offset_b, offset_b_y)
    return Image.merge('RGB', (r, g, b))

def apply_audio_driven_datamoshing(img, features, seed):
    random.seed(seed)
    width, height = img.size
    glitch_intensity = max(3, min(20, int(features['onset_strength'] * 15) + int(features['percussive_energy'] * 70)))

    if features['percussive_energy'] > 0.02:
        for _ in range(random.randint(2, glitch_intensity)):
            x_pos = random.randint(0, width - 10)
            strip_width = random.randint(1, 8)
            if x_pos + strip_width < width:
                strip = img.crop((x_pos, 0, x_pos + strip_width, height))
                new_x = max(0, min(width - strip_width, x_pos + random.randint(-30, 30)))
                img.paste(strip, (new_x, 0))
    else:
        for _ in range(random.randint(2, glitch_intensity)):
            y_pos = random.randint(0, height - 10)
            strip_height = random.randint(1, 6)
            if y_pos + strip_height < height:
                strip = img.crop((0, y_pos, width, y_pos + strip_height))
                new_y = max(0, min(height - strip_height, y_pos + random.randint(-20, 20)))
                img.paste(strip, (0, new_y))
    return img

def generate_advanced_glitch_image(features, seed, size=(800, 800)):
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    img = create_frequency_visualization(features, size, seed)

    if features['harmonic_energy'] > 0.03:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
    if features['rms'] > 0.05:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)

    img = apply_advanced_chromatic_aberration(img, features, seed + 1000)
    img = apply_audio_driven_datamoshing(img, features, seed + 2000)

    if features['emotion'] == 'Aggressive':
        img = apply_advanced_chromatic_aberration(img, features, seed + 3000)
    elif features['emotion'] == 'Ambient':
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    elif features['emotion'] == 'Ritmico':
        img = img.quantize(colors=32).convert('RGB')

    descrizione = [
        f"🎵 BPM: {features['bpm']:.1f} → Velocità effetti",
        f"🔊 RMS: {features['rms']:.3f} → Intensità colori",
        f"📡 Centro spettrale: {features['spectral_centroid']:.0f} Hz → Pattern",
        f"⚡ Energia percussiva: {features['percussive_energy']:.3f} → Tipo glitch",
        f"🎼 Energia armonica: {features['harmonic_energy']:.3f} → Contrasto",
        f"🎭 Emozione: {features['emotion']} → Stile generale"
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

if audio_file:
    with st.spinner("🎵 Analizzando caratteristiche audio..."):
        features = analyze_audio_librosa(audio_file)
    if features:
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

        base_seed = features['file_hash']
        if st.button("🔄 Rigenera Copertina Glitch"):
            st.session_state.regen_count += 1
            st.rerun()

        if st.session_state.regen_count == 0:
            seed = base_seed
        else:
            timestamp_seed = int(time.time() * 1000000) % 1000000
            seed = (base_seed + timestamp_seed + st.session_state.regen_count * 12345) % 2147483647

        with st.spinner("🎨 Generando glitch cover..."):
            img, descrizione = generate_advanced_glitch_image(features, seed, size=dimensions)

        st.image(img, caption=f"Glitch Cover - {features['emotion']} Style - {aspect_ratio} (Gen #{st.session_state.regen_count + 1})", use_container_width=True)

        if show_analysis:
            with st.expander("🔧 Dettagli Tecnici"):
                st.write(f"**Seed utilizzato:** {seed}")
                st.write(f"**Firma audio:** {features['audio_signature']}")
                st.write(f"**Rigenerazioni:** {st.session_state.regen_count + 1}")
                st.write("**Mappatura audio → effetti visivi:**")
                for d in descrizione:
                    st.markdown(f"- {d}")

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

        st.success("✅ Effetti Glitch Applicati:")
        st.write("• Visualizzazione frequenze basata su analisi audio")
        st.write("• Aberrazione cromatica adattiva")
        st.write("• Datamoshing responsive al contenuto")
        st.write("• Palette colori emotiva")
        st.write("• Effetti personalizzati per genere musicale")
else:
    st.session_state.regen_count = 0
    st.info("👆 Carica un file audio per iniziare!")
