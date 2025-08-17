import io
import numpy as np
import streamlit as st
import soundfile as sf
import librosa
from librosa.effects import time_stretch as lb_time_stretch
from librosa.effects import pitch_shift as lb_pitch_shift

# -----------------------------
# Constantes y configuraciones
# -----------------------------
MAJOR_KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
DEFAULT_SR = 44100

# -----------------------------
# Funciones mejoradas
# -----------------------------
def load_audio(file, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    try:
        y, sr = librosa.load(file, sr=sr, mono=True)
        return y.astype(np.float32), sr
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return np.array([], dtype=np.float32), sr

def estimate_bpm(y: np.ndarray, sr: int) -> float:
    try:
        # Usamos start_bpm para evitar el error NoneType
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)
        return float(tempo) if tempo and tempo > 0 else 120.0
    except:
        return 120.0

def estimate_key(y: np.ndarray, sr: int) -> tuple[str, int]:
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        profile = chroma.mean(axis=1)
        
        major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        
        corr_major = np.array([np.dot(np.roll(profile, -k), major_profile) for k in range(12)])
        corr_minor = np.array([np.dot(np.roll(profile, -k), minor_profile) for k in range(12)])
        
        if corr_major.max() >= corr_minor.max():
            pc = int(np.argmax(corr_major))
            return f"{MAJOR_KEYS[pc]} Major", pc
        else:
            pc = int(np.argmax(corr_minor))
            return f"{MAJOR_KEYS[pc]} Minor", pc
    except:
        return "C Major", 0

def semitones_to_shift(src_pc: int, tgt_pc: int) -> int:
    d = tgt_pc - src_pc
    if d > 6: d -= 12
    if d < -6: d += 12
    return int(d)

def apply_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    if rate <= 0 or np.isclose(rate, 1.0): 
        return y
    try:
        return lb_time_stretch(y, rate=rate).astype(np.float32)
    except Exception:
        # Fallback usando phase vocoder
        hop_length = 512
        stft = librosa.stft(y, hop_length=hop_length)
        stft_stretch = librosa.phase_vocoder(stft, rate=rate, hop_length=hop_length)
        return librosa.istft(stft_stretch, hop_length=hop_length).astype(np.float32)

def apply_pitch_shift(y: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    if abs(n_semitones) <= 1e-6: 
        return y
    try:
        return lb_pitch_shift(y, sr=sr, n_steps=n_semitones).astype(np.float32)
    except Exception:
        # Fallback: resample + time stretch
        factor = 2.0 ** (n_semitones / 12.0)
        y_shifted = librosa.resample(y, orig_sr=sr, target_sr=int(sr * factor))
        return apply_time_stretch(y_shifted, rate=1/factor)

def align_to_target(y: np.ndarray, sr: int, bpm: float, key_pc: int,
                    target_bpm: float, target_key_pc: int | None,
                    do_time: bool, do_pitch: bool) -> np.ndarray:
    x = y
    if do_time and bpm > 0:
        rate = float(target_bpm) / float(bpm)
        x = apply_time_stretch(x, rate)
    if do_pitch and target_key_pc is not None:
        shift = semitones_to_shift(key_pc, target_key_pc)
        x = apply_pitch_shift(x, sr, shift)
    return x

def find_beat_position(y: np.ndarray, sr: int, start_time: float) -> float:
    """Encuentra el beat más cercano al tiempo especificado"""
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=120)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Buscar el beat más cercano al tiempo de inicio
        closest_beat = min(beat_times, key=lambda x: abs(x - start_time))
        return closest_beat
    except:
        return start_time

def beat_sync_crossfade(a: np.ndarray, b: np.ndarray, sr: int, 
                        a_start: float, b_start: float, 
                        overlap_beats: int = 8) -> np.ndarray:
    """Mezcla con sincronización de beats para transiciones profesionales"""
    # 1. Encontrar posición exacta de beat para alineación
    a_beat_start = find_beat_position(a, sr, a_start)
    b_beat_start = find_beat_position(b, sr, b_start)
    
    # 2. Convertir tiempos a muestras
    a_start_samp = int(a_beat_start * sr)
    b_start_samp = int(b_beat_start * sr)
    
    # 3. Calcular duración de mezcla basada en beats
    tempo = estimate_bpm(a, sr)
    beat_duration = 60.0 / tempo
    overlap_duration = overlap_beats * beat_duration
    overlap_samples = int(overlap_duration * sr)
    
    # 4. Asegurar que tenemos suficiente audio
    n = min(overlap_samples, len(a) - a_start_samp, len(b) - b_start_samp)
    if n <= 0:
        return np.concatenate([a, b])
    
    # 5. Crear curvas de fade no lineales (sqrt para transición más natural)
    t = np.linspace(0, 1, n)
    fade_out = np.sqrt(1 - t)  # Curva descendente no lineal
    fade_in = np.sqrt(t)       # Curva ascendente no lineal
    
    # 6. Aplicar mezcla
    a_seg = a[a_start_samp:a_start_samp + n]
    b_seg = b[b_start_samp:b_start_samp + n]
    
    mixed_overlap = a_seg * fade_out + b_seg * fade_in
    
    # 7. Construir el resultado final
    return np.concatenate([
        a[:a_start_samp],
        mixed_overlap,
        b[b_start_samp + n:]
    ])

def write_wav_to_bytes(y: np.ndarray, sr: int) -> bytes:
    """Escribe el audio a bytes con normalización"""
    # Normalización para evitar clipping
    peak = np.max(np.abs(y))
    if peak > 0.99:
        y = y * (0.99 / peak)
    
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf.getvalue()

# -----------------------------
# Interfaz de usuario
# -----------------------------
st.set_page_config(page_title="DJ Mixer Pro", page_icon="🎛️", layout="wide")
st.title("🎛️ Mezclador DJ Profesional")

with st.sidebar:
    st.header("⚙️ Configuración de mezcla")
    target_bpm = st.slider("BPM objetivo", 60, 180, 124)
    align_tempo = st.checkbox("Alinear tempo", value=True)
    align_key = st.checkbox("Alinear tono (pitch)", value=True)
    
    st.subheader("Transición")
    mix_method = st.radio("Método de mezcla", 
                         ["Crossfade Simple", "Sincronización por Beats"], 
                         index=1)
    
    if mix_method == "Crossfade Simple":
        overlap_sec = st.slider("Duración transición (seg)", 3.0, 20.0, 8.0)
    else:
        overlap_beats = st.slider("Beats de transición", 4, 16, 8)
    
    st.subheader("Puntos de mezcla")
    a_mix_start = st.number_input("Inicio mezcla Track A (seg)", min_value=0.0, value=30.0, step=1.0)
    b_mix_start = st.number_input("Inicio mezcla Track B (seg)", min_value=0.0, value=0.0, step=1.0)
    
    extra_sec = st.slider("Audio extra (seg)", 5, 30, 15)

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Track A", type=["wav","mp3","ogg","flac"], key="a")
with col2:
    file_b = st.file_uploader("Track B", type=["wav","mp3","ogg","flac"], key="b")

track_info = {}
for label, file in zip(["A","B"], [file_a, file_b]):
    if file:
        try:
            y, sr = load_audio(file)
            bpm = estimate_bpm(y, sr)
            key_label, key_pc = estimate_key(y, sr)
            track_info[label] = dict(y=y, sr=sr, bpm=bpm, key=key_label, pc=key_pc)
            
            st.sidebar.markdown(f"### Track {label}")
            st.sidebar.metric(f"BPM {label}", f"{bpm:.1f}")
            st.sidebar.caption(f"Clave {label}: {key_label}")
        except Exception as e:
            st.error(f"Error procesando Track {label}: {str(e)}")

if file_a and file_b and st.button("Mezclar 🎶", type="primary"):
    with st.spinner("Procesando y mezclando pistas..."):
        try:
            # Preparar pistas
            y_a = track_info["A"]["y"]
            sr_a = track_info["A"]["sr"]
            y_b = track_info["B"]["y"]
            sr_b = track_info["B"]["sr"]
            
            # Unificar sample rate
            if sr_a != sr_b:
                y_b = librosa.resample(y_b, orig_sr=sr_b, target_sr=sr_a)
                sr_b = sr_a
            
            # Obtener metadatos
            bpm_a = track_info["A"]["bpm"]
            bpm_b = track_info["B"]["bpm"]
            key_pc_a = track_info["A"]["pc"]
            key_pc_b = track_info["B"]["pc"]
            
            # Alinear al BPM y clave objetivo
            target_key_pc = key_pc_a if align_key else None
            
            y_a_aligned = align_to_target(
                y_a, sr_a, bpm_a, key_pc_a, 
                target_bpm, target_key_pc, 
                align_tempo, align_key
            )
            
            y_b_aligned = align_to_target(
                y_b, sr_b, bpm_b, key_pc_b, 
                target_bpm, target_key_pc, 
                align_tempo, align_key
            )
            
            # Aplicar la mezcla seleccionada
            if mix_method == "Sincronización por Beats":
                mixed = beat_sync_crossfade(
                    y_a_aligned, y_b_aligned, sr_a,
                    a_mix_start, b_mix_start,
                    overlap_beats
                )
            else:
                mixed = beat_sync_crossfade(  # Usamos la misma función pero con duración en segundos
                    y_a_aligned, y_b_aligned, sr_a,
                    a_mix_start, b_mix_start,
                    overlap_beats=int(overlap_sec * target_bpm / 60)
                )
            
            # Añadir segmento extra
            extra_samples = int(extra_sec * sr_a)
            start = max(0, int(a_mix_start * sr_a) - extra_samples
            end = min(len(mixed), int(b_mix_start * sr_a) + extra_samples)
            final_mix = mixed[start:end]
            
            # Normalizar y convertir a bytes
            wav_bytes = write_wav_to_bytes(final_mix, sr_a)
            
            # Mostrar resultados
            st.success("✅ Mezcla completada con éxito!")
            st.audio(wav_bytes, format='audio/wav')
            st.download_button(
                "⬇️ Descargar mezcla", 
                data=wav_bytes,
                file_name="mezcla_dj_pro.wav",
                mime="audio/wav"
            )
            
        except Exception as e:
            st.error(f"Error durante la mezcla: {str(e)}")
            st.exception(e)
else:
    st.info("👆 Sube dos archivos de audio y haz clic en 'Mezclar 🎶'")
