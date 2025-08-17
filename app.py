import io
import math
import sys
import logging
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import librosa
import streamlit as st
import matplotlib.pyplot as plt

# Configuración avanzada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("dj_mixer_ai")

# Intentar usar pyrubberband si está disponible
try:
    import pyrubberband as pyrb
    RUBBERBAND_AVAILABLE = True
    logger.info("pyrubberband disponible para pitch shifting de alta calidad")
except ImportError:
    RUBBERBAND_AVAILABLE = False
    logger.warning("pyrubberband no disponible, usando implementación alternativa")

# -----------------------------
# Dataclasses y constantes
# -----------------------------
@dataclass
class AudioTrack:
    data: np.ndarray
    sr: int
    bpm: float
    key: str
    pc: int
    duration: float
    channels: int
    loudness: float

MAJOR_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CROSSFADE_CURVES = {
    'equal_power': ('cos', 'sin'),
    'linear': ('linear', 'linear'),
    'exponential': ('exp', 'exp'),
    'logarithmic': ('log', 'log')
}

# -----------------------------
# Funciones de utilidad
# -----------------------------
def _samps(sec: float, sr: int) -> int:
    """Convierte segundos a muestras."""
    return max(0, int(round(float(sec) * sr)))

def _db_to_linear(db: float) -> float:
    """Convierte dB a valor lineal."""
    return 10 ** (db / 20.0)

def _linear_to_db(linear: float) -> float:
    """Convierte valor lineal a dB."""
    return 20.0 * math.log10(max(1e-9, linear))

def _normalize_audio(y: np.ndarray) -> np.ndarray:
    """Normaliza el audio a -1.0 a 1.0."""
    peak = np.max(np.abs(y)) if y.size > 0 else 1.0
    return y * (0.99 / peak) if peak > 0.99 else y

# -----------------------------
# Funciones de procesamiento de audio
# -----------------------------
@st.cache_data(show_spinner=False)
def read_audio_preserve_channels(file: Union[str, io.BytesIO], target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """Lee audio preservando los canales y realiza resample si es necesario."""
    try:
        data, sr = sf.read(file, always_2d=True, dtype='float32')
        data = data.T  # Convertir a (canales, muestras)
        
        if sr != target_sr:
            logger.info(f"Resampleando de {sr}Hz a {target_sr}Hz")
            data = np.vstack([librosa.resample(ch, orig_sr=sr, target_sr=target_sr) for ch in data])
            sr = target_sr
            
        return data, sr
    except Exception as e:
        logger.error(f"Error al leer audio: {e}")
        raise

def to_mono(y: np.ndarray) -> np.ndarray:
    """Convierte una señal multicanal a mono."""
    if y.ndim == 1:
        return y
    if y.shape[0] == 2:  # Optimización para stereo
        return 0.5 * (y[0] + y[1])
    return np.mean(y, axis=0)

@st.cache_data(show_spinner=False)
def estimate_bpm(y: np.ndarray, sr: int) -> float:
    """Estima el BPM usando análisis de ritmo."""
    try:
        y_mono = to_mono(y)
        tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr, units='time')
        return float(tempo) if tempo and tempo > 0 else 120.0
    except Exception as e:
        logger.warning(f"Error al estimar BPM: {e}, usando valor por defecto 120.0")
        return 120.0

@st.cache_data(show_spinner=False)
def estimate_key(y: np.ndarray, sr: int) -> Tuple[str, int]:
    """Estima la tonalidad musical del audio."""
    try:
        y_mono = to_mono(y)
        chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
        profile = chroma.mean(axis=1)

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        corr_major = np.array([np.correlate(np.roll(profile, -k), major_profile)[0] for k in range(12)])
        corr_minor = np.array([np.correlate(np.roll(profile, -k), minor_profile)[0] for k in range(12)])

        if corr_major.max() >= corr_minor.max():
            pc = int(np.argmax(corr_major))
            return f"{MAJOR_KEYS[pc]} Major", pc
        else:
            pc = int(np.argmax(corr_minor))
            return f"{MAJOR_KEYS[pc]} Minor", pc
    except Exception as e:
        logger.warning(f"Error al estimar tonalidad: {e}, usando C Major por defecto")
        return "C Major", 0

def semitones_to_shift(src_pc: int, tgt_pc: int) -> int:
    """Calcula la diferencia en semitonos entre dos notas."""
    d = tgt_pc - src_pc
    if d > 6: d -= 12
    if d < -6: d += 12
    return int(d)

def compute_rms_db(y: np.ndarray) -> float:
    """Calcula el nivel RMS en dBFS."""
    y_mono = to_mono(y)
    if y_mono.size == 0:
        return -np.inf
    squared = np.mean(y_mono**2)
    return _linear_to_db(squared)

def match_loudness(y: np.ndarray, target_db: float = -14.0) -> np.ndarray:
    """Ajusta el volumen para alcanzar el nivel RMS objetivo."""
    current_db = compute_rms_db(y)
    gain = _db_to_linear(target_db - current_db)
    return (y * gain).astype(np.float32)

def _phase_vocoder_stretch(y: np.ndarray, rate: float, hop_length: int = 512, win_length: int = 2048) -> np.ndarray:
    """Implementación alternativa de time stretching usando phase vocoder."""
    if rate <= 0 or np.isclose(rate, 1.0):
        return y
    
    stft = librosa.stft(y, n_fft=win_length, hop_length=hop_length, win_length=win_length)
    stretched = librosa.phase_vocoder(stft, rate=rate, hop_length=hop_length)
    y_out = librosa.istft(stretched, hop_length=hop_length, win_length=win_length, length=int(round(len(y)/rate)))
    return y_out.astype(np.float32)

def apply_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Aplica time stretching preservando canales."""
    if rate <= 0 or np.isclose(rate, 1.0):
        return y
    
    try:
        if y.ndim == 1:
            return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
        else:
            return np.vstack([librosa.effects.time_stretch(ch, rate=rate).astype(np.float32) for ch in y])
    except Exception:
        logger.warning("Falló time_stretch de librosa, usando phase vocoder alternativo")
        if y.ndim == 1:
            return _phase_vocoder_stretch(y, rate)
        else:
            return np.vstack([_phase_vocoder_stretch(ch, rate) for ch in y])

def apply_pitch_shift(y: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    """Aplica pitch shift usando el mejor método disponible."""
    if abs(n_semitones) < 0.1:
        return y
    
    try:
        if RUBBERBAND_AVAILABLE:
            return pyrb.pitch_shift(y, sr, n_semitones).astype(np.float32)
        else:
            if y.ndim == 1:
                return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_semitones).astype(np.float32)
            else:
                return np.vstack([librosa.effects.pitch_shift(ch, sr=sr, n_steps=n_semitones).astype(np.float32) for ch in y])
    except Exception as e:
        logger.error(f"Error en pitch shift: {e}, usando método de resample alternativo")
        factor = 2.0 ** (n_semitones / 12.0)
        if y.ndim == 1:
            y_ps = librosa.resample(y, orig_sr=sr, target_sr=int(round(sr*factor)))
            return apply_time_stretch(y_ps, rate=factor)
        else:
            return np.vstack([
                apply_time_stretch(
                    librosa.resample(ch, orig_sr=sr, target_sr=int(round(sr*factor))), 
                    rate=factor
                ) for ch in y
            ])

@st.cache_data(show_spinner=False)
def compute_beat_grid(y: np.ndarray, sr: int, start_bpm: Optional[float] = None):
    """Calcula los tiempos de beat y el tempo."""
    y_mono = to_mono(y)
    
    # Si start_bpm es None, usa un valor predeterminado (ej. 120)
    tempo, beat_frames = librosa.beat.beat_track(
        y=y_mono, 
        sr=sr, 
        start_bpm=start_bpm if start_bpm is not None else 120  # Valor predeterminado
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times, float(tempo)

def snap_time_to_nearest_beat(time_sec: float, beat_times: np.ndarray) -> float:
    """Ajusta un tiempo al beat más cercano."""
    if beat_times.size == 0:
        return time_sec
    idx = np.argmin(np.abs(beat_times - time_sec))
    return float(beat_times[idx])

def create_fade_curve(length: int, curve_type: str = 'equal_power') -> Tuple[np.ndarray, np.ndarray]:
    """Genera curvas de fade in/out."""
    t = np.linspace(0, 1, num=length, dtype=np.float32)
    
    if curve_type == 'equal_power':
        fade_out = np.cos(t * np.pi / 2)
        fade_in = np.sin(t * np.pi / 2)
    elif curve_type == 'linear':
        fade_out = 1.0 - t
        fade_in = t
    elif curve_type == 'exponential':
        fade_out = np.exp(-t * 3)  # más suave al final
        fade_in = 1.0 - np.exp(-(1-t) * 3)
    elif curve_type == 'logarithmic':
        fade_out = 1.0 - np.log1p(t * (math.e - 1))
        fade_in = np.log1p(t * (math.e - 1))
    else:
        fade_out = np.cos(t * np.pi / 2)
        fade_in = np.sin(t * np.pi / 2)
        
    return fade_in, fade_out

def crossfade_tracks(a: np.ndarray, b: np.ndarray, sr: int, overlap_sec: float = 10.0,
                     a_mix_start: float = 0.0, b_mix_start: float = 0.0,
                     curve: str = 'equal_power') -> Tuple[np.ndarray, int, int]:
    """Realiza crossfade entre dos pistas de audio."""
    # Convertir a (canales, muestras) si es necesario
    if a.ndim == 1: a = a[np.newaxis, :]
    if b.ndim == 1: b = b[np.newaxis, :]
    
    # Asegurar mismo número de canales
    max_ch = max(a.shape[0], b.shape[0])
    if a.shape[0] < max_ch:
        a = np.vstack([a] + [a[0:1]] * (max_ch - a.shape[0]))
    if b.shape[0] < max_ch:
        b = np.vstack([b] + [b[0:1]] * (max_ch - b.shape[0]))

    a_start = min(a.shape[1], _samps(a_mix_start, sr))
    b_start = min(b.shape[1], _samps(b_mix_start, sr))
    fade_len = _samps(overlap_sec, sr)
    
    # Calcular longitud de overlap real
    n = min(fade_len, a.shape[1] - a_start, b.shape[1] - b_start)
    if n <= 0:
        out = np.concatenate([a, b], axis=1)
        return out.squeeze(), a_start, a_start

    # Obtener segmentos y aplicar crossfade
    a_seg = a[:, a_start:a_start + n]
    b_seg = b[:, b_start:b_start + n]
    
    fade_in, fade_out = create_fade_curve(n, curve)
    mixed_overlap = a_seg * fade_out[np.newaxis, :] + b_seg * fade_in[np.newaxis, :]

    # Construir resultado final
    out = np.concatenate([
        a[:, :a_start],
        mixed_overlap,
        b[:, b_start + n:],
    ], axis=1)
    
    return out.squeeze(), a_start, a_start + n

def create_mix_segment(a: np.ndarray, b: np.ndarray, sr: int, extra_sec: int = 10,
                       a_mix_start: float = 0.0, b_mix_start: float = 0.0,
                       overlap_sec: float = 10.0, curve: str = 'equal_power') -> np.ndarray:
    """Crea un segmento de mezcla con crossfade y secciones limpias."""
    core, cross_start, cross_end = crossfade_tracks(
        a, b, sr, overlap_sec=overlap_sec,
        a_mix_start=a_mix_start, b_mix_start=b_mix_start,
        curve=curve
    )
    
    # Extraer sección con tiempo adicional
    pre = max(0, cross_start - _samps(extra_sec, sr))
    post = min(core.shape[0] if core.ndim == 1 else core.shape[1], cross_end + _samps(extra_sec, sr))
    
    if core.ndim == 1:
        y = core[pre:post]
    else:
        y = core[:, pre:post]
    
    return _normalize_audio(y).astype(np.float32)

def write_wav_to_bytes(y: np.ndarray, sr: int, subtype: str = 'PCM_16') -> bytes:
    """Escribe audio a bytes en formato WAV."""
    buf = io.BytesIO()
    try:
        # Convertir a (muestras, canales) si es necesario
        data = y.T if y.ndim > 1 else y
        sf.write(buf, data, sr, format='WAV', subtype=subtype)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"Error al escribir WAV: {e}")
        raise

# -----------------------------
# Interfaz de Streamlit
# -----------------------------
def setup_ui() -> Dict:
    """Configura la interfaz de usuario y devuelve los parámetros."""
    st.set_page_config(page_title="DJ Mixer AI Pro", page_icon="🎧", layout="wide")
    st.title("🎧 Mezclador DJ con IA - Versión Profesional")
    st.write("Sube dos canciones y el sistema creará una mezcla sincronizada a beat, nivelada y con key matching.")
    
    with st.sidebar:
        st.header("⚙️ Parámetros de Mezcla")
        
        with st.expander("Configuración Avanzada", expanded=False):
            target_bpm = st.slider("BPM objetivo (0=usar Track A)", 60, 180, 0)
            align_tempo = st.checkbox("Sincronizar tempo", value=True)
            align_key = st.checkbox("Alinear tonalidad", value=True)
            max_semitone_shift = st.slider("Máx. semitonos para pitch shift", 0, 12, 4)
            
        with st.expander("Niveles y Efectos", expanded=False):
            target_lufs = st.slider("Nivel RMS objetivo (dBFS)", -24, -6, -14)
            overlap_sec = st.slider("Duración de crossfade (seg)", 2, 20, 8)
            curve = st.selectbox("Curva de crossfade", list(CROSSFADE_CURVES.keys()))
            extra_sec = st.slider("Segundos extra antes/después", 2, 30, 10)
            snap_to_beat = st.checkbox("Alinear inicios a beat", True)
            
        with st.expander("Puntos de Mezcla", expanded=True):
            a_mix_start = st.number_input("Inicio mezcla en Track A (seg)", min_value=0.0, value=30.0, step=1.0)
            b_mix_start = st.number_input("Inicio mezcla en Track B (seg)", min_value=0.0, value=0.0, step=1.0)
    
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Track A (WAV/MP3/OGG/FLAC)", type=["wav","mp3","ogg","flac"], key="a")
    with col2:
        file_b = st.file_uploader("Track B (WAV/MP3/OGG/FLAC)", type=["wav","mp3","ogg","flac"], key="b")
    
    return {
        'file_a': file_a,
        'file_b': file_b,
        'target_bpm': target_bpm,
        'align_tempo': align_tempo,
        'align_key': align_key,
        'target_lufs': target_lufs,
        'overlap_sec': overlap_sec,
        'curve': curve,
        'extra_sec': extra_sec,
        'snap_to_beat': snap_to_beat,
        'a_mix_start': a_mix_start,
        'b_mix_start': b_mix_start,
        'max_semitone_shift': max_semitone_shift
    }

def display_track_info(track: AudioTrack, label: str):
    """Muestra información de la pista en la sidebar."""
    st.sidebar.markdown(f"### Track {label}")
    st.sidebar.metric(f"BPM {label}", f"{track.bpm:.1f}")
    st.sidebar.caption(f"Tonalidad {label}: {track.key}")
    st.sidebar.caption(f"Duración: {track.duration:.1f}s")
    st.sidebar.caption(f"Canales: {track.channels}")
    st.sidebar.caption(f"RMS: {track.loudness:.1f} dBFS")
    
    # Mostrar forma de onda con beats
    beat_times, tempo = compute_beat_grid(track.data[0] if track.channels > 1 else track.data, track.sr)
    fig, ax = plt.subplots(figsize=(6, 1.5))
    t = np.linspace(0, len(track.data[0] if track.channels > 1 else track.data) / track.sr, 
                    num=len(track.data[0] if track.channels > 1 else track.data))
    
    ax.plot(t, track.data[0] if track.channels > 1 else track.data, linewidth=0.4)
    ax.set_xlim(0, min(t[-1], 60))
    
    for bt in beat_times[::max(1, len(beat_times)//50)]:
        ax.axvline(bt, color='orange', linewidth=0.5, alpha=0.7)
    
    ax.set_yticks([])
    ax.set_xlabel('Tiempo (s)')
    st.sidebar.pyplot(fig)

def process_track(file, label: str) -> Optional[AudioTrack]:
    """Procesa un track y devuelve un objeto AudioTrack."""
    if not file:
        return None
    
    try:
        y, sr = read_audio_preserve_channels(file)
        y_mono = to_mono(y)
        duration = len(y_mono) / sr
        loudness = compute_rms_db(y)
        
        # Calcular BPM y tonalidad en paralelo para mejor rendimiento
        bpm = estimate_bpm(y_mono, sr)
        key_label, key_pc = estimate_key(y_mono, sr)
        
        return AudioTrack(
            data=y,
            sr=sr,
            bpm=bpm,
            key=key_label,
            pc=key_pc,
            duration=duration,
            channels=y.shape[0] if y.ndim > 1 else 1,
            loudness=loudness
        )
    except Exception as e:
        st.sidebar.error(f"Error procesando track {label}: {str(e)}")
        logger.exception(f"Error procesando track {label}")
        return None

def align_track(track: AudioTrack, reference: AudioTrack, params: Dict) -> AudioTrack:
    """Alinea un track con el track de referencia según los parámetros."""
    y = track.data.copy()
    sr = track.sr
    
    # Alinear tempo si está habilitado
    if params['align_tempo'] and track.bpm > 0:
        target_bpm = params['target_bpm'] if params['target_bpm'] > 0 else reference.bpm
        rate = target_bpm / track.bpm
        
        # Limitar el rate para evitar artefactos
        rate = np.clip(rate, 0.5, 2.0)
        if abs(rate - 1.0) > 0.01:
            y = apply_time_stretch(y, rate)
            logger.info(f"Aplicado time stretch a track: {rate:.2f}x")
    
    # Alinear tonalidad si está habilitado
    if params['align_key']:
        shift = semitones_to_shift(track.pc, reference.pc)
        if abs(shift) <= params['max_semitone_shift']:
            y = apply_pitch_shift(y, sr, shift)
            logger.info(f"Aplicado pitch shift a track: {shift} semitonos")
        else:
            logger.warning(f"Shift requerido {shift} semitonos excede el máximo permitido")
    
    # Ajustar nivel de volumen
    y = match_loudness(y, params['target_lufs'])
    
    return AudioTrack(
        data=y,
        sr=sr,
        bpm=params['target_bpm'] if params['target_bpm'] > 0 else track.bpm,
        key=track.key,
        pc=track.pc,
        duration=len(y[0] if y.ndim > 1 else y) / sr,
        channels=track.channels,
        loudness=compute_rms_db(y)
    )

def create_preview(track_a: AudioTrack, track_b: AudioTrack, params: Dict) -> Optional[bytes]:
    """Crea una vista previa de la mezcla."""
    try:
        with st.spinner("Generando vista previa..."):
            # Alinear tracks
            aligned_a = align_track(track_a, track_a, params)  # A se alinea consigo mismo (solo ajuste de volumen)
            aligned_b = align_track(track_b, track_a, params)
            
            # Snap a beats si está habilitado
            if params['snap_to_beat']:
                bt_a, _ = compute_beat_grid(aligned_a.data[0] if aligned_a.channels > 1 else aligned_a.data, aligned_a.sr)
                bt_b, _ = compute_beat_grid(aligned_b.data[0] if aligned_b.channels > 1 else aligned_b.data, aligned_b.sr)
                
                a_mix_start = snap_time_to_nearest_beat(params['a_mix_start'], bt_a) if bt_a.size else params['a_mix_start']
                b_mix_start = snap_time_to_nearest_beat(params['b_mix_start'], bt_b) if bt_b.size else params['b_mix_start']
            else:
                a_mix_start = params['a_mix_start']
                b_mix_start = params['b_mix_start']
            
            # Crear segmento de preview (más corto que la mezcla final)
            preview = create_mix_segment(
                aligned_a.data, aligned_b.data, aligned_a.sr,
                extra_sec=int(params['extra_sec']/2),
                a_mix_start=a_mix_start,
                b_mix_start=b_mix_start,
                overlap_sec=params['overlap_sec'],
                curve=params['curve']
            )
            
            return write_wav_to_bytes(preview, aligned_a.sr)
    except Exception as e:
        st.error(f"Error generando vista previa: {str(e)}")
        logger.exception("Error en create_preview")
        return None

def create_final_mix(track_a: AudioTrack, track_b: AudioTrack, params: Dict) -> Optional[bytes]:
    """Crea la mezcla final."""
    try:
        with st.spinner("Creando mezcla final..."):
            # Alinear tracks
            aligned_a = align_track(track_a, track_a, params)
            aligned_b = align_track(track_b, track_a, params)
            
            # Snap a beats si está habilitado
            if params['snap_to_beat']:
                bt_a, _ = compute_beat_grid(aligned_a.data[0] if aligned_a.channels > 1 else aligned_a.data, aligned_a.sr)
                bt_b, _ = compute_beat_grid(aligned_b.data[0] if aligned_b.channels > 1 else aligned_b.data, aligned_b.sr)
                
                a_mix_start = snap_time_to_nearest_beat(params['a_mix_start'], bt_a) if bt_a.size else params['a_mix_start']
                b_mix_start = snap_time_to_nearest_beat(params['b_mix_start'], bt_b) if bt_b.size else params['b_mix_start']
            else:
                a_mix_start = params['a_mix_start']
                b_mix_start = params['b_mix_start']
            
            st.info(f"Puntos de mezcla ajustados: A={a_mix_start:.2f}s, B={b_mix_start:.2f}s")
            
            # Crear mezcla final
            mixed = create_mix_segment(
                aligned_a.data, aligned_b.data, aligned_a.sr,
                extra_sec=params['extra_sec'],
                a_mix_start=a_mix_start,
                b_mix_start=b_mix_start,
                overlap_sec=params['overlap_sec'],
                curve=params['curve']
            )
            
            return write_wav_to_bytes(mixed, aligned_a.sr)
    except Exception as e:
        st.error(f"Error creando mezcla final: {str(e)}")
        logger.exception("Error en create_final_mix")
        return None

def main():
    """Función principal de la aplicación."""
    params = setup_ui()
    
    # Procesar tracks
    track_a = process_track(params['file_a'], 'A')
    track_b = process_track(params['file_b'], 'B')
    
    if track_a:
        display_track_info(track_a, 'A')
    if track_b:
        display_track_info(track_b, 'B')
    
    # Mostrar información de ambos tracks si están disponibles
    if track_a and track_b:
        st.markdown("### Información de las Pistas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Track A**")
            st.json({
                "BPM": f"{track_a.bpm:.1f}",
                "Tonalidad": track_a.key,
                "Duración": f"{track_a.duration:.1f}s",
                "Canales": track_a.channels,
                "RMS": f"{track_a.loudness:.1f} dBFS"
            })
        
        with col2:
            st.write("**Track B**")
            st.json({
                "BPM": f"{track_b.bpm:.1f}",
                "Tonalidad": track_b.key,
                "Duración": f"{track_b.duration:.1f}s",
                "Canales": track_b.channels,
                "RMS": f"{track_b.loudness:.1f} dBFS"
            })
        
        # Botones de acción
        if st.button("🎧 Previsualizar Mezcla", help="Escucha una vista previa corta de la mezcla"):
            preview_bytes = create_preview(track_a, track_b, params)
            if preview_bytes:
                st.audio(preview_bytes, format='audio/wav')
        
        if st.button("🔄 Crear Mezcla Final", type="primary"):
            mix_bytes = create_final_mix(track_a, track_b, params)
            if mix_bytes:
                st.markdown("### 🎶 Mezcla Final")
                st.audio(mix_bytes, format='audio/wav')
                
                st.download_button(
                    label="⬇️ Descargar Mezcla",
                    data=mix_bytes,
                    file_name="mezcla_dj_ai_pro.wav",
                    mime="audio/wav",
                    type="primary"
                )
    else:
        st.info("👆 Sube dos archivos de audio para comenzar.")

if __name__ == "__main__":
    main()
