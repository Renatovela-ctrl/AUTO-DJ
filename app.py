"""
DJ Mixer AI - Versión mejorada
Mejoras añadidas basadas en prácticas profesionales de DJs:
 - Soporte estéreo (preservando canales)
 - Beatgrid y sincronización a compases/tiempos (snap a beat)
 - Ajuste de tempo beat-sincrónico (phase vocoder) con fallback
 - Ajuste de pitch (con opción de usar pyrubberband si está instalada)
 - Emparejamiento de nivel (RMS) y normalización suave
 - Crossfades sincronizados a beats y con curvas: equal-power, linear, exponencial
 - Visualización de forma de onda y marcadores de beat (matplotlib)
 - Previsualización de pistas alineadas antes de mezclar
 - Protección contra cambios extremos de pitch/tempo y mensajes al usuario
 - Mejor manejo de errores y logs

Requerimientos (pip):
 librosa, soundfile, numpy, streamlit, matplotlib
 Opcionales: pyrubberband (mejor preservación de formantes), pyloudnorm (LUFS)

Uso: abre con `streamlit run DJ_Mixer_AI_mejorado.py`
"""

import io
import math
import sys
import logging
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import librosa

import streamlit as st
import matplotlib.pyplot as plt

# Intentar usar pyrubberband si está disponible para pitch shifting de mejor calidad
try:
    import pyrubberband as pyrb
    RUBBERBAND_AVAILABLE = True
except Exception:
    RUBBERBAND_AVAILABLE = False

# -----------------------------
# Helpers y utilidades
# -----------------------------
MAJOR_KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dj_mixer_ai")


def read_audio_preserve_channels(file, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """Lee audio con soundfile para preservar canales. Devuelve array shape (channels, samples)."""
    data, sr = sf.read(file, always_2d=True)
    # data shape: (samples, channels) -> convert to (channels, samples)
    data = data.T.astype(np.float32)
    if sr != target_sr:
        # resample per canal
        data = np.vstack([librosa.resample(ch, orig_sr=sr, target_sr=target_sr) for ch in data])
        sr = target_sr
    return data, sr


def to_mono(x: np.ndarray) -> np.ndarray:
    """Convierte un array (channels, samples) a mono mezclando canales."""
    if x.ndim == 1:
        return x
    return np.mean(x, axis=0)


def estimate_bpm(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo) if tempo and tempo > 0 else 120.0


def estimate_key(y: np.ndarray, sr: int) -> Tuple[str, int]:
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


def semitones_to_shift(src_pc: int, tgt_pc: int) -> int:
    d = tgt_pc - src_pc
    if d > 6: d -= 12
    if d < -6: d += 12
    return int(d)


def compute_rms_db(y: np.ndarray) -> float:
    """RMS en dBFS aprox (0 dBFS = full scale)."""
    y = to_mono(y)
    eps = 1e-9
    rms = np.sqrt(np.mean(y**2) + eps)
    db = 20.0 * np.log10(rms + eps)
    return db


def match_loudness(y: np.ndarray, target_db: float = -14.0) -> np.ndarray:
    """Escala la pista para aproximarse a target_db RMS (dBFS)."""
    current_db = compute_rms_db(y)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20.0)
    return (y * gain).astype(np.float32)


def _pv_time_stretch(y: np.ndarray, rate: float, hop_length: int = 512, win_length: int = 2048) -> np.ndarray:
    if rate <= 0: return y
    stft = librosa.stft(y, n_fft=win_length, hop_length=hop_length, win_length=win_length)
    stretched = librosa.phase_vocoder(stft, rate=rate, hop_length=hop_length)
    y_out = librosa.istft(stretched, hop_length=hop_length, win_length=win_length, length=int(round(len(y)/rate)))
    return y_out.astype(np.float32)


def apply_time_stretch_mono(y: np.ndarray, rate: float) -> np.ndarray:
    if rate <= 0 or np.isclose(rate, 1.0): return y
    try:
        return librosa.effects.time_stretch(y, rate).astype(np.float32)
    except Exception:
        return _pv_time_stretch(y, rate)


def apply_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Aplica time-stretch preservando canales (si corresponde)."""
    if y.ndim == 1:
        return apply_time_stretch_mono(y, rate)
    else:
        return np.vstack([apply_time_stretch_mono(ch, rate) for ch in y])


def apply_pitch_shift(y: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    """Pitch shift preservando canales. Usa pyrubberband si está disponible para mejor calidad."""
    if abs(n_semitones) <= 1e-6: return y
    if RUBBERBAND_AVAILABLE:
        # pyrubberband espera shape (samples,) o (channels, samples)? acepta 1D y 2D transposed; usamos samples
        if y.ndim == 1:
            return pyrb.pitch_shift(y, sr, n_semitones).astype(np.float32)
        else:
            # pyrubberband works on shape (n_channels, n_samples) returning same
            return pyrb.pitch_shift(y, sr, n_semitones).astype(np.float32)
    else:
        try:
            if y.ndim == 1:
                return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_semitones).astype(np.float32)
            else:
                return np.vstack([librosa.effects.pitch_shift(ch, sr=sr, n_steps=n_semitones).astype(np.float32) for ch in y])
        except Exception:
            factor = 2.0 ** (n_semitones / 12.0)
            if y.ndim == 1:
                y_ps = librosa.resample(y, orig_sr=sr, target_sr=int(round(sr*factor)))
                return apply_time_stretch_mono(y_ps, rate=factor)
            else:
                chans = []
                for ch in y:
                    y_ps = librosa.resample(ch, orig_sr=sr, target_sr=int(round(sr*factor)))
                    chans.append(apply_time_stretch_mono(y_ps, rate=factor))
                return np.vstack(chans)


def compute_beat_grid(y: np.ndarray, sr: int, start_bpm: Optional[float] = None) -> np.ndarray:
    y_mono = to_mono(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr, start_bpm=start_bpm)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times, float(tempo)


def snap_time_to_nearest_beat(time_sec: float, beat_times: np.ndarray) -> float:
    if beat_times.size == 0:
        return time_sec
    idx = np.argmin(np.abs(beat_times - time_sec))
    return float(beat_times[idx])


def crossfade_tracks(a: np.ndarray, b: np.ndarray, sr: int, overlap_sec: float = 10.0,
                     a_mix_start: float = 0.0, b_mix_start: float = 0.0,
                     curve: str = 'equal_power') -> Tuple[np.ndarray, int, int]:
    """Crossfade que soporta arrays (channels, samples) o mono. Devuelve out, inicio_cross, fin_cross (en muestras)"""
    # convertir a (channels, samples)
    if a.ndim == 1: a = a[np.newaxis, :]
    if b.ndim == 1: b = b[np.newaxis, :]

    # asegurarse de que ambas tengan mismo numero de canales
    max_ch = max(a.shape[0], b.shape[0])
    if a.shape[0] < max_ch:
        a = np.vstack([a] + [a[0:1]] * (max_ch - a.shape[0]))
    if b.shape[0] < max_ch:
        b = np.vstack([b] + [b[0:1]] * (max_ch - b.shape[0]))

    a_start = min(a.shape[1], _samps(a_mix_start, sr))
    b_start = min(b.shape[1], _samps(b_mix_start, sr))
    fade_len = _samps(overlap_sec, sr)

    n = min(fade_len, a.shape[1] - a_start, b.shape[1] - b_start)
    if n <= 0:
        out = np.concatenate([a, b], axis=1)
        return (out.squeeze(), a_start, a_start)

    a_seg = a[:, a_start:a_start + n]
    b_seg = b[:, b_start:b_start + n]

    t = np.linspace(0, 1, num=n, dtype=np.float32)
    if curve == 'equal_power':
        fade_out = np.cos(t * np.pi / 2)[np.newaxis, :]
        fade_in = np.sin(t * np.pi / 2)[np.newaxis, :]
    elif curve == 'linear':
        fade_out = (1.0 - t)[np.newaxis, :]
        fade_in = t[np.newaxis, :]
    elif curve == 'exp':
        fade_out = (1.0 - t**2)[np.newaxis, :]
        fade_in = (t**0.5)[np.newaxis, :]
    else:
        fade_out = np.cos(t * np.pi / 2)[np.newaxis, :]
        fade_in = np.sin(t * np.pi / 2)[np.newaxis, :]

    mixed_overlap = a_seg * fade_out + b_seg * fade_in

    out = np.concatenate([
        a[:, :a_start],
        mixed_overlap,
        b[:, b_start + n:],
    ], axis=1)
    return out.squeeze(), a_start, a_start + n


def add_clean_segments(a: np.ndarray, b: np.ndarray, sr: int, extra_sec: int = 10,
                       a_mix_start: float = 0.0, b_mix_start: float = 0.0,
                       overlap_sec: float = 10.0, curve: str = 'equal_power') -> np.ndarray:
    core, cross_start, cross_end = crossfade_tracks(a, b, sr, overlap_sec=overlap_sec,
                                                    a_mix_start=a_mix_start, b_mix_start=b_mix_start,
                                                    curve=curve)
    pre = max(0, cross_start - _samps(extra_sec, sr))
    post = min(core.shape[0] if core.ndim == 1 else core.shape[1], cross_end + _samps(extra_sec, sr))

    if core.ndim == 1:
        y = core[pre:post]
    else:
        # core is (channels, samples)
        y = core[:, pre:post]
    peak = np.max(np.abs(y)) if y.size else 1.0
    if peak > 0.99:
        y = (0.99 / peak) * y
    return y.astype(np.float32)


def _samps(sec: float, sr: int) -> int:
    return max(0, int(round(float(sec) * sr)))


def write_wav_to_bytes(y: np.ndarray, sr: int, subtype: str = 'PCM_16') -> bytes:
    buf = io.BytesIO()
    # y may be (channels, samples)
    if y.ndim > 1:
        data = y.T
    else:
        data = y
    sf.write(buf, data, sr, format='WAV', subtype=subtype)
    buf.seek(0)
    return buf.read()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="DJ Mixer AI", page_icon="🎧", layout="wide")
st.title("🎧 Mezclador DJ con IA - Mejorado")
st.write("Sube dos canciones y el sistema intentará crear una mezcla profesional: sincronizada a beat, nivelada y con key matching opcional.")

with st.sidebar:
    st.header("Opciones")
    target_bpm = st.slider("BPM objetivo (si 0 -> usar BPM de A)", 60, 180, 0)
    align_tempo = st.checkbox("Alinear tempo (beat-sync)", value=True)
    align_key = st.checkbox("Alinear tono (pitch)", value=True)
    target_lufs = st.slider("Target RMS aprox (dBFS)", -24, -6, -14)
    extra_sec = st.slider("Segundos extra antes/después", 2, 30, 10)
    overlap_sec = st.slider("Solapamiento (crossfade) [s]", 2, 20, 8)
    a_mix_start = st.number_input("Inicio de mezcla en track A (seg)", min_value=0.0, value=30.0, step=1.0)
    b_mix_start = st.number_input("Inicio de mezcla en track B (seg)", min_value=0.0, value=0.0, step=1.0)
    curve = st.selectbox("Curva de crossfade", ['equal_power','linear','exp'])
    snap_to_beat = st.checkbox("Snap de inicios a beat (más profesional)", True)
    max_semitone_shift = st.slider("Máx semitonos permitidos para pitch (abs)", 0, 12, 4)

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Canción A (WAV/MP3/OGG/FLAC)", type=["wav","mp3","ogg","flac"], key="a")
with col2:
    file_b = st.file_uploader("Canción B (WAV/MP3/OGG/FLAC)", type=["wav","mp3","ogg","flac"], key="b")

track_info = {}

for label, file in zip(["A","B"], [file_a, file_b]):
    if file:
        try:
            y, sr = read_audio_preserve_channels(file, target_sr=44100)
            y_mono = to_mono(y)
            bpm = estimate_bpm(y_mono, sr)
            key_label, key_pc = estimate_key(y_mono, sr)
            track_info[label] = dict(y=y, sr=sr, bpm=bpm, key=key_label, pc=key_pc)
            st.sidebar.markdown(f"### Track {label}")
            st.sidebar.metric(f"BPM {label}", f"{bpm:.1f}")
            st.sidebar.caption(f"Clave {label}: {key_label}")

            # mostrar forma de onda con beats
            beat_times, tempo = compute_beat_grid(y_mono, sr)
            fig, ax = plt.subplots(figsize=(6,1.5))
            t = np.linspace(0, y_mono.shape[0] / sr, num=y_mono.shape[0])
            ax.plot(t, y_mono, linewidth=0.4)
            ax.set_xlim(0, min(t[-1], 60))
            for bt in beat_times[::max(1, len(beat_times)//50)]:
                ax.axvline(bt, color='orange', linewidth=0.5)
            ax.set_yticks([])
            ax.set_xlabel('s')
            st.sidebar.pyplot(fig)

        except Exception as e:
            st.sidebar.error(f"No se pudo leer track {label}: {e}")

if file_a and file_b:
    st.markdown("### Información detectada de pistas")
    st.write("Track A: ", track_info['A'])
    st.write("Track B: ", track_info['B'])

    if st.button("Previsualizar alineación y ajustes"):
        # preparar previsualización
        y_a = track_info['A']['y']; y_b = track_info['B']['y']
        sr = track_info['A']['sr']
        if track_info['B']['sr'] != sr:
            y_b = np.vstack([librosa.resample(ch, orig_sr=track_info['B']['sr'], target_sr=sr) for ch in y_b])

        bpm_a = track_info['A']['bpm']; bpm_b = track_info['B']['bpm']
        key_pc_a = track_info['A']['pc']; key_pc_b = track_info['B']['pc']

        use_target_bpm = target_bpm if target_bpm > 0 else bpm_a

        # snap inicios a beat si se requiere
        if snap_to_beat:
            bt_a, _ = compute_beat_grid(y_a[0] if y_a.ndim>1 else y_a, sr)
            bt_b, _ = compute_beat_grid(y_b[0] if y_b.ndim>1 else y_b, sr)
            a_mix_start_snapped = snap_time_to_nearest_beat(a_mix_start, bt_a) if bt_a.size else a_mix_start
            b_mix_start_snapped = snap_time_to_nearest_beat(b_mix_start, bt_b) if bt_b.size else b_mix_start
        else:
            a_mix_start_snapped = a_mix_start
            b_mix_start_snapped = b_mix_start

        st.info(f"Inicios (A,B) usados: {a_mix_start_snapped:.2f}s, {b_mix_start_snapped:.2f}s")

        # aplicar tempo (solo a la parte que será usada) y pitch (si aplica)
        def align_track_for_preview(y, sr, bpm, key_pc):
            x = y.copy()
            if align_tempo and bpm > 0:
                rate = float(use_target_bpm) / float(bpm)
                # limitar rate razonable
                if rate < 0.5 or rate > 2.0:
                    st.warning(f"Cambio de tempo extremo para una pista: {rate:.2f}x - puede sonar raro")
                x = apply_time_stretch(x, rate)
            if align_key:
                shift = semitones_to_shift(key_pc, track_info['A']['pc'])
                if abs(shift) > max_semitone_shift:
                    st.warning(f"Se requeriría shift de {shift} semitonos (mayor que {max_semitone_shift}). No se aplica automáticamente.")
                else:
                    x = apply_pitch_shift(x, sr, shift)
            # nivel
            x = match_loudness(x, target_db=float(target_lufs))
            return x

        y_a_al = align_track_for_preview(y_a, sr, bpm_a, key_pc_a)
        y_b_al = align_track_for_preview(y_b, sr, bpm_b, key_pc_b)

        # crear mezcla simple de preview (sin recortar mucho)
        preview = add_clean_segments(y_a_al, y_b_al, sr, extra_sec=int(extra_sec/2),
                                     a_mix_start=a_mix_start_snapped, b_mix_start=b_mix_start_snapped,
                                     overlap_sec=overlap_sec, curve=curve)
        wav_bytes_preview = write_wav_to_bytes(preview, sr)
        st.audio(wav_bytes_preview, format='audio/wav')

    if st.button("Mezclar 🎶"):
        with st.spinner("Procesando tracks y creando mezcla final..."):
            try:
                y_a = track_info['A']['y']; y_b = track_info['B']['y']
                sr = track_info['A']['sr']
                if track_info['B']['sr'] != sr:
                    y_b = np.vstack([librosa.resample(ch, orig_sr=track_info['B']['sr'], target_sr=sr) for ch in y_b])

                bpm_a = track_info['A']['bpm']; bpm_b = track_info['B']['bpm']
                key_pc_a = track_info['A']['pc']; key_pc_b = track_info['B']['pc']

                use_target_bpm = target_bpm if target_bpm > 0 else bpm_a

                # snap inicios a beat
                if snap_to_beat:
                    bt_a, _ = compute_beat_grid(y_a[0] if y_a.ndim>1 else y_a, sr)
                    bt_b, _ = compute_beat_grid(y_b[0] if y_b.ndim>1 else y_b, sr)
                    a_mix_start_snapped = snap_time_to_nearest_beat(a_mix_start, bt_a) if bt_a.size else a_mix_start
                    b_mix_start_snapped = snap_time_to_nearest_beat(b_mix_start, bt_b) if bt_b.size else b_mix_start
                else:
                    a_mix_start_snapped = a_mix_start
                    b_mix_start_snapped = b_mix_start

                # funciones de alineación
                def align_track(y, sr, bpm, key_pc):
                    x = y.copy()
                    if align_tempo and bpm > 0:
                        rate = float(use_target_bpm) / float(bpm)
                        if rate < 0.5 or rate > 2.0:
                            st.warning(f"Cambio de tempo extremo para una pista: {rate:.2f}x - puede sonar raro")
                        x = apply_time_stretch(x, rate)
                    if align_key:
                        shift = semitones_to_shift(key_pc, track_info['A']['pc'])
                        if abs(shift) > max_semitone_shift:
                            st.warning(f"Se requeriría shift de {shift} semitonos (mayor que {max_semitone_shift}). No se aplica automáticamente.")
                        else:
                            x = apply_pitch_shift(x, sr, shift)
                    x = match_loudness(x, target_db=float(target_lufs))
                    return x

                y_a_aligned = align_track(y_a, sr, bpm_a, key_pc_a)
                y_b_aligned = align_track(y_b, sr, bpm_b, key_pc_b)

                mixed = add_clean_segments(y_a_aligned, y_b_aligned, sr, extra_sec=int(extra_sec),
                                           a_mix_start=a_mix_start_snapped, b_mix_start=b_mix_start_snapped,
                                           overlap_sec=overlap_sec, curve=curve)

                # final normalization to -1..1
                peak = np.max(np.abs(mixed)) if mixed.size else 1.0
                if peak > 0.99:
                    mixed = (0.99 / peak) * mixed

                wav_bytes = write_wav_to_bytes(mixed, sr)

                st.markdown("### Mezcla final")
                st.audio(wav_bytes, format='audio/wav')
                st.download_button("⬇️ Descargar mezcla (WAV)", data=wav_bytes,
                                   file_name="mezcla_dj_ai_mejorada.wav", mime="audio/wav")

            except Exception as e:
                st.error(f"Error durante la mezcla: {e}")
else:
    st.info("👆 Sube dos archivos de audio para comenzar.")

# -----------------------------
# Notas de desarrollo / próximos pasos
# -----------------------------
st.markdown("---")
st.subheader("Sugerencias para mejorar aún más (no incluidas por defecto)")
st.markdown("""
- Usar `pyrubberband` para pitch shifting de mejor calidad (preserva formantes). Si quieres lo activo por defecto.
- Usar `pyloudnorm` para normalización LUFS profesional (ideal para mezcla entre pistas masterizadas de distinto origen).
- Implementar detección de "phrases" (8/16 compases) para que los crossfades ocurran en puntos musicales relevantes.
- Añadir controles en tiempo real (faders, sync toggle) y una vista de grid para mover puntos de mezcla manualmente.
- Exportar un pequeño cue con metadata (BPM, key, semitone shift, gain aplicado).
""")

logger.info("DJ Mixer AI cargado.")
