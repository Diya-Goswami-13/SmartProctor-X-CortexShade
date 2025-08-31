import argparse
import json
import time
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import os


def record_audio(duration=5, fs=44100, device=None, outpath=None):
    print(f"[record_audio] Recording {duration}s @ {fs}Hz (device={device}) ...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', device=device)
    sd.wait()
    audio = recording.flatten()
    if outpath:
        sf.write(outpath, audio, fs)
        print(f"[record_audio] Saved to {outpath}")
    return audio, fs


def extract_features(audio, sr):
    audio = audio.astype(float)

    audio = audio - np.mean(audio)

    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    mean_rms = float(np.mean(rms))
    max_rms = float(np.max(rms))

    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    mean_zcr = float(np.mean(zcr))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    mean_centroid = float(np.mean(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    mean_bandwidth = float(np.mean(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    mean_rolloff = float(np.mean(rolloff))

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=False)
    onset_count = int(len(onsets))
    onset_rate = onset_count / max(1.0, len(audio) / sr)  # onsets per second

    features = {
        "mean_rms": mean_rms,
        "max_rms": max_rms,
        "mean_zcr": mean_zcr,
        "mean_centroid": mean_centroid,
        "mean_bandwidth": mean_bandwidth,
        "mean_rolloff": mean_rolloff,
        "onset_count": onset_count,
        "onset_rate": onset_rate,
        "duration": float(len(audio) / sr)
    }
    return features


def classify_audio(features, thresholds=None):
    """
    Basic rule-based classifier that returns:
      - 'whisper'
      - 'normal_talk'
      - 'background_talk'
      - 'tapping'
      - 'silence'
      - 'unknown'
    thresholds: dict to override defaults
    """
    defaults = {
        "silence_rms": 0.003,
        "whisper_rms": 0.01,
        "normal_rms": 0.02,
        "tap_onset_rate": 6.0,  # stricter
        "tap_zcr": 0.18,  # stricter
        "tap_max_rms": 0.03,  # new
        "centroid_high": 1500.0,
    }
    if thresholds:
        defaults.update(thresholds)
    t = defaults

    mr = features["mean_rms"]
    mz = features["mean_zcr"]
    orate = features["onset_rate"]
    cent = features["mean_centroid"]

    label = "unknown"
    reason = []

    if mr < t["silence_rms"]:
        label = "silence"
        reason.append(f"mean_rms {mr:.5f} < silence_rms {t['silence_rms']}")
        return label, reason

    if (orate >= t["tap_onset_rate"] and
            mz >= t["tap_zcr"] and
            mr < t["tap_max_rms"]):
        label = "tapping"
        reason.append(
            f"onset_rate {orate:.2f} >= {t['tap_onset_rate']} and mean_zcr {mz:.3f} >= {t['tap_zcr']} and mean_rms {mr:.5f} < tap_max_rms {t['tap_max_rms']}"
        )
        return label, reason

    if mr < t["whisper_rms"]:
        label = "whisper"
        reason.append(f"mean_rms {mr:.5f} < whisper_rms {t['whisper_rms']}")

        if cent > t["centroid_high"]:
            reason.append(f"centroid {cent:.1f} > {t['centroid_high']} (consistent with whispered fricatives)")
        return label, reason

    if mr >= t["normal_rms"]:
        label = "normal_talk"
        reason.append(f"mean_rms {mr:.5f} >= normal_rms {t['normal_rms']}")
        return label, reason

    label = "background_talk"
    reason.append(f"mean_rms {mr:.5f} between whisper and normal thresholds")
    return label, reason


def compare_two_mics(features1, features2, mic_names=("mic1", "mic2"), diff_threshold=0.02):
    """
    If one mic hears something substantially louder than the other, assume different source location.
    Returns a 'spatial_suspicion' verdict: None or 'suspicious_direction' plus details.
    diff_threshold: absolute RMS difference to trigger suspicion.
    """
    r1 = features1["mean_rms"]
    r2 = features2["mean_rms"]
    res = {
        "rms1": r1,
        "rms2": r2,
        "diff": abs(r1 - r2),
        "suspicious": False,
        "which_louder": None,
        "reason": None
    }
    if abs(r1 - r2) >= diff_threshold:
        res["suspicious"] = True
        res["which_louder"] = mic_names[0] if r1 > r2 else mic_names[1]
        res["reason"] = f"RMS difference {res['diff']:.5f} >= threshold {diff_threshold}"
    else:
        res["reason"] = f"RMS difference {res['diff']:.5f} < threshold {diff_threshold}"
    return res


def write_event_log(event_dict, logfile="neurosentinel_logs.jsonl"):
    """
    Append event_dict as single-line JSON to logfile.
    """
    os.makedirs(os.path.dirname(logfile), exist_ok=True) if os.path.dirname(logfile) else None
    with open(logfile, "a") as f:
        f.write(json.dumps(event_dict) + "\n")
    print(f"[write_event_log] Logged event: {event_dict['event']} @ {event_dict['timestamp']} -> {logfile}")


def analyze_and_log(audio, sr, name="mic1", thresholds=None, logfile="neurosentinel_logs.jsonl"):
    features = extract_features(audio, sr)
    print("Features:", features)  # Add this line
    label, reasons = classify_audio(features, thresholds=thresholds)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    event = {
        "event": label,
        "timestamp": now,
        "mic": name,
        "features": features,
        "reason": reasons
    }
    write_event_log(event, logfile=logfile)
    return features, label, event


def main():
    parser = argparse.ArgumentParser(description="NeuroSentinel simple detector (whisper, background, tap).")
    parser.add_argument("--duration", type=float, default=5.0, help="record duration seconds")
    parser.add_argument("--fs", type=int, default=44100, help="sampling rate")
    parser.add_argument("--out", type=str, default=None, help="save recording to WAV (path)")
    parser.add_argument("--log", type=str, default="neurosentinel_logs.jsonl", help="log file path")
    parser.add_argument("--second_mic", action="store_true",
                        help="record a second mic immediately after first (simulate dual-mic hardware)")
    parser.add_argument("--device1", type=int, default=None, help="sounddevice device index for mic1 (optional)")
    parser.add_argument("--device2", type=int, default=None, help="sounddevice device index for mic2 (optional)")
    parser.add_argument("--diff_threshold", type=float, default=0.02,
                        help="rms difference threshold for dual-mic suspicion")
    args = parser.parse_args()

    audio1, sr1 = record_audio(duration=args.duration, fs=args.fs, device=args.device1,
                               outpath=(args.out + "_mic1.wav" if args.out else None))
    features1, label1, event1 = analyze_and_log(audio1, sr1, name="mic1", logfile=args.log)

    print("Features:", features1)

    if args.second_mic:
        print("[main] Recording mic2 now...")
        audio2, sr2 = record_audio(duration=args.duration, fs=args.fs, device=args.device2,
                                   outpath=(args.out + "_mic2.wav" if args.out else None))
        features2, label2, event2 = analyze_and_log(audio2, sr2, name="mic2", logfile=args.log)

        cmp = compare_two_mics(features1, features2, mic_names=("mic1", "mic2"), diff_threshold=args.diff_threshold)
        cmp_event = {
            "event": "dual_mic_compare",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "comparison": cmp
        }
        write_event_log(cmp_event, logfile=args.log)
        print("[main] Dual-mic comparison result:", cmp)

    print(
        f"[main] mic1 -> {label1}; features summarized: rms={features1['mean_rms']:.5f}, onset_rate={features1['onset_rate']:.2f}")
    print(json.dumps(event1, indent=2))


if __name__ == "__main__":
    main()