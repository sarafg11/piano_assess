import numpy as np
import essentia.standard as ess

import simmusic
from pychord_tools import symbolic_analysis
from pychord_tools.models import load_model
from simmusic.onset_detection import onset_measures
import simmusic.feature_extraction as feature_extraction
from pychord_tools.third_party import NNLSChromaEstimator
from simmusic.feature_extraction import estimate_segment_scores


def piano_timing_features(
        anno_file,
        audio_file,
        latency,
        bpm,
        max_spectral_centroid=3500,
        onset_threshold=2,
        series_delta=0.22,
        sample_rate=44100):
    bars, beats, events, chords = symbolic_analysis.rhythm_for_file(anno_file)
    beats =  np.array(beats)
    events = np.array(events)

    is_defined = [x[0] != 'N' for x in chords]
    chords = chords[is_defined]
    events = events[is_defined]

    # LOAD AUDIO
    audio = ess.MonoLoader(filename=audio_file)()
    duration = float(len(audio)) / sample_rate
    half_ibi = (beats[1:] - beats[:-1]).mean() / 2
    start = max(events[0] - half_ibi, 0)
    end = min(events[-1] + half_ibi, duration)

    # LOAD BEATS FROM AUDIO
    onset_func = ess.OnsetDetectionGlobal()(audio)

    # CHANGE SILENCE THRESHOLD DEPENDING ON THE BPM
    silence_th = 0.2
    if bpm >= 40 and bpm < 50:
        silence_th = 0.2
    if bpm >= 50 and bpm < 60:
        silence_th = 0.15
    if bpm >= 60 and bpm < 70:
        silence_th = 0.1
    if bpm >= 70 and bpm < 80:
        silence_th = 0.05
    if bpm >= 80:
        silence_th = 0.02

    # COMPUTE ONSETS FROM AUDIO
    onsets = np.array(list(ess.Onsets(alpha=1, silenceThreshold=silence_th)([onset_func],[1])))

    # COMPUTE DEVIATIONS BETWEEN ANNOTATION AND COMPUTED ONSETS
    devs = feature_extraction.attack_deviations(events, onsets, start, end)
    f, p, r = onset_measures(
        events, onsets, f_measure_threshold=0.25)

    features = {
        'onsets': onsets,
        'devs': devs,
        'f_measure' : f,
        'precision': p,
        'recall': r
    }
    return features


def piano_chroma_scores(anno_file, audio_file):
    m = load_model(os.path.join(simmusic.__path__[0], 'extractors/chord_models/solo_picking_pdf.pkl'))

    # LOAD TWO DIFFERENT CHROMA ESTIMATORS
    chroma_estimator1 = NNLSChromaEstimator(hop_size=2048)
    chroma_estimator2 = feature_extraction.MyNNLSChromaEstimator(hop_size=2048)

    # COMPUTE CHROMA SCORES AND PREDICTED CHROMAS FROM THE 2 ESTIMATORS
    lu1, nlu1, real_segments1, bpm = estimate_segment_scores(anno_file, audio_file, m, chroma_estimator1)
    predicted1, plu1 = m.predict(real_segments1.chromas)
    lu2, nlu2, real_segments2, bpm = estimate_segment_scores(anno_file, audio_file, m, chroma_estimator2)
    predicted2, plu2 = m.predict(real_segments2.chromas)
    nlus = np.array([nlu1, nlu2])
    predicts = np.array([predicted1, predicted2])

    nlus_t = nlus.transpose()
    predicts_t = predicts.transpose()

    # 1. GET THE SCORE THAT MADE THE BEST PREDICTION OR THE HIGHEST CHROMA VALUE
    # 2. COMPUTE SCORE WITH HOW MANY PREDICTIONS WERE RIGHT
    final_nlu = []
    score = 0
    for i, nlu in enumerate(nlus_t):
        predict = predicts_t[i]
        real = real_segments1.labels[i].split(":")[0]
        tmp_nlu = []
        tmp_score = 0
        for j, pred in enumerate(predict):
            pred = str(pred).split(":")[0]
            if pred == real:
                tmp_nlu.append(nlu[j])
                tmp_score += 1
                continue

        if tmp_score > 0:
            score += 1
        if len(tmp_nlu) == 0:
            final_nlu.append(np.max(nlu, axis=0))
        elif len(tmp_nlu) == 1:
            final_nlu.append(tmp_nlu[0])
        elif len(tmp_nlu) > 1:
            final_nlu.append(np.max(tmp_nlu, axis=0))

    return final_nlu, score / len(final_nlu), bpm