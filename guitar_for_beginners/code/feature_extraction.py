import copy
import vamp
import numpy as np
import essentia
import essentia.standard as ess
import essentia.streaming as esstr
from scipy import interpolate
import os
import string
import subprocess

import math
import scipy.stats as stats
import random
from pychord_tools import symbolic_analysis

import simmusic
from simmusic import segmentation
from simmusic.containers import Recording
from simmusic import utilities as utils
from simmusic import constants
from simmusic import utilities_io
from simmusic.onset_detection import GuitarOnsetDetector, onset_measures
from simmusic.chroma_labels import GuitarLabelTranslator
from simmusic.dtw import dtw
import simmusic.feature_extraction as feature_extraction

from pychord_tools.models import load_model
from pychord_tools.third_party import NNLSChromaEstimator
from pychord_tools.low_level_features import UidExtractor, ChromaEstimator, SegmentChromaEstimator, AnnotatedChromaSegments
from pychord_tools.low_level_features import audio_duration, smooth

from guitar_for_beginners.code.low_level_features import AnnotatedBeatChromaEstimator


def piano_timing_features(
        anno_file,
        audio_file,
        latency,
        max_spectral_centroid=3500,
        onset_threshold=2,
        series_delta=0.22,
        sample_rate=44100):
    bars, beats, events, chords = symbolic_analysis.rhythm_for_file(anno_file)
    beats =  np.array(beats)
    events = np.array(events)
    chords = np.array(chords)

    is_defined = [x[0] != 'N' for x in chords]
    chords = chords[is_defined]
    events = events[is_defined]

    audio = ess.MonoLoader(filename=audio_file)()
    duration = float(len(audio)) / sample_rate
    half_ibi = (beats[1:] - beats[:-1]).mean() / 2
    start = max(events[0] - half_ibi, 0)
    end = min(events[-1] + half_ibi, duration)
    _onsets = np.array(GuitarOnsetDetector(max_spectral_centroid=max_spectral_centroid, onset_threshold=onset_threshold, series_delta=series_delta).predict(audio_file, start, end))
    
    # LOAD BEATS FROM AUDIO
    audio = ess.MonoLoader(filename=audio_file)()
    onset_func = ess.OnsetDetectionGlobal()(audio)
    silence_th = 0.2
    onsets = np.array(list(ess.Onsets(alpha=1, silenceThreshold=silence_th)([onset_func],[1])))
    # onsets = onsets - latency
    # TODO: FIX AUDIOS WITH SAME LATENCY EXCEPT BAD RHYTHM ONES
    import pdb; pdb.set_trace()
    
    devs = feature_extraction.attack_deviations(events, onsets, start, end)
    f, p, r = onset_measures(
        events, onsets, f_measure_threshold=0.25)
    features = {
        'events': events,
        'beats': beats,
        'bars': bars,
        'duration': duration,
        'half_ibi': half_ibi,
        'start': start,
        'end': end,
        'onsets': onsets,
        'devs': devs,
        'f_measure' : f,
        'precision': p,
        'recall': r
    }
    return features


def piano_chroma_scores(anno_file, audio_file):
    m = load_model(os.path.join(simmusic.__path__[0], 'extractors/chord_models/solo_picking_pdf.pkl'))
    
    chroma_estimator1 = NNLSChromaEstimator(hop_size=2048)
    chroma_estimator2 = feature_extraction.MyNNLSChromaEstimator(hop_size=2048)

    lu1, nlu1, real_segments1 = estimate_segment_scores(anno_file, audio_file, m, chroma_estimator1)
    predicted1, plu1 = m.predict(real_segments1.chromas)
    lu2, nlu2, real_segments2 = estimate_segment_scores(anno_file, audio_file, m, chroma_estimator2)
    predicted2, plu2 = m.predict(real_segments2.chromas)
    import pdb; pdb.set_trace()
    nlus = np.array([nlu1, nlu2])
    predicts = np.array([predicted1, predicted2])

    nlus_t = nlus.transpose()
    predicts_t = predicts.transpose()

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

    return final_nlu, score / len(final_nlu)


def estimate_segment_scores(
        annotation_filename,
        student_filename,
        chroma_pattern_model,
        chroma_estimator=NNLSChromaEstimator()):
    """
    Estimates averaged segments chroma scores accoriding to given annotation.

    :param annotation_filename: annotation file
    :param student_filename: Name of the performance audio file of a student
    :param chroma_pattern_model: ChromaPatternModel
        model for estimating chords quality
    :param chroma_estimator: chorma estimator
    :return:
    """
    chromaEstimator = AnnotatedBeatChromaEstimator(
        chroma_estimator=chroma_estimator,
        segment_chroma_estimator=feature_extraction.AdaptiveChromaEstimator(),
        label_translator=GuitarLabelTranslator(),
        uid_extractor=feature_extraction.ConstUIDExtractor(student_filename),
        roll_to_c_root=False)

    realSegments = chromaEstimator.load_chromas_for_annotation_file(annotation_filename)
    # filter out unclassified:

    is_defined = [x != 'unclassified' for x in realSegments.kinds]
    realSegments = AnnotatedChromaSegments(
        realSegments.labels[is_defined],
        realSegments.pitches[is_defined],
        realSegments.kinds[is_defined],
        realSegments.chromas[is_defined],
        realSegments.uids[is_defined],
        realSegments.start_times[is_defined],
        realSegments.durations[is_defined])

    #predicted, plu = chromaPatternModel.predict(realSegments.chromas)

    nlu = chroma_pattern_model.log_utilities_given_sequence(
        chromas=realSegments.chromas, pitched_patterns=realSegments.pitched_patterns(), normalize=True)
    lu = chroma_pattern_model.log_utilities_given_sequence(
        chromas=realSegments.chromas, pitched_patterns=realSegments.pitched_patterns(), normalize=False)

    return lu, nlu, realSegments