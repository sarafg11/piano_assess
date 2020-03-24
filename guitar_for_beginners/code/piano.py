import numpy as np
import matplotlib.pyplot as plt
import jinja2
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import os
import re
from subprocess import call
from scipy.stats import gamma
import essentia.standard as ess
from PIL import Image
from tempfile import NamedTemporaryFile
import simmusic.feature_extraction as feature_extraction
import joblib
from pychord_tools.models import load_model
import simmusic
import simmusic.extractors.guitar as guitar

from guitar_for_beginners.code.feature_extraction import piano_chroma_scores, piano_timing_features


def assess_piano_exercise(
    anno_file,
    lilypond_template_file,
    latency,
    student_filename,
    sample_rate=44100,
    image_format='png',
    attack_color_func=guitar.attack_rgba
):
    # WORKFLOW
    assessment_workflow = load_model(os.path.join(simmusic.__path__[0], 'extractors/guitar_models/picking_workflow.pkl'))

    # AUDIO
    audio = ess.MonoLoader(filename=student_filename)()
    frames = int(latency * sample_rate)
    audio = audio[frames:]
    audio_file = NamedTemporaryFile(suffix='.wav', delete=False)
    audio_file.close()
    os.unlink(audio_file.name)
    ess.MonoWriter(filename=audio_file.name, format='wav')(audio)

    # FEATURES
    timing_features = piano_timing_features(
        anno_file, audio_file.name, latency,
        max_spectral_centroid=3500,
        onset_threshold=10,
        series_delta=0.1)
    chroma_scores, _score = piano_chroma_scores(anno_file, audio_file.name)
    # Just for the purpose of visualisation
    # _chroma_scores = feature_extraction.picking_chroma_scores(anno_file, audio_file.name)
    
    # SHEET 
    # solve this shit # v_chroma_scores = feature_extraction.picking_chroma_scores(anno_file, audio_file)
    v_timing_features = feature_extraction.timing_features(
        anno_file, audio_file.name, latency)
    lilypond_basedir = os.path.dirname(lilypond_template_file)
    lilypond_shortname = os.path.basename(lilypond_template_file)
    img_file_name = guitar.visualize(
        lilypond_basedir,
        lilypond_shortname,
        v_timing_features['bars'],
        v_timing_features['events'],
        v_timing_features['onsets'],
        chroma_scores,
        audio,
        image_format=image_format,
        attack_color_func=attack_color_func)
    os.unlink(audio_file.name)

    # STATISTICS
    timing_statistics = feature_extraction.timing_statistics(timing_features['devs'])
    _timing_statistics = {**timing_statistics, "dev_abs": abs(sum(timing_features["devs"])) / len(timing_features["devs"])}
    timing_statistics.update(timing_features)
    chroma_statistics = feature_extraction.chroma_statistics(chroma_scores)
    _chroma_statistics = {**chroma_statistics, "chroma_pred_score": _score}
    X_tuning = [0]
    X_timing = [timing_statistics[f] for f in assessment_workflow.timing_feature_names]
    X_chroma = [chroma_statistics[f] for f in assessment_workflow.chroma_feature_names]
    X_overall = []
    X_overall.extend(X_tuning)
    X_overall.extend(X_timing)
    X_overall.extend(X_chroma)

    #RESULTS
    results = {}
    results["Rhythm"] = guitar.fool_proof_assess(assessment_workflow.rhythm_estimator, X_timing)
    results["Pitch"] = guitar.fool_proof_assess(assessment_workflow.chroma_estimator, X_chroma)
    results["Overall"] = guitar.fool_proof_assess(assessment_workflow.overall_estimator, X_overall)
    results["Overall"] = max(results["Overall"], min( results["Rhythm"], results["Pitch"]))

    with open(img_file_name, "rb") as img_file:
        results["ImageBytes"] = img_file.read()
    os.unlink(img_file_name)

    return results, _timing_statistics, _chroma_statistics