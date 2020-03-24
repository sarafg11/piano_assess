import json

import essentia
import essentia.standard as ess
import essentia.streaming as esstr
import numpy as np

from pychord_tools.labels import MajMinLabelTranslator, PitchedPattern
from pychord_tools import cacher
from pychord_tools import common_utils
from pychord_tools.common_utils import ChordSegment
from pychord_tools.path_db import get_audio_path
import pychord_tools.low_level_features as features


class AnnotatedBeatChromaEstimator:
    def __init__(self,
                 chroma_estimator=features.HPCPChromaEstimator(
                     audio_path_extractor=features.MBIDUidAndAudioPathExtractor()),
                 uid_extractor=features.MBIDUidAndAudioPathExtractor(),
                 segment_chroma_estimator=features.SmoothedStartingBeatChromaEstimator(),
                 label_translator=features.MajMinLabelTranslator(),
                 roll_to_c_root=True):
        self.chroma_estimator = chroma_estimator
        self.uid_extractor = uid_extractor
        self.beat_chroma_estimator = segment_chroma_estimator
        self.label_translator = label_translator
        self.roll_to_c_root = roll_to_c_root

    def load_chromas_for_annotation_file_list(self, file_list):
        res = features.AnnotatedChromaSegments(
            labels=np.array([], dtype='object'),
            pitches=np.array([], dtype='int'),
            kinds=np.array([], dtype='object'),
            chromas=np.zeros((0, 12), dtype='float32'),
            uids=np.array([], dtype='object'),
            start_times=np.array([], dtype='float32'),
            durations=np.array([], dtype='float32'))
        for file in file_list:
            chunk = self.load_chromas_for_annotation_file(file)
            res.chromas = np.concatenate((res.chromas, chunk.chromas))
            res.labels = np.concatenate((res.labels, chunk.labels))
            res.pitches = np.concatenate((res.pitches, chunk.pitches))
            res.kinds = np.concatenate((res.kinds, chunk.kinds))
            res.uids = np.concatenate((res.uids, chunk.uids))
            res.start_times = np.concatenate((res.start_times, chunk.start_times))
            res.durations = np.concatenate((res.durations, chunk.durations))
        return res

    # returns AnnotatedChromaSegments for the file list
    def load_chromas_for_annotation_file_list_file(self, file_list_file):
        return self.load_chromas_for_annotation_file_list(
            common_utils.load_file_list(file_list_file))

    def load_beats_and_annotations(self, json_file_name, uid):
        with open(json_file_name) as json_file:
            data = json.load(json_file)
            duration = float(data['duration'])

            # LOAD BEATS FROM AUDIO
            audio = ess.MonoLoader(filename=uid)()
            onset_func = ess.OnsetDetectionGlobal()(audio)
            silence_th = 0.2
            if data['bpm'] == 80:
                silence_th = 0.1
            beats = list(ess.Onsets(alpha=1, silenceThreshold=silence_th)([onset_func],[1]))
            real_duration = len(audio)/44100
            
            all_beats = []
            all_chords = []
            # common_utils.process_parts(data['metre'], data, beats, all_chords, 'chords')
            # segments = common_utils.to_beat_chord_segment_list(all_beats[0], duration, beats, all_chords)
            common_utils.process_parts(data['metre'], data, all_beats, all_chords, 'chords')
            # min_onset_dur = (all_beats[1] - all_beats[0]) / 16 #max amount of notes that can be in a 
            if len(beats) > len(all_beats):
                beats = beats[:len(all_beats)]
            if len(beats) < len(all_beats):
                chords = all_chords[:len(beats)]
            else: 
                chords = all_chords
            segments = common_utils.to_beat_chord_segment_list(beats[0], real_duration, beats, chords)
            #
            chromas = None
            labels = np.empty(len(segments), dtype='object')
            pitches = np.empty(len(segments), dtype='int')
            kinds = np.empty(len(segments), dtype='object')
            uids = np.empty(len(segments), dtype='object')
            start_times = np.zeros(len(segments), dtype='float32')
            durations = np.zeros(len(segments), dtype='float32')

            for i in range(len(segments)):
                pitch, kind = self.label_translator.label_to_pitch_and_kind(segments[i].symbol)
                s = int(float(segments[i].start_time) *
                        self.chroma_estimator.sample_rate / self.chroma_estimator.hop_size)
                e = int(float(segments[i].end_time) *
                        self.chroma_estimator.sample_rate / self.chroma_estimator.hop_size)
                if s == e:
                    print("empty segment ", segments[i].start_time, segments[i].end_time)
                    raise Exception("empty segment")
                labels[i] = segments[i].symbol
                pitches[i] = pitch
                kinds[i] = kind
                uids[i] = uid
                start_times[i] = segments[i].start_time
                durations[i] = float(segments[i].end_time) - float(segments[i].start_time)
            return features.AnnotatedChromaSegments(labels, pitches, kinds, chromas, uids, start_times, durations)

    def load_chromas_for_annotation_file(self, annotation_file_name):
        uid = self.uid_extractor.uid(annotation_file_name)
        chroma = self.chroma_estimator.estimate_chroma(uid)
        annotated_chroma_segments = self.load_beats_and_annotations(annotation_file_name, uid)
        self.beat_chroma_estimator.fill_segments_with_chroma(annotated_chroma_segments, chroma)

        if self.roll_to_c_root:
            for i in range(len(annotated_chroma_segments.chromas)):
                shift = 12 - annotated_chroma_segments.pitches[i]
                annotated_chroma_segments.chromas[i] = np.roll(
                    annotated_chroma_segments.chromas[i], shift=shift)
        return annotated_chroma_segments