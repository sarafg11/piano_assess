# Piano Asess

This repository contains part of the code used to develop the technologies for the automatic assessment of Piano Exercises.

Contents of the reporsitory:

## sample_assessment_piano

Has two different functions: `main`, that corresponds to the overall feature extraction and training process of a dataset of piano exercises; and `single_file_assessment`, that corresponds to the feature extraction and assessment of a single file submission.
The process for both is similar: first the features are computed, using the `feature_extraction` module, then statistics are computed. 

<a href="https://ibb.co/8dsX0cz"><img src="https://i.ibb.co/xfL3DHm/Screenshot-2020-05-21-at-17-30-40.png" alt="Screenshot-2020-05-21-at-17-30-40" border="0"></a>

- For the **training**, the models are defined, trained with those statistics (of the whole dataset) and then saved to files. 
- For the **assessment**, the model files are loaded and the submission is assessed, with the computed statistics. 
### `main()`
Inputs:
- `experiments_dir (str)`: path of the directory containing the dataset of exercises (audios + annotations + lilypond files)

Output files:
- Trained models

### `single_file_assessment()`
Inputs:
- `evaluation_audio (str)`: path of the audio recording to be assessed
- `ex_num (str)`: number of the exercise to assess, that corresponds to the audio submitted
- `latency (float)`: computed latency of the audio 

## sample_musiccritic_exercise

A sample of a Music Critic class for a piano exercise assessment, using the `sample_assessment_piano` module.

## feature_extraction

Contains the two main functions for the extraction of the onset and chroma features of an audio. 
### `piano_timing_features()`
Computes the onset features and statistics of an audio file.

Inputs: 
- `anno_file (str)`: path of the annotation (JSON) file of the corresponding exercise
- `audio_file (str)`: path of the audio file to be assessed
- `latency (float)`: computed latency of the audio 
- `bpm (int)`: BPM of the exercise
- `max_spectral_centroid: default=3500`
- `onset_threshold: default=2`
- `series_delta: default=0.22`
- `sample_rate: default=44100`

Outputs:
- `features (dict)`: dictionary containing an array of the computed onsets, the deviation between those and the ones in the anotation file, as well as the precision, recall and f_measure between those.

### `piano_chroma_scores()`
Computes the chroma features of an audio file

Inputs: 
- `anno_file (str)`: path of the annotation (JSON) file of the corresponding exercise
- `audio_file (str)`: path of the audio file to be assessed

Outputs:
- `final_nlu (List)`: chroma features 
- `score (float)`: score of the predicted chroma values vs the expected
- `bpm (int)`: BPM of the exercise
