import io
import os
import pickle
import numpy as np
import pandas as pd 
from PIL import Image
from sklearn import svm
from joblib import dump, load
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import simmusic
from pychord_tools.models import load_model
import guitar_for_beginners.code.piano as piano


experiments_dir = "guitar_for_beginners/data/exercises/piano_experiments"
latency = 0.0

if __name__ == "__main__":
    main()

def main():
    # COMPUTE STATISTICS
    for exercise in os.listdir(experiments_dir):
        if exercise.startswith("ex"):
            exercise_dir = f"{experiments_dir}/{exercise}"
            audio_filenames = []
            lilypond_filename = ""
            annotation_filename = ""

            for file in os.listdir(exercise_dir):
                if file.endswith(".ly"):
                    lilypond_filename = f"{exercise_dir}/{file}"
                elif file.endswith(".json"):
                    annotation_filename = f"{exercise_dir}/{file}"
                elif file.endswith(".wav"):
                    audio_filenames.append(f"{exercise_dir}/{file}")

            for index, audio_filename in enumerate(audio_filenames):
                pitch_score = audio_filename.split('.')[0].split('_')[-2]
                timing_score = audio_filename.split('.')[0].split('_')[-1]
                score = '_'.join([pitch_score, timing_score])
                results, timing_statistics, chroma_statistics = piano.assess_piano_exercise(
                    annotation_filename, lilypond_filename, latency, audio_filename, image_format="pdf")

                # timing_dict = {"f_measure": timing_statistics["f_measure"], "precision": timing_statistics["precision"], "recall": timing_statistics["recall"]}
                del timing_statistics["thresholded"]
                # statistics = {"kurtosis0": chroma_statistics["kurtosis0"], "std_6th_moment0": chroma_statistics["std_6th_moment0"], "dev.std_6th_moment0": timing_statistics["dev.std_6th_moment0"]}
                statistics = {"variance0": chroma_statistics["variance0"], "chroma_pred_score": chroma_statistics["chroma_pred_score"], "dev.variance0": timing_statistics["dev.variance0"], "dev.mean_diff_ext": timing_statistics["dev.mean_diff_ext"]}
                _data = pd.DataFrame.from_dict({**statistics, "score": score}, "index").transpose()
                if index == 0:
                    data = _data
                else:
                    data = data.append(_data)

            data.to_csv(f"{exercise_dir}/{exercise}.csv")


    #################################################################################################


    # PREPROCESS DATA
    for index, exercise in enumerate(os.listdir(experiments_dir)):
        if exercise.startswith("ex"):
            data_filename = f"{experiments_dir}/{exercise}/{exercise}.csv"
            data = pd.read_csv(data_filename)
            data = data.drop(columns=["Unnamed: 0"])
            data_modif = data.copy()
            min_max_scaler = preprocessing.MinMaxScaler()
            # data_modif.iloc[:, :-1] = min_max_scaler.fit_transform(data.iloc[:, :-1].values)

            # PREPARE AND SPLIT DATA
            X = data_modif.iloc[:, :-1].values 
            data_modif.score = pd.Categorical(data_modif.score)
            y = np.array(data_modif.score.cat.codes)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

            # DEFINE MODEL AND TRAIN
            clf = svm.SVC(gamma = 1 / (X.shape[-1] * X.var()))
            clf.fit(X, y) 

            # SAVE PRETRAINED MODEL
            dump(clf, f"{experiments_dir}/{exercise}/{exercise}.joblib")


#################################################################################################


def single_file_assessment(evaluation_audio: str, ex_num: str, latency: float):

    evaluation_lilypond = f"{experiments_dir}/{ex_num}/{ex_num}.ly"
    evaluation_annotation = f"{experiments_dir}/{ex_num}/{ex_num}.json"
    evaluation_model = f"{experiments_dir}/{ex_num}/{ex_num}.joblib"


    # COMPUTE FEATURES
    results, timing_statistics, chroma_statistics = piano.assess_piano_exercise(
        evaluation_annotation, evaluation_lilypond, latency, evaluation_audio, image_format="pdf"
    )
    # TODO: YET TO CHOSE ALL THE RELEVANT FIELDS
    #timing_dict = {"f_measure": timing_statistics["f_measure"], "precision": timing_statistics["precision"], "recall": timing_statistics["recall"]}
    del timing_statistics["thresholded"]
    # statistics = {"kurtosis0": chroma_statistics["kurtosis0"], "std_6th_moment0": chroma_statistics["std_6th_moment0"], "dev.std_6th_moment0": timing_statistics["dev.std_6th_moment0"]}
    statistics = {"variance0": chroma_statistics["variance0"], "chroma_pred_score": chroma_statistics["chroma_pred_score"], "dev.variance0": timing_statistics["dev.variance0"], "dev.mean_diff_ext": timing_statistics["dev.mean_diff_ext"]}
    import pdb; pdb.set_trace()
    data = pd.DataFrame.from_dict({**statistics}, "index").transpose()
    
    # PREPARE DATA FOR CLASSIFICATION
    X_eval = data.iloc[:, :].values 


    # LOAD MODEL
    clf = load(evaluation_model)


    # EVALUATION
    y_pred = clf.predict(X_eval)


    # WRITE IMAGE FILE
    with open("piano.pdf", "wb") as out_file:
        out_file.write(results["ImageBytes"])


    # PRINT RESULTS AND OPEN IMAGE FILE
    print("SCORES:")
    print('Overall:', results['Overall'])
    print('Rhythm:', results['Rhythm'])
    print('Pitch:', results['Pitch'])
    print("FEEDBACK:")
    if y_pred == 0:
        print("Maybe you should practice a bit more first.")
    elif y_pred == 1:
        print("The rhythm was good but you should practice the pitch a little bit more.")
    elif y_pred == 2:
        print("The pitch was good but you should practice the rhythm a little bit more.")
    elif y_pred == 3:
        print("Well done! Keep up the good work.")

    os.system("xdg-open piano.pdf")
