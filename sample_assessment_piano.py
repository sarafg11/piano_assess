import io
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import svm, linear_model
from joblib import dump, load
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import simmusic
from pychord_tools.models import load_model

import code.piano as piano


EXPERIMENTS_DIR = "guitar_for_beginners/data/exercises/piano_experiments"
LATENCY = 0.0
AW = load_model(os.path.join(simmusic.__path__[0], 'extractors/guitar_models/picking_workflow.pkl'))


def main(experiments_dir: str):

    # COMPUTE STATISTICS
    index1 = 0
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

            for index2, audio_filename in enumerate(audio_filenames):
                pitch_score = audio_filename.split('.')[0].split('_')[-2]
                timing_score = audio_filename.split('.')[0].split('_')[-1]
                score = '_'.join([pitch_score, timing_score])
                results, timing_statistics, chroma_statistics = piano.assess_piano_exercise(
                    annotation_filename, lilypond_filename, LATENCY, audio_filename, image_format="pdf")
                # timing_dict = {"f_measure": timing_statistics["f_measure"], "precision": timing_statistics["precision"], "recall": timing_statistics["recall"]}
                # del timing_statistics["thresholded"]
                # statistics = {"kurtosis0": chroma_statistics["kurtosis0"], "std_6th_moment0": chroma_statistics["std_6th_moment0"], "dev.std_6th_moment0": timing_statistics["dev.std_6th_moment0"]}
                statistics = {"variance0": chroma_statistics["variance0"], "chroma_pred_score": chroma_statistics["chroma_pred_score"], "dev.variance0": timing_statistics["dev.variance0"], "dev.mean_diff_ext": timing_statistics["dev.mean_diff_ext"], "f_measure": timing_statistics["f_measure"]}
                _data = pd.DataFrame.from_dict({**statistics, "score": score}, "index").transpose()
                if index2 == 0:
                    data = _data
                else:
                    data = data.append(_data)
            data.to_csv(f"{exercise_dir}/{exercise}.csv")
            if index1 == 0:
                all_data = data
            else:
                all_data = all_data.append(data)
            index1 +=1

    all_data.to_csv(f"{experiments_dir}/all_exs.csv")

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
            # clf = linear_model.LogisticRegression(max_iter=10000, multi_class='multinomial', solver='saga').fit(X,y)


            # SAVE PRETRAINED MODEL
            dump(clf, f"{experiments_dir}/{exercise}/{exercise}.joblib")

    data_filename = f"{experiments_dir}/all_exs.csv"
    data = pd.read_csv(data_filename)
    data = data.drop(columns=["Unnamed: 0"])
    data_modif = data.copy()
    data_modif_p = data.copy()
    data_modif_r = data.copy()
    data_modif_p.score = [score.split("_")[0] for score in data_modif_p.score]
    data_modif_p = data_modif_p.drop("dev.variance0", 1)
    data_modif_p = data_modif_p.drop("dev.mean_diff_ext", 1)
    data_modif_r.score = [score.split("_")[1] for score in data_modif_r.score]
    data_modif_r = data_modif_r.drop("variance0", 1)
    data_modif_r = data_modif_r.drop("chroma_pred_score", 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    # data_modif.iloc[:, :-1] = min_max_scaler.fit_transform(data.iloc[:, :-1].values)

    # PREPARE AND SPLIT DATA
    X = data_modif.iloc[:, :-1].values
    data_modif.score = pd.Categorical(data_modif.score)
    y = np.array(data_modif.score.cat.codes)
    Xp = data_modif_p.iloc[:, :-1].values
    data_modif_p.score = pd.Categorical(data_modif_p.score)
    yp = np.array(data_modif_p.score.cat.codes)
    Xr = data_modif_r.iloc[:, :-1].values
    data_modif_r.score = pd.Categorical(data_modif_r.score)
    yr = np.array(data_modif_r.score.cat.codes)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    # DEFINE MODEL AND TRAIN
    clf = svm.SVC(gamma = 1 / (X.shape[-1] * X.var()))
    clf.fit(X, y)
    clfr = linear_model.LogisticRegression(C=1000, max_iter=10000, multi_class='ovr', solver='saga').fit(Xr,yr)
    clfp = linear_model.LogisticRegression(max_iter=10000, multi_class='ovr', solver='saga').fit(Xp,yp)


    # SAVE PRETRAINED MODEL
    dump(clf, f"{experiments_dir}/all_exs.joblib")
    dump(clfr, f"{experiments_dir}/all_exs_r.joblib")
    dump(clfp, f"{experiments_dir}/all_exs_p.joblib")



#################################################################################################


def single_file_assessment(evaluation_audio: str, ex_num: str, latency: float = LATENCY):

    evaluation_lilypond = f"{EXPERIMENTS_DIR}/{ex_num}/{ex_num}.ly"
    evaluation_annotation = f"{EXPERIMENTS_DIR}/{ex_num}/{ex_num}.json"
    # evaluation_model = f"{EXPERIMENTS_DIR}/{ex_num}/{ex_num}.joblib"
    # evaluation_model = f"{experiments_dir}/all_exs.joblib"
    rhythm_evaluation_model = f"{EXPERIMENTS_DIR}/all_exs_r.joblib"
    pitch_evaluation_model = f"{EXPERIMENTS_DIR}/all_exs_p.joblib"

    # COMPUTE FEATURES
    results, timing_statistics, chroma_statistics = piano.assess_piano_exercise(
        evaluation_annotation, evaluation_lilypond, latency, evaluation_audio, image_format="pdf"
    )
    #timing_dict = {"f_measure": timing_statistics["f_measure"], "precision": timing_statistics["precision"], "recall": timing_statistics["recall"]}
    # del timing_statistics["thresholded"]
    # statistics = {"kurtosis0": chroma_statistics["kurtosis0"], "std_6th_moment0": chroma_statistics["std_6th_moment0"], "dev.std_6th_moment0": timing_statistics["dev.std_6th_moment0"]}
    statistics = {"variance0": chroma_statistics["variance0"], "chroma_pred_score": chroma_statistics["chroma_pred_score"], "dev.variance0": timing_statistics["dev.variance0"], "dev.mean_diff_ext": timing_statistics["dev.mean_diff_ext"], "f_measure": timing_statistics["f_measure"]}
    data = pd.DataFrame.from_dict({**statistics}, "index").transpose()
    data_p = data.copy()
    data_p = data_p.drop("dev.variance0", 1)
    data_p = data_p.drop("dev.mean_diff_ext", 1)
    data_r = data.copy()
    data_r = data_r.drop("variance0", 1)
    data_r = data_r.drop("chroma_pred_score", 1)

    # PREPARE DATA FOR CLASSIFICATION
    X_eval = data.iloc[:, :].values
    X_eval_r = data_r.iloc[:, :].values
    X_eval_p = data_p.iloc[:, :].values

    # LOAD MODEL
    # clf = load(evaluation_model)
    clfr = load(rhythm_evaluation_model)
    clfp = load(pitch_evaluation_model)

    # EVALUATION
    # y_pred = clf.predict(X_eval)
    yr_pred = clfr.predict(X_eval_r)
    yp_pred = clfp.predict(X_eval_p)

    # WRITE IMAGE FILE
    with open("piano.pdf", "wb") as out_file:
        out_file.write(results["ImageBytes"])

    # PRINT RESULTS AND OPEN IMAGE FILE
    print("SCORES:")
    print(f"Pitch: {yp_pred}")
    print(f"Rhythm: {yr_pred}")
    print("FEEDBACK:")
    if yp_pred[0] == 0 and yr_pred[0] == 0:
        message = "Maybe you should practice a bit more first."
    elif yp_pred[0] == 0 and yr_pred[0] == 1:
        message = "The rhythm was good but you should practice the pitch a little bit more."
    elif yp_pred[0] == 1 and yr_pred[0] == 0:
        message = "The pitch was good but you should practice the rhythm a little bit more."
    elif yp_pred[0] == 1 and yr_pred[0] == 1:
        message = "Well done! Keep up the good work."
    print(message)
    # os.system("xdg-open piano.pdf") # Ubuntu
    # os.system('open -a "Adobe Acrobat Reader DC.app" piano.pdf') # MAC
    results["Feedback"] = message
    return results


if __name__ == "__main__":
    main(EXPERIMENTS_DIR)
