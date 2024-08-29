import os
import pickle
import warnings
from typing import Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from malwareDetector import malwareDetector
from dataLoader import DatasetLoad
from gensim.models import Word2Vec


class OpcodeVectorizer:
    def __init__(self, detector, dataset, method = "tfidf"):
        self.byte_sequence_length = detector.byte_sequence_length
        self.vectorize_method = method
        self.embeddingFolder = detector.embeddingFolder
        self.cpuArch = detector.cpuArch
        self.splitByCpu = dataset.splitByCpu
        self.val = dataset.val
        self.trainDataset, self.testDataset, self.valDataset = dataset.trainData, dataset.testData, dataset.valData
        self.vectorize_func = self._get_vectorize_func()
        self.embedInfo = None
        self.dataPath = "./data_5_9_4_text/"

    def _get_vectorize_func(self) -> Any:
        if self.vectorize_method == "tfidf":
            print("Vectorizing byte sequence using TF-IDF.")
            return self.vectorize_tfidf
        elif self.vectorize_method == "word2vec":
            print("Vectorizing byte sequence using Word2Vec.")
            return self.vectorize_word2vec
        else:
            warnings.warn(f"Unsupported vectorize method: {self.vectorize_method}. Defaulting to TF-IDF.", RuntimeWarning)
            return self.vectorize_tfidf

    def _get_file_paths(self) -> Tuple[str, ...]:
        base_path = f"{self.embeddingFolder}/{self.vectorize_method}/{self.cpuArch}_{self.embedInfo}"
        if not os.path.exists(f"{self.embeddingFolder}/{self.vectorize_method}"):
            os.makedirs(f"{self.embeddingFolder}/{self.vectorize_method}")
            
        splitByCpu = "_splitByCpu" if self.splitByCpu else ""
        val = "_withVal" if self.val else ""

        return (
            f"{base_path}_vectorzie_train{splitByCpu}{val}.pickle",
            f"{base_path}_vectorzie_test{splitByCpu}{val}.pickle",
            f"{base_path}_vectorzie_val{splitByCpu}{val}.pickle",
            f"{base_path}_y_train_label{splitByCpu}{val}.pickle",
            f"{base_path}_y_test_label{splitByCpu}{val}.pickle",
            f"{base_path}_y_val_label{splitByCpu}{val}.pickle",
            f"{base_path}_train_label_mapping{splitByCpu}{val}.pickle",
            f"{base_path}_test_label_mapping{splitByCpu}{val}.pickle",
            f"{base_path}_val_label_mapping{splitByCpu}{val}.pickle"
        )

    def _load_pickle(self, file_path: str) -> Any:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def _save_pickle(self, file_path: str, data: Any) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def _check_files_exist(self, file_paths: Tuple[str, ...]) -> bool:
        return all(os.path.exists(path) for path in file_paths)

    def vectorize_tfidf(self, featureDim: int, n_gramRange: tuple) -> Tuple[np.ndarray, ...]:
        self.embedInfo = f"{featureDim}_{n_gramRange[0]}_{n_gramRange[1]}"
        file_paths = self._get_file_paths()

        if not os.path.exists(self.embeddingFolder):
            os.makedirs(self.embeddingFolder)

        if self._check_files_exist(file_paths[:]):  
            print(f"Loading vectorized opcode from {self.embeddingFolder}...")
            return tuple(self._load_pickle(path) for path in file_paths if os.path.exists(path))
        
        print(f"Vectorizing opcode and saving to {self.embeddingFolder}...")

        val_opcode, y_val, val_label_mapping = None, None, None
        train_opcode = self.loadOpcodeData(self.trainDataset)
        test_opcode = self.loadOpcodeData(self.testDataset)
        y_train = self.trainDataset['family'].values
        y_test = self.testDataset['family'].values
        val_opcode = self.loadOpcodeData(self.valDataset) if self.valDataset is not None else None
        y_val = self.valDataset['family'].values if self.valDataset is not None else None
        
        le_train, le_test, le_val = LabelEncoder(), LabelEncoder(), LabelEncoder()
        # print(f"y_train: {y_train}")
        y_train_label, y_test_label, y_val_label = le_train.fit_transform(y_train), le_test.fit_transform(y_test), le_val.fit_transform(y_val) if y_val is not None else None
        # print(f"y_train_label: {y_train_label}")
        train_label_mapping = dict(enumerate(le_train.classes_))
        test_label_mapping = dict(enumerate(le_test.classes_))
        val_label_mapping = dict(enumerate(le_val.classes_)) if y_val is not None else None

        tfidf_vec = TfidfVectorizer(analyzer='word', ngram_range=n_gramRange, max_features=featureDim)

        vectorzie_train = tfidf_vec.fit_transform(train_opcode).toarray()
        vectorzie_test = tfidf_vec.transform(test_opcode).toarray()
        vectorzie_val = tfidf_vec.transform(val_opcode).toarray() if val_opcode is not None else None

        print("Vectorizing byte sequence done.")
        print(f"vectorzie_train shape: {vectorzie_train.shape}")
        print(f"vectorzie_test shape: {vectorzie_test.shape}")
        print(f'train_label_mapping: {train_label_mapping}')
        print(f'test_label_mapping: {test_label_mapping}')
        if vectorzie_val is not None:
            print(f"vectorzie_val shape: {vectorzie_val.shape}")
            print(f'val_label_mapping: {val_label_mapping}')

        data = (vectorzie_train, vectorzie_test, vectorzie_val, y_train_label, y_test_label, y_val_label, 
                train_label_mapping, test_label_mapping, val_label_mapping)

        for path, d in zip(file_paths, data):
            self._save_pickle(path, d)

        return data
    
    def vectorize_word2vec(self, vector_size: int, window: int, min_count: int) -> Tuple[np.ndarray, ...]:
        self.embedInfo = f"{vector_size}_{window}_{min_count}"
        file_paths = self._get_file_paths()

        if not os.path.exists(self.embeddingFolder):
            os.makedirs(self.embeddingFolder)
        
        if self._check_files_exist(file_paths[:]):  
            print(f"Loading vectorized opcode from {self.embeddingFolder}...")
            return tuple(self._load_pickle(path) for path in file_paths if os.path.exists(path))
        
        print(f"Vectorizing opcode and saving to {self.embeddingFolder}...")
        val_opcode, y_val, val_label_mapping = None, None, None
        train_opcode = self.loadOpcodeDataArray(self.trainDataset)
        test_opcode = self.loadOpcodeDataArray(self.testDataset)
        y_train = self.trainDataset['family'].values
        y_test = self.testDataset['family'].values
        val_opcode = self.loadOpcodeDataArray(self.valDataset) if self.valDataset is not None else None
        y_val = self.valDataset['family'].values if self.valDataset is not None else None
        
        le_train, le_test, le_val = LabelEncoder(), LabelEncoder(), LabelEncoder()

        y_train_label, y_test_label, y_val_label = le_train.fit_transform(y_train), le_test.fit_transform(y_test), le_val.fit_transform(y_val) if y_val is not None else None

        train_label_mapping = dict(enumerate(le_train.classes_))
        test_label_mapping = dict(enumerate(le_test.classes_))
        val_label_mapping = dict(enumerate(le_val.classes_)) if y_val is not None else None

        if not os.path.exists(f"{self.embeddingFolder}/word2vec_{self.embedInfo}.model"):
            print("Training Word2Vec model...")
            model = Word2Vec(sentences=train_opcode, vector_size=vector_size, window=window, min_count=min_count, workers=24)
            print("Training Word2Vec model done.")
            model.save(f"{self.embeddingFolder}/word2vec_{self.embedInfo}.model")
        else:
            print("Loading Word2Vec model...")
            model = Word2Vec.load(f"{self.embeddingFolder}/word2vec_{self.embedInfo}.model")    
        vectorzie_train = np.array([np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(vector_size)], axis=0) for words in train_opcode])
        vectorzie_test = np.array([np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(vector_size)], axis=0) for words in test_opcode])
        if val_opcode is not None:
            vectorzie_val = np.array([np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(vector_size)], axis=0) for words in val_opcode])

        print("Vectorizing byte sequence done.")
        print("Vectorizing byte sequence done.")
        print(f"vectorzie_train shape: {vectorzie_train.shape}")
        print(f"vectorzie_test shape: {vectorzie_test.shape}")
        print(f'train_label_mapping: {train_label_mapping}')
        print(f'test_label_mapping: {test_label_mapping}')
        if vectorzie_val is not None:
            print(f"vectorzie_val shape: {vectorzie_val.shape}")
            print(f'val_label_mapping: {val_label_mapping}')

        data = (vectorzie_train, vectorzie_test, vectorzie_val, y_train_label, y_test_label, y_val_label, 
                train_label_mapping, test_label_mapping, val_label_mapping)

        for path, d in zip(file_paths, data):
            self._save_pickle(path, d)

        return data

    def loadOpcodeData(self, data: pd.DataFrame):
        returnValue = None
        for i in range(len(data)):
            cpu = data.iloc[i]["CPU"]
            family = data.iloc[i]["family"]
            filePath = f"{self.dataPath}/{cpu}/{family}/{data.iloc[i]['file_name']}.txt"
            with open(filePath, "r") as f:
                opcode = f.read()
                opcode = opcode.replace("\n", " ")
                if returnValue is None:
                    returnValue = [opcode]
                else:
                    returnValue.append(opcode)         
        return returnValue

    def loadOpcodeDataArray(self, data: pd.DataFrame):
        returnValue = None
        for i in range(len(data)):
            cpu = data.iloc[i]["CPU"]
            family = data.iloc[i]["family"]
            filePath = f"{self.dataPath}/{cpu}/{family}/{data.iloc[i]['file_name']}.txt"
            with open(filePath, "r") as f:
                opcode = f.read().split()
                if returnValue is None:
                    returnValue = [opcode]
                else:
                    returnValue.append(opcode)  
        return returnValue


    # def load_dataset(self, file_path: str) -> Tuple[DataFrame, DataFrame]:
    #     '''
    #     Load the dataset from the dataset path.

    #     Args:
    #         file_path: The path to the dataset.
    #     Returns:
    #         trainDataset: The training dataset.
    #         testDataset: The testing dataset.
    #     '''
    #     print(f"Loading dataset from {file_path}...")
    #     try:
    #         dataset = pd.read_csv(file_path)
    #     except FileNotFoundError:
    #         warnings.warn(f"File not found: {file_path}", RuntimeWarning)
    #         return None, None
        
    #     try:
    #         train_dataset = dataset[dataset["train_test"] == "train"]
    #         test_dataset = dataset[dataset["train_test"] == "test"]
    #     except KeyError:
    #         warnings.warn("KeyError: 'train_test' column not found in dataset", RuntimeWarning)
    #         return None, None
        
        
    #     print(f"Train dataset shape: {train_dataset.shape}")
    #     print(f"Test dataset shape: {test_dataset.shape}")

    #     print("Sorting the dataset by family...")

    #     try:
    #         train_dataset = train_dataset.sort_values(by="family", ignore_index=True)
    #         test_dataset = test_dataset.sort_values(by="family", ignore_index=True)
    #     except KeyError:
    #         warnings.warn("KeyError: 'family' column not found. Unable to sort dataset by family.", RuntimeWarning)
    #         warnings.warn("If the dataset is not sorted by family, the clustering may not work.", RuntimeWarning)

    #     return train_dataset, test_dataset
        


    
    # def vectorize_tfidf(self, datasetTrain: DataFrame, datasetTest: DataFrame, 
    #                     featureDim: int, n_gramRange: tuple, path: str) -> Tuple[array, array, array, array, dict, DataFrame, DataFrame]:
    #     '''
    #     Vectorize the byte sequence of malware samples.
    #     If the vectorized byte sequence exists, load it directly.

    #     Args:
    #         datasetTrain: The training dataset. 
    #         datasetTest: The testing dataset.
    #         featureDim: The feature dimension.
    #         n_gramRange: The n-gram range.
    #         path: The path to save the vectorized byte sequence.
    #     Returns:
    #         vectorzieTrain: The vectorized training dataset.
    #         vectorzieTest: The vectorized testing dataset.
    #         y_train_label: The label of the training dataset.
    #         y_test_label: The label of the testing dataset.
    #         label_mapping: The label mapping.
    #     '''
    #     train_file = f"{path}/vectorzieTrain.csv"
    #     test_file = f"{path}/vectorzieTest.csv"
    #     label_file = f"{path}/label_mapping.pkl"

    #     if self.detector.label == "_arch_label":
    #         train_file = f"{path}/vectorzieTrain{self.detector.label}.csv"
    #         test_file = f"{path}/vectorzieTest{self.detector.label}.csv"
    #         label_file = f"{path}/label_mapping{self.detector.label}.pkl"


    #     vectorzieTrainDf = datasetTrain[["file_name","CPU","family","byte_sequence"]]
    #     vectorzieTestDf = datasetTest[["file_name","CPU","family","byte_sequence"]]
    #     if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(label_file):
    #         print(f"Loading vectorized byte sequence from {path}...")
    #         vectorzieTrainDf = pd.read_csv(train_file)
    #         vectorzieTestDf = pd.read_csv(test_file)
    #         with open(label_file, "rb") as f:
    #             label_mapping = pickle.load(f)
    #             f.close()
    #     elif not os.path.exists(path):
    #         print(f"Creating folder {path}...")
    #         self.detector.mkdir(path)

    #     col = f'tfidf_{featureDim}_{n_gramRange[0]}_{n_gramRange[1]}'
    #     ycol = 'y_label'
    #     feature_col = [f"{col}_{i}" for i in range(featureDim)]
    #     if feature_col[0] not in vectorzieTrainDf.columns or feature_col[0] not in vectorzieTestDf.columns or ycol not in vectorzieTrainDf.columns or ycol not in vectorzieTestDf.columns or label_file is None:
    #         print(f"Vectorizing byte sequence and saving to {path}...")
    #         byteSequenceTrain = datasetTrain['byte_sequence'].values
    #         byteSequenceTest = datasetTest['byte_sequence'].values
    #         y_train = datasetTrain['family'].values
    #         y_test = datasetTest['family'].values

    #         le = LabelEncoder()
    #         y_train_label, y_test_label, label_mapping = self._encode_labels(le, y_train, y_test)

    #         tfidf_vec = TfidfVectorizer(analyzer='word', ngram_range=n_gramRange, max_features=featureDim)
    #         tfidf_matrix_train = tfidf_vec.fit_transform(byteSequenceTrain)
    #         tfidf_matrix_test = tfidf_vec.transform(byteSequenceTest)

    #         vectorzieTrain = tfidf_matrix_train.toarray()
    #         vectorzieTest = tfidf_matrix_test.toarray()
    #         vectorzieTrainDf = pd.concat([
    #             vectorzieTrainDf,
    #             pd.DataFrame(vectorzieTrain, columns=feature_col, index=vectorzieTrainDf.index)
    #         ], axis=1)
            
    #         vectorzieTestDf = pd.concat([
    #             vectorzieTestDf,
    #             pd.DataFrame(vectorzieTest, columns=feature_col, index=vectorzieTestDf.index)
    #         ], axis=1)

    #         vectorzieTrainDf.loc[:, ycol] = y_train_label
    #         vectorzieTestDf.loc[:, ycol] = y_test_label
    #         print("Vectorizing byte sequence done.")
    #         vectorzieTrainDf.to_csv(train_file, index=False)
    #         vectorzieTestDf.to_csv(test_file, index=False)
    #         with open(label_file, "wb") as f:
    #             pickle.dump(label_mapping, f)
    #             f.close()
    #     else:
    #         # Load the vectorized byte sequence
    #         vectorzieTrain = vectorzieTrainDf[feature_col].values
    #         vectorzieTest = vectorzieTestDf[feature_col].values
    #         y_train_label = vectorzieTrainDf['y_label'].values
    #         y_test_label = vectorzieTestDf['y_label'].values
    #         with open(label_file, "rb") as f:
    #             label_mapping = pickle.load(f)
    #             f.close()
        
    #     print("vectorzieTrain shape:", vectorzieTrain.shape)
    #     print("vectorzieTest shape:", vectorzieTest.shape)
    #     print('label_mapping:', label_mapping)
    #     return vectorzieTrain, vectorzieTest, y_train_label, y_test_label, label_mapping, vectorzieTrainDf, vectorzieTestDf
    
    # def _encode_labels(self, le: LabelEncoder, y_train: Any, y_test: Any) -> Tuple[Any, Any, dict]:
    #     le.fit(list(y_train) + list(y_test))
    #     y_train_label = le.transform(y_train)
    #     y_test_label = le.transform(y_test)
    #     label_mapping = {index: label for index, label in enumerate(le.classes_)}
    #     return y_train_label, y_test_label, label_mapping