import gc
import itertools
from abc import abstractmethod
from functools import partial
from multiprocessing.pool import Pool
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from features import FeatherFeaturePath, FeatherFeatureDF
from utils import simple_timer


def create_word_list(df: pd.DataFrame, col1: str, col2: str) -> List[str]:
    col1_size = df[col1].max() + 1
    col2_list = [[] for _ in range(col1_size)]
    for val1, val2 in zip(df[col1], df[col2]):
        col2_list[val1].append(val2)
    return [' '.join(map(str, list)) for list in col2_list]


class OneVsOneCoOccurrenceLatentVector(FeatherFeaturePath):
    def create_document_term_matrix(self, df, col1, col2):
        word_list = create_word_list(df, col1, col2)
        vectorizer = self.vectorizer_factory()
        return vectorizer.fit_transform(word_list)

    def compute_latent_vectors(self, col_pair, train_path: str, test_path: str) -> Tuple[str, str, np.ndarray]:
        col1, col2 = col_pair
        with simple_timer("[{}] Create {}-{} latent vectors".format(self.name, col1, col2)):
            df_train = pd.read_feather(train_path)
            df_test = pd.read_feather(test_path)
            df_data: pd.DataFrame = pd.concat([df_train, df_test])
            del df_train, df_test
            document_term_matrix = self.create_document_term_matrix(df_data, col1, col2)
            transformer = self.transformer_factory()
            return col1, col2, transformer.fit_transform(document_term_matrix)

    def create_features_from_path(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        column_pairs = self.get_column_pairs()

        col1s = []
        col2s = []
        latent_vectors = []
        with Pool(4) as p:
            for col1, col2, latent_vector in p.map(
                    partial(self.compute_latent_vectors, train_path=train_path, test_path=test_path), column_pairs):
                col1s.append(col1)
                col2s.append(col2)
                latent_vectors.append(latent_vector.astype(np.float32))
        gc.collect()

        return self.get_feature(train_path, col1s, col2s, latent_vectors), \
               self.get_feature(test_path, col1s, col2s, latent_vectors)

    def get_column_pairs(self):
        columns = ['ip', 'app', 'os', 'device', 'channel']
        return [(col1, col2) for col1, col2 in itertools.product(columns, repeat=2) if col1 != col2]

    @staticmethod
    def categorical_features():
        return []

    @property
    @abstractmethod
    def width(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def transformer_factory(self):
        raise NotImplementedError

    @abstractmethod
    def vectorizer_factory(self) -> TransformerMixin:
        raise NotImplementedError

    def get_feature(self, path: str, cs1: List[str], cs2: List[str], vs: List[np.ndarray]) -> pd.DataFrame:
        df_data = pd.read_feather(path)
        features = np.zeros(shape=(len(df_data), len(cs1) * self.width), dtype=np.float32)
        columns = []
        for i, (col1, col2, latent_vector) in enumerate(zip(cs1, cs2, vs)):
            offset = i * self.width
            for j in range(self.width):
                columns.append(self.name + '-' + col1 + '-' + col2 + '-' + str(j))
            for j, val1 in enumerate(df_data[col1]):
                features[j, offset:offset + self.width] = latent_vector[val1]

        return pd.DataFrame(data=features, columns=columns)


class KomakiLDA5(OneVsOneCoOccurrenceLatentVector):
    def vectorizer_factory(self):
        return CountVectorizer(min_df=2)

    def transformer_factory(self) -> TransformerMixin:
        return LatentDirichletAllocation(n_components=self.width, learning_method='online', random_state=71)

    @property
    def width(self) -> int:
        return 5


class KomakiLDA5NoDevice(KomakiLDA5):
    def get_column_pairs(self):
        columns = ['ip', 'app', 'os', 'channel']
        return [(col1, col2) for col1, col2 in itertools.product(columns, repeat=2) if col1 != col2]


class KomakiLDA5MinDF1(OneVsOneCoOccurrenceLatentVector):
    def vectorizer_factory(self):
        return CountVectorizer(min_df=1)

    def transformer_factory(self) -> TransformerMixin:
        return LatentDirichletAllocation(n_components=self.width, learning_method='online', random_state=71)

    @property
    def width(self) -> int:
        return 5


class KomakiPCA5(OneVsOneCoOccurrenceLatentVector):
    def vectorizer_factory(self):
        return TfidfVectorizer(min_df=2, dtype=np.float32)

    def transformer_factory(self) -> TransformerMixin:
        return TruncatedSVD(n_components=self.width, random_state=71)

    @property
    def width(self) -> int:
        return 5


class KomakiPCA5NoDevice(KomakiPCA5):
    def get_column_pairs(self):
        columns = ['ip', 'app', 'os', 'channel']
        return [(col1, col2) for col1, col2 in itertools.product(columns, repeat=2) if col1 != col2]


class KomakiNMF5(OneVsOneCoOccurrenceLatentVector):
    def vectorizer_factory(self):
        return TfidfVectorizer(min_df=2, dtype=np.float32)

    def transformer_factory(self) -> TransformerMixin:
        return NMF(n_components=self.width, random_state=71)

    @property
    def width(self) -> int:
        return 5


class KomakiNMF5NoDevice(KomakiNMF5):
    def get_column_pairs(self):
        columns = ['ip', 'app', 'os', 'channel']
        return [(col1, col2) for col1, col2 in itertools.product(columns, repeat=2) if col1 != col2]


class SinglePCACount(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return []

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        train_length = len(df_train)
        n_components = 30
        df_data: pd.DataFrame = pd.concat([df_train, df_test])
        pipeline = make_pipeline(
            OneHotEncoder(),
            TruncatedSVD(n_components=n_components, random_state=71)
        )
        features = pipeline.fit_transform(df_data[['ip', 'app', 'os', 'device', 'channel']].values).astype(np.float32)
        feature_columns = []
        for i in range(n_components):
            feature_columns.append(self.name + '_{}'.format(i))
        return pd.DataFrame(data=features[:train_length], columns=feature_columns), \
               pd.DataFrame(data=features[train_length:], columns=feature_columns)


class SinglePCATfIdf(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return []

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        train_length = len(df_train)
        n_components = 30
        df_data: pd.DataFrame = pd.concat([df_train, df_test])
        pipeline = make_pipeline(
            OneHotEncoder(),
            TfidfTransformer(),
            TruncatedSVD(n_components=30, random_state=71)
        )
        features = pipeline.fit_transform(df_data[['ip', 'app', 'os', 'device', 'channel']].values).astype(np.float32)
        feature_columns = []
        for i in range(n_components):
            feature_columns.append(self.name + '_{}'.format(i))
        return pd.DataFrame(data=features[:train_length], columns=feature_columns), \
               pd.DataFrame(data=features[train_length:], columns=feature_columns)


class UserItemLDA(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return []

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        train_length = len(df_train)
        n_components = 30
        df_data: pd.DataFrame = pd.concat([df_train, df_test])

        with simple_timer("Create document term matrix"):
            mask_to_id = {}
            values = []
            for ip, app, os, device, channel in zip(df_data.ip, df_data.app, df_data.os, df_data.device,
                                                    df_data.channel):
                mask = (ip << 44) | (device << 20) | (os << 10) | channel
                if mask in mask_to_id:
                    mask_id = mask_to_id[mask]
                else:
                    mask_id = len(mask_to_id)
                    mask_to_id[mask] = mask_id
                while len(values) <= mask_id:
                    values.append([])
                values[mask_id].append(app)
        with simple_timer("Convert to documents"):
            values = [' '.join(map(str, ls)) for ls in values]

        with simple_timer("Vectorize document"):
            if len(df_data) < 1000 * 1000:
                vectorizer = CountVectorizer(min_df=1)
            else:
                vectorizer = CountVectorizer(min_df=2)
            values = vectorizer.fit_transform(values)

        with simple_timer("Run LDA"):
            lda = LatentDirichletAllocation(n_components=n_components, learning_method='online', random_state=71,
                                            n_jobs=-2)
            components = lda.fit_transform(values)

        with simple_timer("Create feature matrix"):
            features = np.zeros(shape=(len(df_data), n_components), dtype=np.float32)
            for i, (ip, app, os, device, channel) in enumerate(
                    zip(df_data.ip, df_data.app, df_data.os, df_data.device, df_data.channel)):
                mask = (ip << 44) | (device << 20) | (os << 10) | channel
                mask_id = mask_to_id[mask]
                features[i, :] = components[mask_id]

        feature_columns = [self.name + '_{}'.format(i) for i in range(n_components)]
        return pd.DataFrame(data=features[:train_length], columns=feature_columns), \
               pd.DataFrame(data=features[train_length:], columns=feature_columns)


class ItemUserLDA(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return []

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        train_length = len(df_train)
        n_components = 30
        df_data: pd.DataFrame = pd.concat([df_train, df_test])

        with simple_timer("Create document term matrix"):
            mask_to_id = {}
            values = []
            for ip, app, os, device, channel in zip(df_data.ip, df_data.app, df_data.os, df_data.device,
                                                    df_data.channel):
                mask = (ip << 44) | (device << 20) | (os << 10) | channel
                if mask in mask_to_id:
                    mask_id = mask_to_id[mask]
                else:
                    mask_id = len(mask_to_id)
                    mask_to_id[mask] = mask_id
                while len(values) <= app:
                    values.append([])
                values[app].append(mask_id)

        with simple_timer("Convert to documents"):
            values = [' '.join(map(str, ls)) for ls in values]

        with simple_timer("Vectorize document"):
            # to handle smaller data
            if len(df_data) < 1000 * 1000:
                vectorizer = CountVectorizer(min_df=1)
            else:
                vectorizer = CountVectorizer(min_df=2)
            values = vectorizer.fit_transform(values)

        with simple_timer("Run LDA"):
            lda = LatentDirichletAllocation(n_components=n_components, learning_method='online', random_state=71,
                                            n_jobs=-2)
            components = lda.fit_transform(values)

        with simple_timer("Create feature matrix"):
            features = np.zeros(shape=(len(df_data), n_components), dtype=np.float32)
            for i, (ip, app, os, device, channel) in enumerate(
                    zip(df_data.ip, df_data.app, df_data.os, df_data.device, df_data.channel)):
                features[i, :] = components[app]

        feature_columns = [self.name + '_{}'.format(i) for i in range(n_components)]
        return pd.DataFrame(data=features[:train_length], columns=feature_columns), \
               pd.DataFrame(data=features[train_length:], columns=feature_columns)


if __name__ == '__main__':
    df_train = pd.read_feather('../data/input/train.feather.small')
    df_test = pd.read_feather('../data/input/test.feather.small')
    lda = UserItemLDA('')
    lda.create_features_from_dataframe(df_train, df_test)
