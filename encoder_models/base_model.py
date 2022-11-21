import os.path
import shutil
import typing
import numpy as np
import faiss
import torch
import glob
from cached_property import cached_property
from PIL import Image
import time


class BaseModel:

    NAME = "Please override in subclass"
    CACHE_DIR = 'cache'

    @cached_property
    def transform(self):
        raise NotImplementedError("Data preprocessing must be implemented in the subclass.")

    def encode(self, preprocessed_image) -> typing.List[float]:
        raise NotImplementedError("Encoding must be implemented in the subclass")

    @property
    def cache_dir(self):
        return os.path.join(self.question_dir, self.CACHE_DIR)

    def __init__(self,  weights_path: str = ""):
        self.weights_path = weights_path
        self.vector_cache = set()

    def _preprocess_image(self, image_path: str):
        """Applies transform to image."""
        image_tensor = Image.open(image_path).convert('RGB')
        return self.transform(image_tensor)

    def encode_images(self, batch_size: int = 128):
        """Applies transform to image."""
        counter = 0
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.mkdir(self.cache_dir)
        image_vectors = []
        image_names = []
        type = f'{self.question_dir}/*/*.jp*'
        num_images = len(glob.glob(type))
        print(f"{self.question_dir}/*/*.jpg", num_images)
        
        for image_path in glob.glob(type):
            file_name = os.path.basename(image_path)
            counter += 1
            if file_name in self.vector_cache:
                continue
            image_vectors.append(self._preprocess_image(image_path))
            image_names.append(file_name)
            self.vector_cache.add(file_name)
            if len(image_vectors) == batch_size:
                preprocessed_images = self.encode(torch.stack(image_vectors))
                for i in range(batch_size):
                    np.save(os.path.join(self.cache_dir, f"{image_names[i]}.npy"), preprocessed_images[i,:])
                image_vectors, image_names = [], []
                print(f"Preprocessed {counter}/{num_images} images.")
        for i, image_name in enumerate(image_names):
            np.save(os.path.join(self.cache_dir, f"{image_name}.npy"), preprocessed_images[i,:])

    def _preprocess_image_list(self,
                               image_path_list: typing.List[str],
                               expected_length: int) -> typing.List[np.array]:
        """Feature extraction for image list."""
        image_vectors = []
        for query_image in image_path_list:
            file_name = os.path.basename(query_image)
            if file_name in self.vector_cache:
                query_vector = np.load(os.path.join(self.cache_dir, f"{file_name}.npy"))
            else:
                preprocessed_image = self._preprocess_image(query_image)
                query_vector = np.squeeze(self.encode(preprocessed_image))
            image_vectors.append(query_vector)
        assert len(image_vectors) == expected_length
        return image_vectors

    @classmethod
    def get_k_nearest_neighbors(cls, query_vectors, answer_vectors, k) -> typing.Tuple[np.array, np.array]:
        """Gets numpy arrays representing distance and number of k nearest answer vectors to query vectors"""
        start = time.time()
        index = faiss.IndexFlatL2(query_vectors.shape[-1])  # build the index
        index.add(np.stack(answer_vectors))
        distance, indices = index.search(query_vectors, k)
        print(f"Neighbour search took {time.time() - start}")
        return distance, indices

    def question1(self,
                  query_image_paths: typing.List[str],
                  answer_image_paths: typing.List[str],
                  ):
        raise NotImplementedError("Implement in subclass.")

    def question2(self,
                  query_image_paths: typing.List[str],
                  answer_image_paths: typing.List[str],
                  ):
        raise NotImplementedError("Implement in subclass.")

    def question3(self,
                  query_image_paths1: typing.List[str],
                  query_image_paths2: typing.List[str],
                  answer_image_paths: typing.List[str],
                  ):
        raise NotImplementedError("Implement in subclass.")

    def question4(self,
                  query_image_paths: typing.List[str],
                  answer_image_paths: typing.List[str],
                  ):
        raise NotImplementedError("Implement in subclass.")

    def group2(self,
               query_image_paths: typing.List[str],
               answer_image_paths: typing.List[str],
               ):
        raise NotImplementedError("Implement in subclass.")
