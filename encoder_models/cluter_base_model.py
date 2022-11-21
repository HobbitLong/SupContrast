import typing
import numpy as np
from .base_model import BaseModel


class ClusterBaseModel(BaseModel):

    @classmethod
    def get_centroid(cls, encoded_vector):
        """Get centroid."""
        return np.expand_dims(np.sum(encoded_vector, axis=0) / encoded_vector.shape[0], axis=0)

    def question1(self,
                  query_image_paths: typing.List[str],
                  answer_image_paths: typing.List[str],
                  ) -> int:
        """Get response for question 1."""
        query_vectors = np.stack(self._preprocess_image_list(query_image_paths, 1))
        answer_vectors1 = np.stack(self._preprocess_image_list(answer_image_paths[:3], 3))
        answer_vectors2 = np.stack(self._preprocess_image_list(answer_image_paths[3:], 3))
        centroid1 = self.get_centroid(answer_vectors1)
        centroid2 = self.get_centroid(answer_vectors2)
        # Find which centroid the query image is closest to.
        D, I = self.get_k_nearest_neighbors(query_vectors, np.concatenate((centroid1, centroid2)), 2)
        print(f"Answer centroid distance for group {I[0][0] + 1} - {D[0][0]} "
              f"is closer to query than answer centroid distance to group {I[0][1] + 1} - {D[0][1]}")
        return I[0][0] + 1

    def question2(self,
                  query_image_paths: typing.List[str],
                  answer_image_paths: typing.List[str],
                  ) -> bool:
        """Get response for question 2."""
        query_vectors = np.stack(self._preprocess_image_list(query_image_paths, 3))
        answer_vectors = np.stack(self._preprocess_image_list(answer_image_paths, 1))
        total_vectors = np.concatenate((query_vectors, answer_vectors), axis=0)
        centroid = self.get_centroid(total_vectors)
        D, I = self.get_k_nearest_neighbors(centroid, total_vectors, 4)
        for i, d in zip(I[0], D[0]):
            if i != 3:
                print(f"Distance to centroid for query image {i + 1} is {d}")
            else:
                print(f"Distance to centroid for answer image is {d}")
        # Return true if the answer image is not the furthest image away from the centroid.
        is_similar = I[0][-1] != 3
        print(
            "Answer is similar!" if is_similar else "Answer is different"
              )
        return is_similar

    def question3(self,
                  query_image_paths1: typing.List[str],
                  query_image_paths2: typing.List[str],
                  answer_image_paths: typing.List[str]
                  ) -> typing.Tuple[int, int]:
        """Get response for question 3."""
        query1_vectors = np.stack(self._preprocess_image_list(query_image_paths1, 3))
        query2_vectors = np.stack(self._preprocess_image_list(query_image_paths2, 3))

        centroid1 = self.get_centroid(query1_vectors)
        centroid2 = self.get_centroid(query2_vectors)

        D1, _ = self.get_k_nearest_neighbors(centroid1, query1_vectors, 3)
        D2, _ = self.get_k_nearest_neighbors(centroid2, query2_vectors, 3)
        # Choose which group of images in closer in euclidian space as the positive image group.
        print(f"Total euclidian distance from group1 query images to group 1 centroid  is {sum(D1[0])}")
        print(f"Total euclidian distance from group2 query images to group 2 centroid  is {sum(D2[0])}")
        centroid, query_answer = ((centroid1, 0) if sum(D1[0]) < sum(D2[0]) else (centroid2, 1))
        print(f"Group {query_answer + 1} is more similar.")

        answer_vectors1 = np.stack(self._preprocess_image_list(answer_image_paths[:3], 3))
        answer_vectors2 = np.stack(self._preprocess_image_list(answer_image_paths[3:6], 3))
        answer_vectors3 = np.stack(self._preprocess_image_list(answer_image_paths[6:], 3))
        # Calculate the distance of all answer image groups to the chosen query centroid.
        D1, _ = self.get_k_nearest_neighbors(centroid, answer_vectors1, 3)
        D2, _ = self.get_k_nearest_neighbors(centroid, answer_vectors2, 3)
        D3, _ = self.get_k_nearest_neighbors(centroid, answer_vectors3, 3)
        distance_to_centroid = [sum(D1[0]), sum(D2[0]), sum(D3[0])]
        # Choose the smallest distance to the chosen centroid
        print(f"Total euclidian distance from group1 answer images to centroid  is {sum(D1[0])}")
        print(f"Total euclidian distance from group2 answer images to centroid  is {sum(D2[0])}")
        print(f"Total euclidian distance from group3 answer images to centroid  is {sum(D3[0])}")
        chosen_group = (distance_to_centroid.index(min(distance_to_centroid)) + 1)
        print(f"Closest group to centroid is {chosen_group}")
        return query_answer, chosen_group

    def question4(self,
                  query_image_paths: typing.List[str],
                  answer_image_paths: typing.List[str],
                  ) -> typing.Tuple[int, int]:
        """Get response for question 4."""
        query_vectors = np.stack(self._preprocess_image_list(query_image_paths, 3))
        answer_vectors = np.stack(self._preprocess_image_list(answer_image_paths, 5))
        # Get the two closest answer images to each query centroid.
        centroid = self.get_centroid(query_vectors)
        D, I = self.get_k_nearest_neighbors(centroid, answer_vectors, 5)
        for i, d in zip(I[0], D[0]):
            print(f"Euclidean distance to query centroid for answer image {i + 1} is {d}")
        print(f"Two closest images are {I[0][0] + 1, I[0][1] + 1}")
        return I[0][0] + 1, I[0][1] + 1

    def group2(self,
               query_image_paths: typing.List[str],
               answer_image_paths: typing.List[str],
               ) -> int:
        """Get response for group 2 questions."""
        query_vectors = np.stack(self._preprocess_image_list(query_image_paths, 1))
        answer_vectors = np.stack(self._preprocess_image_list(answer_image_paths, 3))
        # Find which answer image the query is closest 2.
        D, I = self.get_k_nearest_neighbors(query_vectors, answer_vectors, 3)
        for i, d in zip(I[0], D[0]):
            print(f"Euclidean distance to query for answer image {i + 1} is {d}")
        print(f"Closest image is {I[0][0] + 1}")
        return I[0][0] + 1
