import typing
import numpy as np
import faiss
import sys

from .base_model import BaseModel


class RankBaseModel(BaseModel):

    def _get_distance_between_vector_group(self, query_vectors) -> float:
        """Get the distance between a group of 3 query vectors"""

        assert len(query_vectors) == 3
        index = faiss.IndexFlatL2(query_vectors[0].shape[-1])
        index.add(np.stack(query_vectors[1:]))
        # Get distance from 1 vector to 2 other.
        D1, I = index.search(np.expand_dims(query_vectors[0], axis=0), 2)

        # Get distance between remaining 2 vectors.
        index = faiss.IndexFlatL2(query_vectors[0].shape[-1])
        index.add(np.expand_dims(query_vectors[1], axis=0))
        D2, I = index.search(np.expand_dims(query_vectors[2], axis=0), 1)
        return sum(D1[0]) + sum(D2[0])

    def question1(self,
                  query_image_paths: typing.List[str],
                  answer_image_paths: typing.List[str],
                  k: int = 3) -> int:
        """Get response for question 1."""
        query_vectors = np.stack(self._preprocess_image_list(query_image_paths, 1))
        answer_vectors = np.stack(self._preprocess_image_list(answer_image_paths, 6))
        _, I = self.get_k_nearest_neighbors(query_vectors, answer_vectors, k)
        return self._process_question_1_response(I)

    def _process_question_1_response(self, indices):
        """Process response for question 1."""
        SET = {0, 1, 2}
        indices, = indices
        # indices contains the 3 nearest neighbours to the query image.
        # If 2 or more of those images are in group 1, choose group 1
        # If 2 or more of those images are in group 2, choose group 2
        return 2 if len(list(set(indices) - SET)) > 1 else 1

    def question2(self,
                  query_image_paths: typing.List[str],
                  answer_image_paths: typing.List[str],
                  k: int = 3) -> bool:
        """Get response for question 2."""
        query_vectors = np.stack(self._preprocess_image_list(query_image_paths, 3))
        answer_vectors = np.stack(self._preprocess_image_list(answer_image_paths, 1))
        D, _ = self.get_k_nearest_neighbors(answer_vectors, query_vectors, k)
        return self._process_question_2_response(D)

    def _process_question_2_response(self, distances) -> bool:
        """Process response for question 2."""
        if (distances[0] < 11000).sum():
            ans = True
        elif (distances[0] > 14000).sum():
            ans = False
        else:
            ans = (distances[0] < 12500).sum() == 3
        return ans

    def question3(self,
                  query_image_paths1: typing.List[str],
                  query_image_paths2: typing.List[str],
                  answer_image_paths: typing.List[str]
                  ) -> typing.Tuple[int, int]:
        """Get response for question 3."""
        query1_vectors = self._preprocess_image_list(query_image_paths1, 3)
        query2_vectors = self._preprocess_image_list(query_image_paths2, 3)

        distance_between_query_vectors1 = self._get_distance_between_vector_group(query1_vectors)
        distance_between_query_vectors2 = self._get_distance_between_vector_group(query2_vectors)
        # Choose which group of images in closer in euclidian space as the positive image group.
        query_vectors, query_answer = ((query1_vectors, 0)
                                       if distance_between_query_vectors1 < distance_between_query_vectors2
                                       else (query2_vectors, 1))
        query_vectors = np.stack(query_vectors)

        answer_vectors = np.stack(self._preprocess_image_list(answer_image_paths, 9))
        # For each of the 3 query images, get k=4 nearest neighbours from the 9 answer images
        distance, indices = self.get_k_nearest_neighbors(query_vectors, answer_vectors, 4)

        return query_answer, (self._process_question_3_response(distance, indices) + 1)

    def _process_question_3_response(self, distances, indices):
        """Process response for question 3."""
        count = []
        distance_list = [sys.maxsize, sys.maxsize, sys.maxsize]
        # TODO: This code could probably be cleaned up!
        # Iterate over nearest neighbours for each query image
        for im_indices, im_dist in zip(indices, distances):
            # Get number of answer images from group 1 closest to a query image
            group1 = [list(im_indices).index(i) for i in im_indices if i in [0, 1, 2]]
            #  Get number of answer images from group 2 closest to a query image
            group2 = [list(im_indices).index(i) for i in im_indices if i in [3, 4, 5]]
            # Get number of answer images from group 3 closest to a query image
            group3 = [list(im_indices).index(i) for i in im_indices if i in [6, 7, 8]]

            groups = [group1, group2, group3]
            group_numbers = [len(group1), len(group2), len(group3)]
            # Select the group with the most answer images closest to the query image
            selected_group = group_numbers.index(max(group_numbers))
            im_dist = list(im_dist)
            distance = sum(sorted([im_dist[i] for i in groups[selected_group]])[:2])
            if selected_group in count:
                # If 2 query images select the same answer group, return that answer group
                return selected_group
            else:
                count.append(selected_group)
                distance_list[selected_group] = distance
        # If each query image is closes to a different group, select the answer group closest to the query image
        # in euclidian space.
        answer = distance_list.index(min(distance_list))
        return answer

    def question4(self,
                  query_image_paths: typing.List[str],
                  answer_image_paths: typing.List[str],
                  k: int = 2) -> typing.Tuple[int, int]:
        """Get response for question 4."""
        query_vectors = np.stack(self._preprocess_image_list(query_image_paths, 3))
        answer_vectors = np.stack(self._preprocess_image_list(answer_image_paths, 5))
        # Get the two closest answer images to each query image.
        distance, indices = self.get_k_nearest_neighbors(query_vectors, answer_vectors, k)
        ans1, ans2 = self._process_question_4_response(distance, indices)
        ans1 += 1
        ans2 += 1
        return ans1, ans2

    def _process_question_4_response(self, distances, indices) -> typing.List[int]:
        """Get response for question 4."""
        count_dict = {}
        distance_dict = {}
        # TODO: This code can probably be cleaned up.
        for im_indices, im_dist in zip(indices, distances):
            # Iterate over each query image and mark which answer image was chosen as closest.
            for i, d in zip(im_indices, im_dist):
                if i in count_dict:
                    count_dict[i] += 1
                    distance_dict[i] += d
                else:
                    count_dict[i] = 1
                    distance_dict[i] = d
        # Sort the answer images by the amount of times they were chosen k=2 neighbour of a query image.
        sorted_indices = sorted(list(count_dict.keys()), key=count_dict.get, reverse=True)
        answers = []
        while len(answers) < 2:
            if len(sorted_indices) == 1:
                # If there is only 1 image left in the sorted indices list then it must be chosen,
                answers.append(sorted_indices.pop())
                continue
            if count_dict[sorted_indices[0]] > count_dict[sorted_indices[1]]:
                # If there is one answer image selected as a k=2 nearest image more than the other images. This image
                # Should be chosen
                answers.append(sorted_indices[0])
                sorted_indices.pop(0)
                continue
            # If multiple answer images are selected as a k=2 nearest neighbour with the same frequency,
            # choose which ever answer image is closest to the query image.
            top_indices = [i for i in sorted_indices if count_dict[i] == count_dict[sorted_indices[0]]]
            top_distance = min(top_indices, key=distance_dict.get)
            sorted_indices.remove(top_distance)
            answers.append(top_distance)
        return answers
