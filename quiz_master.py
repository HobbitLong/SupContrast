import json
import os
import traceback
import typing
from datetime import datetime
from encoder_models.base_model import BaseModel


def quiz1(model: BaseModel, question_dir: str, file_name: typing.Optional[str] = None):
    """
    Code for solving question 1. Given 1 query image and 2 answer groups each containing 3 images, choose which
    image group is most similar to the query image.

    :param model: The model to solve the challenge.
    :param question_dir: The path to the directory containing the questions.
    """

    question_dirs = os.listdir(question_dir)
    file_name = file_name or f'{model.NAME}q1{str(datetime.now()).replace(" ", "").replace(":", "")}.csv'
    question_dirs.remove(BaseModel.CACHE_DIR)
    incorrect_questions = []
    total = 0
    correct = 0
    with open(file_name, 'w+') as fh:
        fh.write('question,category,gt,ans\n')
        fh.flush()
        for i, question_number in enumerate(question_dirs):
            print(f'Answering question {question_number}')
            qdir_path = os.path.join(question_dir, question_number)
            try:
                with open(os.path.join(qdir_path,
                                       f'{question_number}.json')) as q:
                    question_info = json.load(q)
                    category = question_info['category']
                    assert len(question_info['correct_answer_group_ID']) == 1
                    gt, = question_info['correct_answer_group_ID']
                    query_images = [
                        os.path.join(qdir_path, im_dict['image_url'])
                        for im_dict in question_info['Questions'][0]["images"]
                    ]
                    answer_images = [
                        os.path.join(qdir_path, image['image_url'])
                        for im_dict in question_info['Answers']
                        for image in im_dict['images']
                    ]

                    ans = model.question1(query_images, answer_images)

            except Exception:
                traceback.print_exc()
                print(f'Invalid question for {question_number}')
                incorrect_questions.append(question_number)
            else:
                fh.write(f'{question_number},{category},{gt},{ans}\n')
                fh.flush()
                total += 1
                if ans == gt:
                    correct += 1
                print(f'Accuracy after {i} is {correct / total}')
        print('These questions are incorrect')
        print(incorrect_questions)


def quiz2(model: BaseModel,
          question_dir: str,
          quiz_name: typing.Optional[str] = 'q2',
          file_name: typing.Optional[str] = None,
          ):
    """
    Code for solving question 2. Given 3 query images and 1 answer image, decide if the query images are similar to
    the answer image.

    :param model: The model to solve the challenge.
    :param question_dir: The path to the directory containing the questions.
    """

    question_dirs = os.listdir(question_dir)
    file_name = file_name or f'{model.NAME}{quiz_name}{str(datetime.now()).replace(" ", "").replace(":", "")}.csv'
    question_dirs.remove(BaseModel.CACHE_DIR)
    incorrect_questions = []
    total = 0
    correct = 0
    with open(file_name, 'w+') as fh:
        fh.write('question,category,gt,ans\n')
        fh.flush()
        for i, question_number in enumerate(question_dirs):
            print(f'Answering question {question_number}')
            qdir_path = os.path.join(question_dir, question_number)
            try:
                with open(os.path.join(qdir_path,
                                       f'{question_number}.json')) as q:
                    question_info = json.load(q)
                    category = question_info['category']
                    gt = question_info['is_correct']
                    query_images = [
                        os.path.join(qdir_path, im_dict['image_url'])
                        for im_dict in question_info['Questions'][0]["images"]
                    ]
                    answer_images = [
                        os.path.join(qdir_path, image['image_url'])
                        for im_dict in question_info['Answers']
                        for image in im_dict['images']
                    ]

                    ans = model.question2(query_images, answer_images)
            except Exception:
                traceback.print_exc()
                print(f'Invalid question for {question_number}')
                incorrect_questions.append(question_number)
            else:
                fh.write(f'{question_number},{category},{gt},{ans}\n')
                fh.flush()
                total += 1
                if ans == gt:
                    correct += 1
                print(f'Accuracy after {i} is {correct / total}')
        print('These questions are incorrect')
        print(incorrect_questions)


def quiz3(model: BaseModel, question_dir: str, file_name: typing.Optional[str] = None):
    """
    Code for solving question 3. Given 2 groups of query image pick which group has similar characteristics. Next given
    3 groups of answer images, choose which group is most similar to the chosen query image group.


    :param model: The model to solve the challenge.
    :param question_dir: The path to the directory containing the questions.
    """

    question_dirs = os.listdir(question_dir)
    file_name = file_name or f'{model.NAME}q3{str(datetime.now()).replace(" ", "").replace(":", "")}.csv'
    question_dirs.remove(BaseModel.CACHE_DIR)
    incorrect_questions = []
    total = 0
    correct = 0
    with open(file_name, 'w+') as fh:
        fh.write('question,category,qgt,agt,qans,ans\n')
        fh.flush()
        for i, question_number in enumerate(question_dirs):
            print(f'Answering question {question_number}')
            qdir_path = os.path.join(question_dir, question_number)
            try:
                with open(os.path.join(qdir_path,
                                       f'{question_number}.json')) as q:
                    question_info = json.load(q)
                    category = question_info['category']
                    assert len(question_info['correct_question_group_ID']) == 1
                    gt = question_info['correct_answer_group_ID']
                    question_gt, = question_info['correct_question_group_ID']
                    query1, query2 = [
                        [os.path.join(qdir_path, im['image_url']) for im in im_dict["images"]]
                        for im_dict in question_info['Questions']
                    ]
                    answer_images = [
                        os.path.join(qdir_path, image['image_url'])
                        for im_dict in question_info['Answers']
                        for image in im_dict['images']
                    ]

                    query_answer, ans = model.question3(query1, query2, answer_images)
                    group_id = question_info['Questions'][query_answer]["group_id"]

            except Exception:
                traceback.print_exc()
                print(f'Invalid question for {question_number}')
                incorrect_questions.append(question_number)
            else:
                fh.write(f'{question_number},{category},{question_gt},{gt},{group_id},{ans}\n')
                fh.flush()
                total += 1
                if ans == gt:
                    correct += 1
                print(f'Accuracy after {i} is {correct / total}')
        print('These questions are incorrect')
        print(incorrect_questions)


def quiz4(model: BaseModel,
          question_dir: str,
          quiz_name: typing.Optional[str] = 'q4',
          file_name: typing.Optional[str] = None):
    """
    Code for solving question 4. Given 3 query images and 5 answer images, choose the 2 answer images most similar to
    query images.

    :param model: The model to solve the challenge.
    :param question_dir: The path to the directory containing the questions.
    """

    question_dirs = os.listdir(question_dir)
    file_name = file_name or f'{model.NAME}{quiz_name}{str(datetime.now()).replace(" ", "").replace(":", "")}.csv'
    question_dirs.remove(BaseModel.CACHE_DIR)
    incorrect_questions = []
    total = 0
    correct = 0
    with open(file_name, 'w+') as fh:
        fh.write('question,category,gt1,gt2,ans1,ans2\n')
        fh.flush()
        for i, question_number in enumerate(question_dirs):
            print(f'Answering question {question_number}')
            qdir_path = os.path.join(question_dir, question_number)
            try:
                with open(os.path.join(qdir_path,
                                       f'{question_number}.json')) as q:
                    question_info = json.load(q)
                    category = question_info['category']
                    assert len(question_info['correct_answer_group_ID']) == 2
                    gt1, gt2 = question_info['correct_answer_group_ID']
                    query_images = [
                        os.path.join(qdir_path, im_dict['image_url'])
                        for im_dict in question_info['Questions'][0]["images"]
                    ]
                    answer_images = [
                        os.path.join(qdir_path, im_dict['images'][0]['image_url'])
                        for im_dict in question_info['Answers']
                    ]

                    ans1, ans2 = model.question4(query_images, answer_images)
            except Exception:
                traceback.print_exc()
                print(f'Invalid question for {question_number}')
                incorrect_questions.append(question_number)
            else:
                fh.write(f'{question_number},{category},{gt1},{gt2},{ans1},{ans2}\n')
                fh.flush()
                total += 2
                if ans1 in [gt1, gt2]:
                    correct += 1
                if ans2 in [gt1, gt2]:
                    correct += 1
                print(f'Accuracy after {i} is {correct / total}')
        print('These questions are incorrect')
        print(incorrect_questions)


def group2(model: BaseModel, question_dir: str, file_name: typing.Optional[str] = None):
    """
    Code for solving group 2 questions. Given 1 query image and 3 answer images, choose which
    answer group is most similar to the query image.

    :param model: The model to solve the challenge.
    :param question_dir: The path to the directory containing the questions.
    """

    question_dirs = os.listdir(question_dir)
    file_name = file_name or f'{model.NAME}gruop2{str(datetime.now()).replace(" ", "").replace(":", "")}.csv'
    question_dirs.remove(BaseModel.CACHE_DIR)
    incorrect_questions = []
    total = 0
    correct = 0
    with open(file_name, 'w+') as fh:
        fh.write('question,category,gt,ans\n')
        fh.flush()
        for i, question_number in enumerate(question_dirs):
            print(f'Answering question {question_number}')
            qdir_path = os.path.join(question_dir, question_number)
            try:
                with open(os.path.join(qdir_path,
                                       f'{question_number}.json')) as q:
                    question_info = json.load(q)
                    category = question_info['category']
                    assert len(question_info['correct_answer_group_ID']) == 1
                    gt, = question_info['correct_answer_group_ID']
                    query_images = [
                        os.path.join(qdir_path, im_dict['image_url'])
                        for im_dict in question_info['Questions'][0]["images"]
                    ]
                    answer_images = [
                        os.path.join(qdir_path, image['image_url'])
                        for im_dict in question_info['Answers']
                        for image in im_dict['images']
                    ]

                    ans = model.group2(query_images, answer_images)

            except Exception:
                traceback.print_exc()
                print(f'Invalid question for {question_number}')
                incorrect_questions.append(question_number)
            else:
                fh.write(f'{question_number},{category},{gt},{ans}\n')
                fh.flush()
                total += 1
                if ans == gt:
                    correct += 1
                print(f'Accuracy after {i} is {correct / total}')
        print('These questions are incorrect')
        print(incorrect_questions)


def group4_question1(model: BaseModel, question_dir: str, file_name: typing.Optional[str] = None):
    """
    Code for solving quiz 4 question 1. Given 3 query images and 1 answer image, decide if the query images are similar to
    the answer image.

    :param model: The model to solve the challenge.
    :param question_dir: The path to the directory containing the questions.
    """
    quiz2(model, question_dir, 'quiz4_q1', file_name=file_name)


def group4_question2(model: BaseModel, question_dir: str, file_name: typing.Optional[str] = None):
    """
    Code for solving quiz 4 question 2. Given 3 query images and 5 answer images, choose the 2 answer images most similar to
    query images.

    :param model: The model to solve the challenge.
    :param question_dir: The path to the directory containing the questions.
    """
    quiz4(model, question_dir, 'quiz4_q2', file_name=file_name)
