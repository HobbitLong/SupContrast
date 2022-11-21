from __future__ import print_function

import argparse
from question_loader import Question1Dataset, Question2Dataset, Question3Dataset, Question4Dataset, Group2Dataset
import wandb
from calculator import calculate1, calculate3, calculate4, calculate4_both
from quiz_master import quiz3, quiz1, quiz2,  quiz4, group2, group4_question1, group4_question2
from encoder_models.vit import RankVit, ClusterVit

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

MODELS = {
    'vit': ClusterVit,
    'rankvit': RankVit
}

CALCULATOR = {
    1: calculate1,
    2: calculate1,
    3: calculate3,
    4: calculate4,
    5: calculate1,
    7: calculate1,
    8: calculate4_both,
}

QUIZ_OPTIONS = {
    1: quiz1,
    2: quiz2,
    3: quiz3,
    4: quiz4,
    5: group2,
    7: group4_question1,
    8: group4_question2,
}

def f1_graph(question_number,model_name,weights_path,opt,question_dir,file_name,title):
    score_model = MODELS[model_name](
            weights_path=weights_path,
            mean=eval(opt.mean),
            std=eval(opt.std),
            question_dir=question_dir,
        )

    score_model.encode_images()

    QUIZ_OPTIONS[question_number](
        score_model,
        question_dir,
        file_name,
    )
    var=CALCULATOR[question_number](
        answer_file = file_name,
    )
    var = list(var)
    print({'micro_f1'+title: var[0]})
    print({'macro_f1'+title: var[1]})

def main():

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--mean', type=str,default="(0.6958, 0.6816, 0.6524)")
    parser.add_argument('--std', type=str,default= "(0.3159, 0.3100, 0.3385)")
    parser.add_argument('--weights_path', type=str,default= "")
    parser.add_argument('--root_dir', type=str,default= "/media0/chris/group4_resize_v2")
    parser.add_argument('--model_name', type=str,default= "vit")
    parser.add_argument('--valid_path', type=str,default= "valid")
    parser.add_argument('--test_path', type=str,default= "test")

    opt = parser.parse_args()

    file_name = opt.weights_path.replace('pth','csv')

    question1_val_dir = f"{opt.root_dir}/{opt.valid_path}/question1"
    f1_graph(7, opt.model_name, opt.weights_path, opt, question1_val_dir, file_name, "question1_valid")

    question1_tst_dir = f"{opt.root_dir}/{opt.test_path}/question1"
    f1_graph(7, opt.model_name, opt.weights_path, opt,question1_tst_dir, file_name, "question1_test")

    question2_val_dir = f"{opt.root_dir}/{opt.valid_path}/question2"
    f1_graph(8, opt.model_name, opt.weights_path, opt,question2_val_dir, file_name, "question2_valid")

    question2_tst_dir = f"{opt.root_dir}/{opt.test_path}/question2"
    f1_graph(8, opt.model_name, opt.weights_path, opt,question2_tst_dir, file_name, "question2_test")

if __name__ == '__main__':
    main()