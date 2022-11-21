
import argparse
import os.path
import typing
from datetime import datetime
import glob

from encoder_models.vit import RankVit, ClusterVit
from encoder_models.resnet import Resnet
from quiz_master import quiz3, quiz1, quiz2,  quiz4, group2, group4_question1, group4_question2
from calculator import calculate1, calculate3, calculate4

MODELS = {
    'vit': ClusterVit,
    'resnet': Resnet,
    'rankvit': RankVit
}

CALCULATOR_OPTIONS = {
    1: calculate1,
    2: calculate1,
    3: calculate3,
    4: calculate4,
    5: calculate1,
    7: calculate1,
    8: calculate4,
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


def get_sorted_weights(weights_dir: str) -> typing.List[typing.Tuple[str, int]]:
    """Get weights files."""
    weights_files = list(glob.glob(f"{weights_dir}/*.pth"))
    weights_files = [(weights,  int(weights.replace('.pth', "").split("_")[-1])) for weights in weights_files]
    return sorted(weights_files,
                  key=lambda x: x[-1])


def main(
        model_name: str,
        weights_path: str,
        question_number: int,
        question_dir: str,
        mean: typing.List[float],
        std: typing.List[float],
        **kwargs
):
    if not os.path.isdir(weights_path):
        model = MODELS[model_name](
            weights_path=weights_path,
            mean=mean,
            std=std,
            question_dir=question_dir,
        )
        model.encode_images()
        QUIZ_OPTIONS[question_number](
            model,
            question_dir
        )
    else:
        with open(f"question_number{question_number}_{str(datetime.now())[:-7].replace(' ', '')}.csv", 'w+') as fh:
            fh.write('epoch,fmacro,fmicro\n')
            fh.flush()
            for weight, epoch in get_sorted_weights(weights_path):
                result_file = weight.replace('pth', 'csv')
                model = MODELS[model_name](
                    weights_path=weight,
                    mean=mean,
                    std=std,
                    question_dir=question_dir,
                )
                model.encode_images()
                QUIZ_OPTIONS[question_number](
                    model,
                    question_dir,
                    file_name=result_file
                )
                fmacro, fmicro = CALCULATOR_OPTIONS[question_number](result_file)
                fh.write(f'{epoch},{fmacro},{fmicro}\n')
                fh.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit', help='Choose vit, simclr or resnet')
    parser.add_argument('--weights_path', type=str, default="")
    parser.add_argument('--question_number', type=int)
    parser.add_argument('--question_dir', type=str)
    parser.add_argument('--mean', type=str)
    parser.add_argument('--std', type=str)
    parser.add_argument('--additional_args', type=str, default="",
                        help="If your model requires additional args")
    args = parser.parse_args()
    additional_args = {}
    if args.additional_args and "=" in args.additional_args:
        for arg_group in args.additional_args.split(','):
            k, v = args.additional_args.split('=')
            additional_args[k] = v
    print('args.mean->', args.mean)
    main(
        model_name=args.model,
        weights_path=args.weights_path,
        question_number=args.question_number,
        question_dir=args.question_dir,
        mean=eval(args.mean),
        std=eval(args.std),
        **additional_args
    )
