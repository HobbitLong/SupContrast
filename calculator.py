
import typing
import sys

def increment(*args) -> typing.List[int]:
    """Increment all given numbers"""
    return [arg + 1 for arg in args]


def print_results(TTP: int, TFP: int, TFN: int, result_dict: typing.Dict) -> typing.Tuple[float, float]:
    """Print results."""
    fscores = []
    for k, v in result_dict.items():
        TP = v['TP']
        FP = v['FP']
        FN = v['FN']
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        try:
            f = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f = 0
        print(f'{k}: {f}')
        fscores.append(f)
    print(f"Macro f score: {sum(fscores) / len(fscores)}")

    precision = TTP / (TTP + TFP)
    recall = TTP / (TTP + TFN)

    f = 2 * ((precision * recall) / (precision + recall))
    print(f'Micro f score {f}')
    return sum(fscores) / len(fscores), f


def calculate1(answer_file: str) -> typing.Tuple[float, float]:
    """Calculate Fscore for problem1"""
    TTP, TFP, TFN = 0, 0, 0  # Total true positive, true negatives, false positives
    result_dict = {}
    header_skipped = False
    with open(answer_file) as fh:
        for line in fh:
            if not header_skipped:
                header_skipped = True
                continue
            _, cat, gt, ans = line.replace('\n', "").split(',')
            _, cat, gt, ans = _, cat, int(gt), int(bool(ans))
            print(f'gt->{gt}, ans->{ans}', gt == ans )
            if cat not in result_dict:
                result_dict[cat] = dict(TP=0, FN=0, FP=0)
            if ans == gt:
                result_dict[cat]['TP'], TTP = increment(result_dict[cat]['TP'], TTP)
            else:
                result_dict[cat]['FP'], TFP,  result_dict[cat]['FN'], TFN = \
                    increment(result_dict[cat]['FP'], TFP,  result_dict[cat]['FN'], TFN)
    return print_results(TTP, TFP, TFN, result_dict)


def calculate3(answer_file: str) -> typing.Tuple[float, float]:
    """Calculate Fscore for problem3"""
    TTP, TFP, TFN = 0, 0, 0  # Total true positive, true negatives, false positives
    result_dict = {}
    header_skipped = False
    with open(answer_file) as fh:
        for line in fh:
            if not header_skipped:
                header_skipped = True
                continue
            _, cat, qgt, agt, qans, ans = line.replace('\n', "").split(',')
            if cat not in result_dict:
                result_dict[cat] = dict(TP=0, FN=0, FP=0)
            if ans == agt:
                result_dict[cat]['TP'], TTP = increment(result_dict[cat]['TP'], TTP)
            else:
                result_dict[cat]['FP'], TFP,  result_dict[cat]['FN'], TFN = \
                    increment(result_dict[cat]['FP'], TFP,  result_dict[cat]['FN'], TFN)
    return print_results(TTP, TFP, TFN, result_dict)


def calculate4(answer_file: str) -> typing.Tuple[float, float]:
    """Calculate Fscore for problem4"""
    TTP, TFP, TFN = 0, 0, 0  # Total true positive, true negatives, false positives
    result_dict = {}
    header_skipped = False
    with open(answer_file) as fh:
        for line in fh:
            if not header_skipped:
                header_skipped = True
                continue
            _, cat, gt1, gt2, ans1, ans2 = line.replace('\n', "").split(',')
            if cat not in result_dict:
                result_dict[cat] = dict(TP=0, FN=0, FP=0)
            for ans in ans1, ans2:
                if ans in [gt1, gt2]:
                    result_dict[cat]['TP'], TTP = increment(result_dict[cat]['TP'], TTP)
                else:
                    result_dict[cat]['FP'], TFP,  result_dict[cat]['FN'], TFN = \
                        increment(result_dict[cat]['FP'], TFP,  result_dict[cat]['FN'], TFN)
    return print_results(TTP, TFP, TFN, result_dict)

def calculate4_both(answer_file: str) -> typing.Tuple[float, float]:
    """Calculate Fscore for problem4"""
    TTP, TFP, TFN = 0, 0, 0  # Total true positive, true negatives, false positives
    result_dict = {}
    header_skipped = False
    with open(answer_file) as fh:
        for line in fh:
            if not header_skipped:
                header_skipped = True
                continue
            _, cat, gt1, gt2, ans1, ans2 = line.replace('\n', "").split(',')
            if cat not in result_dict:
                result_dict[cat] = dict(TP=0, FN=0, FP=0)
            if sorted([ans1,ans2]) == sorted([gt1,gt2]):
                result_dict[cat]['TP'], TTP = increment(result_dict[cat]['TP'], TTP)
            else:
                result_dict[cat]['FP'], TFP,  result_dict[cat]['FN'], TFN = \
                    increment(result_dict[cat]['FP'], TFP,  result_dict[cat]['FN'], TFN)
    return print_results(TTP, TFP, TFN, result_dict)