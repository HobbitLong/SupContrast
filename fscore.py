import argparse

from calculator import calculate1, calculate3, calculate4


CALCULATOR = {
    1: calculate1,
    2: calculate1,
    3: calculate3,
    4: calculate4,
    5: calculate1,
    7: calculate1,
    8: calculate4,
}


def main(
        answer_file: str,
        question_number: int,
):
    CALCULATOR[question_number](
        answer_file=answer_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_file', type=str)
    parser.add_argument('--question_number', type=int)
    args = parser.parse_args()
    main(
        answer_file=args.answer_file,
        question_number=args.question_number,
    )
