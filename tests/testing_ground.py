import numpy as np


def test(*args):
    for i in range(len(args)):
        args.at[i] = args.at[i] + 1

    return args


print(test(1, 2, 3))
