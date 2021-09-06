import pandas as pd
import numpy as np


def simple_inc(x_train, y_train, x_unk, y_unk, method_param):
    method_param = int(method_param)
    if method_param >= len(x_unk):
        return x_train, y_train, x_unk, y_unk, False
        # x_unk_add = x_unk[-method_param:]
    # x_unk = x_unk[:-method_param]
    x_unk_add = x_unk[:method_param]
    x_unk = x_unk[method_param:]
    x_train = np.concatenate((x_train, x_unk_add), axis=0)
    # y_unk_add = y_unk[-method_param:]
    # y_unk = y_unk[:-method_param]
    y_unk_add = y_unk[:method_param]
    y_unk = y_unk[method_param:]
    y_unk.reset_index(drop=True, inplace=True)
    y_train = np.concatenate((y_train, y_unk_add), axis=0)

    return x_train, y_train, x_unk, y_unk, True

def rate_inc(x_train, y_train, x_unk, y_unk, method_param):
    n_add = int(method_param * len(x_unk))
    if n_add < 1:
        return x_train, y_train, x_unk, y_unk, False
    x_unk_add = x_unk[-n_add:]
    x_unk = x_unk[:-n_add]
    x_train = pd.concat((x_train, x_unk_add))
    y_unk_add = y_unk[-n_add:]
    y_unk = y_unk[:-n_add]
    y_train = pd.concat((y_train, y_unk_add))

    return x_train, y_train, x_unk, y_unk, True

def threshold(x_train, y_train, x_unk, y_unk, method_param):
    # 수정해야


    return x_train, y_train, x_unk, y_unk, True

# UNK -> train 추가 방법
def inc_data(x_train, y_train, x_unk, y_unk, result_unk, method, method_param):

    reindex = np.argsort(-result_unk)

    x_unk = x_unk[reindex]
    y_unk = y_unk[reindex]

    return globals()[method](x_train, y_train, x_unk, y_unk, method_param)