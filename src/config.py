"""
main.py
Autor: JungwooChoi, HyeongwonKang
Incremental pseudo labeling for anomaly detection Argument Parser

"""
import argparse


def load_config():
    """
    argument parser

    """
    ap = argparse.ArgumentParser()

    ap.add_argument("-M", "--model", type=str, required=True, help="model(IF, 1SVM, AE, GAN)")
    ap.add_argument("-D", "--data", type=str, required=True, help="dataset")
    ap.add_argument("-L", "--ablabel", type=str, required=True, help="Abnormal label")
    ap.add_argument("-I", "--increment_method", type=str, required=True, help="increment-method")
    ap.add_argument("-P", "--increment_param", type=float, required=True, help="increment-parameter")
    ap.add_argument("-T", "--input_type", type=str, default='vector', help="input_type(vector, image)")

    args = vars(ap.parse_args())

    return args

def str2bool(v):
    if type(v) is not bool:
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    else:
        return v


