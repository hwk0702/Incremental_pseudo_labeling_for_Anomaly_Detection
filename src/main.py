"""
main.py
Autor: JungwooChoi, HyeongwonKang
Incremental_pseudo_labeling_for_Anomaly_Detection
예시 : python main.py
"""

from pandas.core.reshape.merge import merge
from numpyencoder import NumpyEncoder
from config import load_config, str2bool
from preprocessing import normalize, CustomDataset
from util.check_mail import send_mail
from util.splitDataset import getDatasets
from util.plotting import anomaly_dist, AUROC_curve
from incMethod import inc_data
from util.eval import eval
from util.utils import EarlyStopping, set_seed
from model.IF import IF
from model.OCSVM import OCSVM
from model.AE import AutoEncoder
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from einops import rearrange, repeat
import warnings
import logging
import json 
import yaml
import sys
import os
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
import pdb

warnings.filterwarnings('ignore')


def main_iter(k, data, isLabelRatioChg, labelCol, norm,
              save_path, dataset, model_name, method, ab_label,
              method_param, output_path, model_params, input_type, image_shape=None):
    """
        main iteration

        Arguments
        ---------
        - k               : iteration number,
        - data            : data,
        - isLabelRatioChg : ,
        - labelCol        : ,
        - norm            : ,
        - save_path       : ,
        - dataset         : ,
        - model_name      : ,
        - method          : ,
        - ab_label        : ,
        - method_param    : ,
        - output_path     : ,
        - model_params    : ,

        Return
        ------
        - None
    """
    
    if (isLabelRatioChg):
        # normal Label 이 Major가 아닌 경우, label 비율 수정
        dataNormal = data[data[labelCol] == k].copy()
        dataNormal[labelCol] = dataNormal.apply(lambda x: 1, axis=1)
        dataAb = data[data[labelCol] != k].sample(n=round(len(dataNormal) / 10), random_state=k)
        dataAb[labelCol] = dataAb.apply(lambda x: -1, axis=1)
        Data = pd.concat([dataNormal, dataAb])
        X = Data[Data.columns.difference([labelCol])]
        y = Data[labelCol]
    else:
        X = data[data.columns.difference([labelCol])]
        y = data[labelCol]

    datasets = getDatasets(k, X, y)
    x_train = datasets['x_train']
    y_train = datasets['y_train']
    x_train = x_train[y_train == 1]
    y_train = y_train[y_train == 1]
    x_unk = datasets['x_unk']
    y_unk = datasets['y_unk']
    x_test = datasets['x_test']
    y_test = datasets['y_test']
    x_val = datasets['x_val']
    y_val = datasets['y_val']

    
    # normalize
    if norm and input_type=='vector':
        scaler = normalize(x_train)
        x_train = scaler.transform(x_train)
        x_unk = scaler.transform(x_unk)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val)

    if input_type=='image':
        x_train = rearrange(x_train, 'b (c h h) -> b c h h', c=image_shape[0])
        x_unk = rearrange(x_unk, 'b (c h h) -> b c h h', c=image_shape[0])
        x_test = rearrange(x_test, 'b (c h h) -> b c h h', c=image_shape[0])
        x_val = rearrange(x_val, 'b (c h h) -> b c h h', c=image_shape[0])


    model_save_path = f'{save_path}/{dataset}/{model_name}/{method}/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.exists(model_save_path + 'img/'):
        os.makedirs(model_save_path + 'img/')

    early_stopping = EarlyStopping(patience=5, verbose=1)
    repeat = True
    history = dict()
    num_repeat = 0

    # train and test
    logger.info('Training Start!')
    logger.info(f'\n iter : {k}, data : {dataset}, model : {model_name}, abnormal label : {ab_label}, '
                f'increment method : {method}, increment parameter : {method_param}')

    if model_name == 'AutoEncoder':
        model_params['units'].insert(0, x_train.shape[1])
        model_params['units'].append(x_train.shape[1])

    while repeat:
        num_repeat += 1
        model = globals()[model_name](**model_params)
        if model_name in ['IF', 'OCSVM']:
            model.train(x_train)
        else:
            params = {'batch_size': 64,
                      'shuffle': True,
                      'num_workers': 4,
                      'pin_memory': True}
            tr_dataset = CustomDataset(x_train, y_train)
            trainloader = DataLoader(dataset=tr_dataset, **params)
            # valid_dataset = CustomDataset(x_val[y_val == 1], y_val[y_val == 1])
            # validloader = DataLoader(dataset=valid_dataset)
            model.train(trainloader, x_val[y_val == 1])

        model.save(model_save_path + f'{method}_{method_param}_K{k}_{num_repeat}')
        result_val = model.validation(x_val)

        fpr, tpr, _ = roc_curve(y_val, result_val)
        roc_auc = auc(fpr, tpr)
        # roc_auc, tpr, fpr, dd = eval(y_val, result_val)

        logger.info(f'num of repeat : {num_repeat}, AUROC(val) : {roc_auc}')

        result_unk = model.test(x_unk)

        ano_dist = anomaly_dist(result_val, y_val, result_unk)
        curve = AUROC_curve(fpr, tpr, roc_auc)
        ano_dist.savefig(model_save_path + f'img/{method}_{method_param}_K{k}_{num_repeat}_ano.png')
        curve.savefig(model_save_path + f'img/{method}_{method_param}_K{k}_{num_repeat}_curve.png')

        result_test = model.validation(x_test)
        test_fpr, test_tpr, _ = roc_curve(y_test, result_test)
        test_roc_auc = auc(test_fpr, test_tpr)

        logger.info(f'num of repeat : {num_repeat}, AUROC(test) : {test_roc_auc}')

        curve = AUROC_curve(test_fpr, test_tpr, test_roc_auc)
        curve.savefig(model_save_path + f'img/{method}_{method_param}_K{k}_{num_repeat}_curve_test.png')

        history.setdefault('val_roc_auc', []).append(roc_auc)
        history.setdefault('val_fpr', []).append(fpr)
        history.setdefault('val_tpr', []).append(tpr)
        history.setdefault('test_roc_auc', []).append(test_roc_auc)
        history.setdefault('test_fpr', []).append(test_fpr)
        history.setdefault('test_tpr', []).append(test_tpr)
        # history.setdefault('anomaly_dist', []).append(ano_dist)
        # history.setdefault('AUROC_curve', []).append(curve)

        #if early_stopping.validate(-roc_auc):
        #   break
        logger.info(f'===========================================================================')

        logger.info(f'[Before] Train : {Counter(y_train)}, Unlabeld : {Counter(y_unk)}')
        x_train, y_train, x_unk, y_unk, repeat = inc_data(x_train, y_train,
                                                          x_unk, y_unk,
                                                          result_unk,
                                                          method, method_param)

        logger.info(f'[After]  Train : {Counter(y_train)}, Unlabeld : {Counter(y_unk)}')
        logger.info(f'===========================================================================')

    result_path = f'{output_path}/{dataset}/{model_name}/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + f'{method}_{method_param}_val_{k}.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, cls=NumpyEncoder, indent="\t")


def main(args, config):
    # arg parser
    model_name = args['model']
    dataset = args['data']
    ab_label = args['ablabel']
    method = args['increment_method']
    method_param = args['increment_param']
    input_type = args['input_type']

    # yaml parser
    norm = config['norm']
    workers = config['workers']
    model_params = config[model_name]
    output_path = config['output_path']
    save_path = config['save_path']
    senderAddr = config['senderAddr']
    recipientAddr = config['recipientAddr']
    password = config['password']

    # data load
    print ("dataset loading...  :",dataset)
    data = pd.read_pickle(config['dataset'][dataset]['path'])
    labelCol = config['dataset'][dataset]['labelCol']

    if 'cat_cols' in config['dataset'][dataset].keys():
        data = pd.get_dummies(data, columns=config['dataset'][dataset]['cat_cols'])

    #normal Label 이 Majority가 아닌 경우, label 비율 수정
    isLabelRatioChg = config['dataset'][dataset]['isLabelRatioChg']
    if (isLabelRatioChg == False):
        # class 선택해서 normal, abnormal Label로 변경
        ab_label = int(ab_label) if ab_label in list(map(lambda x: str(x), range(10))) else ab_label
        data[labelCol] = data.apply(lambda x: -1 if x[labelCol] == ab_label else 1, axis=1)

    iter_num = 10
    if (dataset == 'cifa100'):
        iter_num = 20

    if input_type == 'image':
        image_shape = config['dataset'][dataset]['image_shape']
    else:
        image_shape = None


    if model_name in ['IF', 'OCSVM']:
        with Pool(workers or cpu_count()) as pool:
            pool.imap(
                func=partial(main_iter,
                             data=data, isLabelRatioChg=isLabelRatioChg, labelCol=labelCol, norm=norm,
                             save_path=save_path, dataset=dataset, model_name=model_name, method=method, ab_label=ab_label,
                             method_param=method_param, output_path=output_path, model_params=model_params,
                             input_type=input_type, image_shape=image_shape),
                iterable=range(iter_num)
            )
            pool.close()
            pool.join()
    else:
        for k in range(iter_num):
            main_iter(k,
                      data=data, isLabelRatioChg=isLabelRatioChg, labelCol=labelCol, norm=norm,
                      save_path=save_path, dataset=dataset, model_name=model_name, method=method, ab_label=ab_label,
                      method_param=method_param, output_path=output_path, model_params=model_params,
                      input_type=input_type, image_shape=image_shape
                      )

    subject = f'model : {model_name}, Dataset : {dataset}, Inc_method : {method}, method_param : {method_param}'
    text = f'model : {model_name}, Dataset : {dataset}, Inc_method : {method}, method_param : {method_param}'
    #send_mail(senderAddr, recipientAddr, password, subject, text)

if __name__ == '__main__':
    # logger 세팅
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./log/IPLAD.log')
    formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # arg parser
    args = load_config()

    # yaml file
    with open('config.yaml') as f:
        config = yaml.load(f)

    # main
    try:
        main(args, config)
    except Exception as e:
        logger.error(e)
        logger.exception("error")


