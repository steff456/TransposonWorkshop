#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
from heapq import heappush, heappop

from solution.transposon import Transposon

#%% Cargar las anotaciones
anotaciones = pd.read_csv('../data/groundtruth.csv')

#%% Cargar los datos de predicción
pred_a = pd.read_csv('../data/predA.csv')
pred_b = pd.read_csv('../data/predB.csv')

#%% Pasar los datos a la clase transposon
def process_info(df):
    act_info = {}
    for _, row in df.iterrows():
        act_seq = row['seq_name']
        act_start = row['start']
        act_end = row['end']
        try:
            act_score = row['score']
        except:
            act_score = 0
        if act_seq in act_info:
            act_info[act_seq].append(
                Transposon(act_seq, act_start, act_end, act_score))
        else:
            act_info[act_seq] = [
                Transposon(act_seq, act_start, act_end, act_score)]
    return act_info

anotaciones = process_info(anotaciones)
pred_a = process_info(pred_a)
pred_b = process_info(pred_b)

#%% Calcular las metricas para un umbral dado


# Calcular verdaderos positivos, falsos positivos y falsos negativo
def get_single_instance_results(gts, preds, thresh):
    tp = 0
    IoUs = []
    gt_index = []
    pred_index = []
    for x, pred in enumerate(preds):
        for y, gt in enumerate(gts):
            act_overlap = pred.get_overlap(gt)
            IoU = act_overlap/(len(pred) + len(gt) - act_overlap)
            if IoU >= thresh:
                IoUs.append(IoU)
                gt_index.append(y)
                pred_index.append(x)
    IoUs = np.argsort(IoUs)[::-1]
    if len(IoUs) == 0:
        # No hay sobrelapamiento
        return 0, len(preds), len(gts)
    gt_match_idx = []
    pred_match_idx = []
    for idx in IoUs:
        gt_idx = gt_index[idx]
        pr_idx = pred_index[idx]
        if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
            gt_match_idx.append(gt_idx)
            pred_match_idx.append(pr_idx)
    tp = len(gt_match_idx)
    fp = len(preds) - len(pred_match_idx)
    fn = len(gts) - len(gt_match_idx)
    return tp, fp, fn


# Calcular precision
def calculate_precision(tp, fp):
    return tp/(tp + fp + 1e-9)


# Calcular cobertura
def calculate_recall(tp, fn):
    return tp/(tp + fn + 1e-9)


# Calcular F medida
def calculate_fmeasure(precision, recall):
    return (2*precision*recall)/(precision + recall + 1e-9)


# Calcular las metricas globales de las predicciones y las anotaciones
def calculate_metrics(gt, pred, thresh=0.5, verbose=True):
    names = list(set(gt.keys()).union(set(pred.keys())))
    cum_tp, cum_fp, cum_fn, total_p, total_gt = 0, 0, 0, 0, 0
    for seq_name in names:
        if seq_name not in pred:
            cum_fn += len(gt[seq_name])
            total_gt += len(gt[seq_name])
            continue
        elif seq_name not in gt:
            cum_fp += len(pred[seq_name])
            total_p += len(pred[seq_name])
            continue
        act_gt = gt[seq_name]
        act_pred = pred[seq_name]
        tp, fp, fn = get_single_instance_results(act_gt, act_pred, thresh)
        if False:
            precision = calculate_precision(tp, fp)
            recall = calculate_recall(tp, fn)
            print('----------- {} -----------'.format(seq_name))
            print('TP:', tp, 'FP:', fp, 'FN:', fn)
            print('precision', precision)
            print('recall', recall)
            print('F-measure', calculate_fmeasure(precision, recall))
            print('-----------')
        cum_tp += tp
        cum_fp += fp
        cum_fn += fn
        total_p += len(pred[seq_name])
        total_gt += len(gt[seq_name])
    precision = calculate_precision(cum_tp, cum_fp)
    recall = calculate_recall(cum_tp, cum_fn)
    fmeasure = calculate_fmeasure(precision, recall)
    if verbose:
        print('-----------')
        print('TP:', cum_tp, 'FP:', cum_fp, 'FN:', cum_fn)
        print('Total pred:', total_p, 'Total gt:', total_gt)
        print('threshold', thresh)
        print('precision', precision)
        print('recall', recall)
        print('F-measure', fmeasure)
        print('-----------')
    return precision, recall, fmeasure, cum_fp, cum_tp


# Sacar los resultados de nuestras predicciones
result_a = calculate_metrics(anotaciones, pred_a)
result_b = calculate_metrics(anotaciones, pred_b)

#%% Graficando las curvas de precisión/cobertura y ROC

# Puntajes de confianza de 0 a 16000 con pasos de 5
scores = np.linspace(0, 16000, 5)


# Calcular todas las métricas para multiples puntajes
def calculate_multiple_scores(gt, preds, thresh, scores):
    precisions = []
    recalls = []
    false_positives = []
    true_positives = []
    for score in scores:
        pred = preds
        total_p = []
        total_r = []
        for name in pred:
            for transposon in pred[name]:
                if transposon.score < score:
                    heappop(pred[name])
        precision, recall, fm, fp, tp = calculate_metrics(
            gt, pred, thresh=thresh)
        total_p.append(precision)
        total_r.append(recall)
        print('-------- Score {} ---------'.format(score))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('Fmeasure: {}'.format(fm))
        precisions.append(precision)
        recalls.append(recall)
        false_positives.append(fp)
        true_positives.append(tp)
    return precisions, recalls, false_positives, true_positives


# Función para generar la gráfica de precision cobertura
def plot_PR(precisions, recalls, threshs=0.5):
    plt.plot(recalls, precisions, label=threshs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve for Transposon Detection')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.show()


# Graficar los resultados de nuestras predicciones
grafica_a = calculate_multiple_scores(anotaciones, pred_a, 0.5, scores)
plot_PR(grafica_a[0], grafica_a[1], 0.5)

grafica_b = calculate_multiple_scores(anotaciones, pred_b, 0.5, scores)
plot_PR(grafica_b[0], grafica_b[1], 0.5)


# Función para generar la gráfica ROC
def plot_ROC(fpr, tpr):
    """Plot ROC curve."""
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                      estimator_name='TransposonFinder')
    display.plot()
    plt.show()


# Graficar los resultados de nuestras predicciones
plot_ROC(grafica_a[2], grafica_a[3])

plot_ROC(grafica_b[2], grafica_b[3])

