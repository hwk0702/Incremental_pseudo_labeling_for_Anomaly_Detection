def eval(y, predict):
    AUROC = 0
    TPR = 0
    FPR = 0
    TP = []
    FP = []
    dd = []

    for i in range(len(predict)):

        if y.iloc[i] == 1:
            TPR = TPR + 1
        else:
            FPR = FPR + 1
            AUROC = AUROC + (TPR / sum(predict == 1)) * (1 / sum(predict == 0))
        TP.append(TPR / sum(predict == 1))
        FP.append(FPR / sum(predict == 0))
        dd.append(i / len(predict))

    return AUROC, TP, FP, dd