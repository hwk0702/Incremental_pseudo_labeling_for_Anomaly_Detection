from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def anomaly_dist(result_y, y_val, result_unk):
    # scaler = MinMaxScaler()
    # scaler.fit(result_y.reshape(-1, 1))
    #
    # result_y = scaler.transform(result_y.reshape(-1, 1))
    # result_unk = scaler.transform(result_unk.reshape(-1, 1))

    fig = plt.figure()
    sns.distplot(-result_y[y_val == 1], color='green',
                 label="normal")
    sns.distplot(-result_y[y_val == -1], color='red',
                 label="abnoraml")
    sns.distplot(result_unk, color='black', label="unknown")
    plt.legend()
    plt.xlabel("anomaly Score")
    plt.ylabel("Conunt")
    # plt.xlim(-0.2, 0.4)
    # plt.ylim(0, 40)
    return fig


def AUROC_curve(fpr, tpr, AUROC):
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(labels=[f"AUROC={AUROC}"])
    return fig