import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    correct = 0
    for i in range(len(LPred)):
        #print(LPred[i])
        #print(LTrue[i])
        if LPred[i] == LTrue[i]:
            correct += 1
    acc = correct / len(LPred)
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    unique_labels = list(set(LTrue))
    matrix = [[0 for _ in range(len(unique_labels))] for _ in range(len(unique_labels))]
    for i in range(len(LPred)):
        matrix[unique_labels.index(LPred[i])][unique_labels.index(LTrue[i])] += 1

     
    cM = np.array(matrix)
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    length = cM.shape[0]
    
    correct = 0
    for i in range(length):
        correct += cM[i, i]
    acc = correct / cM.sum()
    # ============================================
    
    return cM
