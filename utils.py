import torch
import torch.nn as nn

standard = 35.0
space = 2.0
shortTermWindowSize = 10
historyWindowSize = 40


def enforce_feature(raw_feature):
    weight_feature = raw_feature[:, -1]
    length = weight_feature.shape[0]
    extend_feat_tensor = torch.zeros((length, 9))
    for i in range(historyWindowSize, length):
        history_feat = weight_feature[i-historyWindowSize:i]
        extend_feat_tensor[i] = getExtendFeat(history_feat)
    res = torch.cat((raw_feature, extend_feat_tensor), dim=1)
    return res


def getExtendFeat(raw_feature):
    shortTermFeats = raw_feature[(-1 - shortTermWindowSize):-1]

    tolerance = space * 0.25

    highBound = standard + space
    weakHighBound = highBound - tolerance
    strongHighBound = highBound + tolerance
    lowBound = standard - space
    weakLowBound = lowBound + tolerance
    strongLowBound = lowBound - tolerance
    exceedLabel = torch.zeros(5)

    # Long term features
    ave = torch.mean(raw_feature)
    std = torch.std(raw_feature)
    qualUp = torch.quantile(raw_feature, 0.75)
    qualDown = torch.quantile(raw_feature, 0.25)

    # Short term features
    for shortTermData in shortTermFeats:
        if shortTermData >= strongHighBound:
            exceedLabel[4] += 1
        elif shortTermData >= weakHighBound:
            exceedLabel[3] += 1
        elif shortTermData >= weakLowBound:
            exceedLabel[2] += 1
        elif shortTermData >= strongLowBound:
            exceedLabel[1] += 1
        else:
            exceedLabel[0] += 1

    feat_vector = torch.tensor([ave, std, qualUp, qualDown])
    feat_vector = torch.cat((feat_vector, exceedLabel))
    return feat_vector

