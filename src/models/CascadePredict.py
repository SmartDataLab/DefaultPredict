import numpy as np


class CascadePredictor(object):
    def __init__(self, models, certain_fun, certain_label):
        self.models = models
        self.certain_fun = certain_fun
        self.certain_label = certain_label

    def fit_list(self, feature_list, label_list):
        for i in range(len(self.models)):
            self.models[i].fit(feature_list[i], label_list[i])

    def predict_proba(self, feature):
        prob = [0] * len(feature)
        remain_idxs = list(range(len(prob)))
        for i, model in enumerate(self.models[:-1]):
            prob_tmp = model.predict_proba(feature.iloc[remain_idxs, :])
            certain_inner_idxs = [
                remain_idxs[k]
                for k, one in enumerate(prob_tmp)
                if self.certain_fun(one[1])
            ]
            for j, idx in enumerate(certain_inner_idxs):
                prob[idx] = prob_tmp[j]
                remain_idxs.pop(remain_idxs.index(idx))
        prob_tmp = self.models[-1].predict_proba(feature.iloc[remain_idxs, :])
        for j, idx in enumerate(remain_idxs):
            prob[idx] = prob_tmp[j]
        return np.array(prob)

    def predict(self, feature):
        return [
            self.certain_label if self.certain_fun(one) else 1.0 - self.certain_label
            for one in self.predict_proba(feature)[:, 1]
        ]
