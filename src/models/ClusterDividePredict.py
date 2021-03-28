import numpy as np


class ClusterPredictor(object):
    def __init__(self, cluster_model, predict_models):
        self.cluster_model = cluster_model
        self.cluster_num = self.cluster_model.n_clusters
        self.predict_models = predict_models
        if len(predict_models) != self.cluster_num:
            assert EnvironmentError

    def fit(self, feature, label):
        self.cluster_model.fit(feature)
        cluster_pred = self.cluster_model.predict(feature)
        for i in range(self.cluster_num):
            idxs = [
                j
                for j, cluster_category in enumerate(cluster_pred)
                if cluster_category == i
            ]
            self.predict_models[i].fit(feature.iloc[idxs, :], label[idxs])

    def predict_proba(self, feature):
        cluster_pred = self.cluster_model.predict(feature)
        prob = [0] * len(feature)
        for i in range(self.cluster_num):
            idxs = [
                j
                for j, cluster_category in enumerate(cluster_pred)
                if cluster_category == i
            ]
            prob_tmp = self.predict_models[i].predict_proba(feature.iloc[idxs, :])
            for j, idx in enumerate(idxs):
                prob[idx] = prob_tmp[j]
        return np.array(prob)

    def predict(self, feature):
        return self.predict_proba(feature)[:, 1] > 0.5
