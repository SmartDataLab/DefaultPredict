import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import SR1


class TransferLogistic(object):
    def __init__(self, model, mu=None, beta=1.5):
        self.model = model
        self.global_mu = mu
        self.prev_proba = None
        self.beta = beta

    def mu_fun(self, beta_x, beta):
        return 1 / (1 + np.exp(-(beta_x + beta)))

    def target_fun(self, beta):
        beta_x = np.log(self.prev_proba / (1 - self.prev_proba))
        self.global_mu = self.mu_fun(beta_x, beta)
        # mu = self.mu_fun(beta_x, beta)
        sum_ = np.sum(
            self.y * np.log(self.global_mu) + (1 - self.y) * np.log(1 - self.global_mu)
        )
        return -sum_

    def fit(self, feature, label):
        self.prev_proba = self.model.predict_proba(feature)[:, 1]
        self.y = label.astype(int)
        self.optimize_res = minimize(
            self.target_fun, self.beta, jac="2-point", hess=SR1(), method="trust-constr"
        )
        print(self.optimize_res)
        print(self.global_mu)
        self.beta_ = self.optimize_res["x"][0]
        # self.p_value = self.logit_pvalue(self.predict_proba(feature), feature)

    def predict_proba(self, feature):
        proba = self.model.predict_proba(feature)[:, 1]
        beta_x = np.log(proba / (1 - proba))

        p = self.mu_fun(beta_x, self.beta_)
        res_p = np.zeros((p.shape[0], 2))
        res_p[:, 0] = 1 - p
        res_p[:, 1] = p
        return res_p

    def predict(self, feature):
        return self.predict_proba(feature) > 0.5

    def logit_pvalue(self, p, feature):
        """Calculate z-scores for scikit-learn LogisticRegression.
        parameters:
            model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
            x:     matrix on which the model was fit
        This function uses asymtptics for maximum likelihood estimates.
        """
        n = p.shape[0]
        m = len(self.model.coef_[0]) + 1
        print(self.model.coef_)
        coefs = np.concatenate(
            [self.model.intercept_ + self.beta_, self.model.coef_[0]]
        )
        x_full = np.matrix(np.insert(np.array(feature), 0, 1, axis=1))
        print(x_full.shape)
        ans = np.zeros((m, m))
        for i in range(n):
            ans = (
                ans
                + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i, 1] * p[i, 0]
            )
        vcov = np.linalg.inv(np.matrix(ans))
        se = np.sqrt(np.diag(vcov))
        t = coefs / se
        p_value = (1 - norm.cdf(abs(t))) * 2
        return p_value


if __name__ == "__main__":
    feature = np.random.random((100, 10))
    label = np.random.choice([0, 1], 100, replace=True)
    lr_base = LogisticRegression(C=1e30).fit(feature, label)
    tf = TransferLogistic(lr_base)
    tf.fit(feature, label)
    tf_pred = tf.predict_proba(feature)
    print(tf.p_value)
