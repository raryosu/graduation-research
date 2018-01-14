import numpy as np
import math as mt
from sklearn.cluster import KMeans
import warnings
import logging

warnings.filterwarnings('error')

logger = logging.getLogger(__name__)
logger.setLevel(10)
sh = logging.StreamHandler()
logger.addHandler(sh)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
sh.setFormatter(formatter)

class XMeans:
    def __init__(self, X, kmax=20, ic='bic'):
        self.X = X
        self.num = np.size(self.X, axis=0)
        self.dim = np.size(self.X, axis=1)
        self.KMax = kmax
        self.ic = ic.lower()
        self.labels_ = []
        self.k_ = 0
        self.cluster_centers_ = []

    def log_likelihood(self, r, rn, var, m, k):
        l1 = - rn / 2.0 * mt.log(2 * mt.pi)
        l2 = - rn * m / 2.0 * mt.log(var)
        l3 = - (rn - k) / 2.0
        l4 = rn * mt.log(rn)
        l5 = - rn * mt.log(r)

        return l1 + l2 + l3 + l4 + l5

    def fit(self):
        k = 1
        X = self.X
        M = self.dim
        num = self.num


        while (1):
            ok = k

            kmeans = KMeans(n_clusters=k).fit(X)
            labels = kmeans.labels_
            mean = kmeans.cluster_centers_

            p = M + 1

            obic = np.zeros(k)

            for i in range(k):
                rn = np.size(np.where(labels == i))
                try:
                    var = (np.sum((X[labels == i] - mean[i]) ** 2) + 1.0) / float(rn - 1)
                except RuntimeWarning:
                    break
                try:
                    if self.ic == 'aic':
                        obic[i] = self.log_likelihood(rn, rn, var, M, 1) - p
                    elif self.ic == 'bic':
                        obic[i] = self.log_likelihood(rn, rn, var, M, 1) - p/2.0 * mt.log(rn)
                    elif self.ic == 'caic':
                        obic[i] = self.log_likelihood(rn, rn, var, M, 1) - (p * rn) / (rn - p - 1)
                    elif self.ic == 'll':
                        obic[i] = self.log_likelihood(rn, rn, var, M, 1)
                except:
                    break

            sk = 2
            nbic = np.zeros(k)
            addk = 0

            for i in range(k):
                ci = X[labels == i]
                r = np.size(np.where(labels == i))

                kmeans = KMeans(n_clusters=sk).fit(ci)
                ci_labels = kmeans.labels_
                smean = kmeans.cluster_centers_

                for l in range(sk):
                    rn = np.size(np.where(ci_labels == l))
                    try:
                        var = np.sum((ci[ci_labels == l] - smean[l]) ** 2) / (rn - sk)
                    except:
                        break
                    try:
                        tmp = self.log_likelihood(r, rn, var, M, sk)
                    except:
                        break
                    nbic[i] += tmp

                p = sk * (M + 1)
                try:
                    if self.ic == 'aic':
                        nbic[i] -= p
                    elif self.ic == 'bic':
                        nbic[i] -= p/2.0 * mt.log(r)
                    elif self.ic == 'caic':
                        nbic[i] -= (p * r) / (r - p - 1)
                except:
                    break

                logger.debug("obic: {}, nbic: {}".format(obic[i], nbic[i]))

                if obic[i] < nbic[i]:
                    addk += 1

            k += addk

            if (ok == k) or (k >= self.KMax):
                break

        kmeans = KMeans(n_clusters=k).fit(X)
        logger.debug("labels: {}".format(kmeans.labels_))
        self.labels_ = kmeans.labels_
        self.k_ = k
        self.cluster_centers_ = kmeans.cluster_centers_
