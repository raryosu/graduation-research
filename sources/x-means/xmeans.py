import numpy as np
import math as mt
from sklearn.cluster import KMeans

class XMeans:
    def __init__(self, X, kmax=20):
        self.X = X
        self.num = np.size(self.X, axis=0)
        self.dim = np.size(self.X, axis=1)
        self.KMax = kmax

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

        while(1):
            ok = k

            kmeans = KMeans(n_clusters=k).fit(X)
            labels = kmeans.labels_
            mean = kmeans.cluster_centers_

            p = M + 1

            obic = np.zeros(k)

            for i in range(k):
                rn = np.size(np.where(labels == i))
                var = (np.sum((X[labels == i] - mean[i]) ** 2) + 1.0) / float(rn - 1)
                # AIC
                # obic[i] = self.log_likelihood(rn, rn, var, M, 1) - p
                # BIC
                # obic[i] = self.log_likelihood(rn, rn, var, M, 1) - p/2.0 * mt.log(rn)
                # c-AIC
                obic[i] = self.log_likelihood(rn, rn, var, M, 1) - (p * rn) / (rn - p - 1)

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
                    var = np.sum((ci[ci_labels == l] - smean[l]) ** 2) / (rn - sk)
                    nbic[i] += self.log_likelihood(r, rn, var, M, sk)

                p = sk * (M + 1)
                # AIC
                # nbic[i] -= p
                # BIC
                # nbic[i] -= p/2.0 * mt.log(r)
                # cAIC
                nbic[i] -= (p * r) / (r - p - 1)

                if obic[i] < nbic[i]:
                    addk += 1

            k += addk

            if (ok == k) or (k >= self.KMax):
                break

        kmeans = KMeans(n_clusters=k).fit(X)
        self.labels = kmeans.labels_
        self.k = k
        self.m = kmeans.cluster_centers_

