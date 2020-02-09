import sys

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from test_gmm import myrand_gmm

# def myrand_gmm(n, fill=0.0): # n は生成するデータの個数
# 関数を定義しておく．前ページ参照．
n = 5000  # 標本数（サンプル数）．
x = myrand_gmm(n)  # 乱数の生成．実験に使うデータを生成する関数名に書き換える．
m = 3  # 混合数．この値を変えて実験する．
# 初期値の設定．w, mu, sigma2 は m 次元縦ベクトル．
L = -np.inf
w = np.ones(m) / m  # m 個の正規分布の重みの初期値
w = w.reshape(m, 1)
mu = np.linspace(min(x), max(x), m)  # 平均値の初期値
mu = mu.reshape(m, 1)  # m を明示的に縦ベクトルにする
sigma2 = np.ones(m) / 10  # 分散の初期値
sigma2 = sigma2.reshape(m, 1)
while True:
    tmp1 = np.square(np.tile(x, (m, 1)) - np.tile(mu, (1, n)))
    tmp2 = 2 * np.tile(sigma2, (1, n))
    tmp3 = np.tile(w, (1, n)) * np.exp(-tmp1 / tmp2) / np.sqrt(np.pi * tmp2)
    eta = tmp3 / np.tile(np.sum(tmp3, axis=0), (m, 1))  # ここまでがηの計算
    tmp4 = np.sum(eta, axis=1)
    w = tmp4 / n
    w = w.reshape(m, 1)
    mu = (eta.dot(x)) / tmp4
    mu = mu.reshape(m, 1)
    sigma2 = np.sum(tmp1 * eta, axis=1) / tmp4
    sigma2 = sigma2.reshape(m, 1)
    Lnew = np.sum(np.log(np.sum(tmp3, axis=0)))  # 更新後の対数尤度
    if Lnew - L < 0.0001:
        break
    L = Lnew
xx = np.arange(0, 5, 0.01)  # 0 から 5 まで、0.01 間隔．
y0 = norm.pdf(xx, mu[0], np.sqrt(sigma2[0]))
y1 = norm.pdf(xx, mu[1], np.sqrt(sigma2[1]))
y2 = norm.pdf(xx, mu[2], np.sqrt(sigma2[2]))
y = w[0] * y0 + w[1] * y1 + w[2] * y2
plt.plot(xx, y, color='r')
plt.hist(x, bins='auto', density=True)
if is_existed_option("--show"):
    plt.show()
