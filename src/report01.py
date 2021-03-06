import sys

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from settings import IMG_DIR
from test_gmm import myrand_gmm


def is_existed_option(option):
    """
    当該オプションがコマンドライン引数として指定してあるかどうかを確認する

    Parameters
    ----------
    option : str
        確認したいオプション

    Returns
    -------
    bool
        True: 当該オプションが存在する場合は
        False: 当該オプションが存在しない場合
    """
    return option in sys.argv


def main():
    n = 1000  # 標本数（サンプル数）．
    x = myrand_gmm(n)  # 乱数の生成．実験に使うデータを生成する関数名に書き換える．
    m = 2  # 混合数．この値を変えて実験する．
    mu = np.array([1.0, 2.0])
    sigma = np.array([0.1, 0.3])
    sigma2 = sigma * sigma
    w = np.ones(m) / m
    xx = np.arange(0, 5, 0.01)  # 0 から 5 まで、0.01 間隔．
    y0 = norm.pdf(xx, mu[0], np.sqrt(sigma2[0]))  # e(x, mu0, sigma0**2)
    y1 = norm.pdf(xx, mu[1], np.sqrt(sigma2[1]))  # e(x, mu1, sigma1**2)
    y = w[0] * y0 + w[1] * y1  # q(x, sita)

    file_name = IMG_DIR + 'p1.png'

    plt.title("Problem 1")
    plt.plot(xx, y, color='r', label=r'q(x; $\theta$ )')
    plt.plot(
        xx,
        y0,
        color='g',
        label=r'$\varnothing$(x; $\mu_0$, $\sigma_0^2$)')
    plt.plot(
        xx,
        y1,
        color='b',
        label=r'$\varnothing$(x; $\mu_1$, $\sigma_1^2$)')
    plt.hist(x, bins='auto', density=True)
    plt.legend()
    plt.savefig(file_name)
    if is_existed_option("--show"):
        plt.show()


if __name__ == "__main__":
    main()
