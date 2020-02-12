import sys

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from myrand_gmm import myrand_gmm_m5
from settings import IMG_DIR, SEED


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
    # Fixing random state for reproducibility
    np.random.seed(SEED)

    n = 1000  # 標本数（サンプル数）．
    x = myrand_gmm_m5(n)  # 乱数の生成．実験に使うデータを生成する関数名に書き換える．
    # x = myrand(n)
    m = 3  # 混合数．この値を変えて実験する．

    # 初期値の設定．w, mu, sigma2 は m 次元縦ベクトル．
    L = -np.inf
    w = np.ones(m) / m  # m個の正規分布の重みの初期値
    w = w.reshape(m, 1)
    mu = np.linspace(min(x), max(x), m)   # 平均値の初期値
    mu = mu.reshape(m, 1)  # m を明示的に縦ベクトルにする
    sigma2 = np.ones(m) / 10  # 分散の初期値
    sigma2 = sigma2.reshape(m, 1)

    # mu_t = np.empty((1,m), float)

    Lt = L
    wt = w
    mut = mu
    sigma2t = sigma2
    t = 0
    tt = np.array([0])

    while True:
        tmp1 = (np.tile(x, (m, 1)) - np.tile(mu, (1, n))) * \
            (np.tile(x, (m, 1)) - np.tile(mu, (1, n)))
        tmp2 = 2 * np.tile(sigma2, (1, n))
        tmp3 = np.tile(w, (1, n)) * np.exp(-tmp1 / tmp2) / \
            np.sqrt(np.pi * tmp2)
        eta = tmp3 / np.tile(np.sum(tmp3, axis=0), (m, 1))  # ここまでがηの計算
        tmp4 = np.sum(eta, axis=1)
        w = tmp4 / n
        w = w.reshape(m, 1)
        mu = (eta.dot(x)) / tmp4
        mu = mu.reshape(m, 1)

        sigma2 = np.sum(tmp1 * eta, axis=1) / tmp4
        sigma2 = sigma2.reshape(m, 1)
        Lnew = np.sum(np.log(np.sum(tmp3, axis=0)))  # 更新後の対数尤度

        wt = np.append(wt, w, axis=1)
        mut = np.append(mut, mu, axis=1)
        sigma2t = np.append(sigma2t, sigma2, axis=1)

        # mu, sigma2, w

        if Lnew - L < 0.0001:
            break
        L = Lnew
        Lt = np.append(Lt, L)
        t = t + 1
        tt = np.append(tt, t)

    xx = np.arange(0, 8, 0.01)  # 0から5まで、0.01間隔．
    y0 = norm.pdf(xx, mu[0], np.sqrt(sigma2[0]))
    y1 = norm.pdf(xx, mu[1], np.sqrt(sigma2[1]))
    y2 = norm.pdf(xx, mu[2], np.sqrt(sigma2[2]))
    y = w[0] * y0 + w[1] * y1 + w[2] * y2

    print("EM Time: %d(file: %s)" % (len(wt[0]), __file__))

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(xx, y, color='r', label=r'q(x; $\theta$ )')
    axs[0, 0].hist(x, bins='auto', density=True)
    axs[0, 0].legend()

    axs[0, 1].plot(wt[0], label=r'$w_0$')
    axs[0, 1].plot(wt[1], label=r'$w_1$')
    axs[0, 1].plot(wt[2], label=r'$w_2$')
    # axs[0].set_xlim(0, 2)
    axs[0, 1].set_xlabel('time')
    axs[0, 1].set_ylabel(r'$w_0$, $w_1$, and $w_2$')
    axs[0, 1].grid(True)
    axs[0, 1].legend(bbox_to_anchor=(1.05, 1),
                     loc='upper left', borderaxespad=0, fontsize=7)
    axs[0,
        1].annotate(int(wt[0][-1] * 1000) / 1000,
                    xy=(len(wt[0]),
                        wt[0][-1]),
                    xytext=(+15,
                            -10),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3,rad=-0.2"))
    axs[0,
        1].annotate(int(wt[1][-1] * 1000) / 1000,
                    xy=(len(wt[1]),
                        wt[1][-1]),
                    xytext=(+15,
                            -10),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3,rad=-0.2"))
    axs[0,
        1].annotate(int(wt[2][-1] * 1000) / 1000,
                    xy=(len(wt[2]),
                        wt[2][-1]),
                    xytext=(+15,
                            -20),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3,rad=-0.2"))

    axs[1, 0].plot(mut[0], label=r'$\mu_0$')
    axs[1, 0].plot(mut[1], label=r'$\mu_1$')
    axs[1, 0].plot(mut[2], label=r'$\mu_2$')
    # axs[0].set_xlim(0, 2)
    axs[1, 0].set_xlabel('time')
    axs[1, 0].set_ylabel(r'$\mu_0$, $\mu_1$, and $\mu_2$')
    axs[1, 0].grid(True)
    axs[1, 0].legend(bbox_to_anchor=(1.05, 1),
                     loc='upper left', borderaxespad=0, fontsize=7)
    axs[1,
        0].annotate(int(mut[0][-1] * 1000) / 1000,
                    xy=(len(mut[0]),
                        mut[0][-1]),
                    xytext=(+15,
                            -10),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3,rad=-0.2"))
    axs[1,
        0].annotate(int(mut[1][-1] * 1000) / 1000,
                    xy=(len(mut[1]),
                        mut[1][-1]),
                    xytext=(+15,
                            -10),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3,rad=-0.2"))
    axs[1,
        0].annotate(int(mut[2][-1] * 1000) / 1000,
                    xy=(len(mut[2]),
                        mut[2][-1]),
                    xytext=(+15,
                            -10),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3,rad=-0.2"))

    axs[1, 1].plot(sigma2t[0], label=r'$\sigma_0$')
    axs[1, 1].plot(sigma2t[1], label=r'$\sigma_1$')
    axs[1, 1].plot(sigma2t[2], label=r'$\sigma_2$')
    # axs[0].set_xlim(0, 2)
    axs[1, 1].set_xlabel('time')
    axs[1, 1].set_ylabel(r'$\sigma_0$, $\sigma_1$, and $\sigma_2$')
    axs[1, 1].grid(True)
    axs[1, 1].legend(bbox_to_anchor=(1.05, 1),
                     loc='upper left', borderaxespad=0, fontsize=7)
    axs[1,
        1].annotate(int(sigma2t[0][-1] * 1000) / 1000,
                    xy=(len(sigma2t[0]),
                        sigma2t[0][-1]),
                    xytext=(+15,
                            -10),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3,rad=-0.2"))
    axs[1,
        1].annotate(int(sigma2t[1][-1] * 1000) / 1000,
                    xy=(len(sigma2t[1]),
                        sigma2t[1][-1]),
                    xytext=(+15,
                            -10),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3,rad=-0.2"))
    axs[1,
        1].annotate(int(sigma2t[2][-1] * 1000) / 1000,
                    xy=(len(sigma2t[2]),
                        sigma2t[2][-1]),
                    xytext=(+15,
                            -50),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3,rad=-0.2"))

    fig.tight_layout()
    f_name = "%sp3.png" % IMG_DIR
    plt.savefig(f_name)
    if is_existed_option("--show"):
        plt.show()


if __name__ == "__main__":
    main()
