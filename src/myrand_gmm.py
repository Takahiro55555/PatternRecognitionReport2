import numpy as np


def myrand_gmm_m2(n, fill=0.0):  # n は生成するデータの個数
    x = np.zeros(n)
    g = np.random.randn(n)
    u = np.random.rand(n)
    mu = np.array([1.0, 2.0])  # 各ガウス分布の平均値．値を変えていろいろ試す． これは混合数 m=2 の場合
    sigma = np.array([0.1, 0.3])  # 各分布の標準偏差．値を変えていろいろ試す．
    flag = (0 <= u) & (u < 1 / 2)  # この例は，各分布から 1/2 の確率でデータが出現する場合．
    x = (mu[0] + sigma[0] * g) * flag
    flag = (1 / 2 <= u) & (u < 1)
    x += (mu[1] + sigma[1] * g) * flag
    return x


def myrand_gmm_m3(n, fill=0.0):  # n は生成するデータの個数
    x = np.zeros(n)
    g = np.random.randn(n)
    u = np.random.rand(n)
    mu = np.array([1.0, 2.0, 3.0])  # 各ガウス分布の平均値．値を変えていろいろ試す． これは混合数 m=3 の場合
    sigma = np.array([0.1, 0.3, 0.5])  # 各分布の標準偏差．値を変えていろいろ試す．
    flag = (0 <= u) & (u < 1 / 3)  # この例は，各分布から 1/3 の確率でデータが出現する場合．
    x = (mu[0] + sigma[0] * g) * flag
    flag = (1 / 3 <= u) & (u < 2 / 3)
    x += (mu[1] + sigma[1] * g) * flag
    flag = (2 / 3 <= u) & (u <= 1)
    x += (mu[2] + sigma[2] * g) * flag
    return x


def myrand_gmm_m5(n, fill=0.0):  # n は生成するデータの個数
    x = np.zeros(n)
    g = np.random.randn(n)
    u = np.random.rand(n)
    # 各ガウス分布の平均値．値を変えていろいろ試す． これは混合数 m=5 の場合
    mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sigma = np.array([0.1, 0.3, 0.5, 0.8, 1.0])  # 各分布の標準偏差．値を変えていろいろ試す．
    flag = (0 <= u) & (u < 1 / 5)  # この例は，各分布から 1/3 の確率でデータが出現する場合．
    x = (mu[0] + sigma[0] * g) * flag
    flag = (1 / 5 <= u) & (u < 2 / 5)
    x += (mu[1] + sigma[1] * g) * flag
    flag = (2 / 5 <= u) & (u <= 3 / 5)
    x += (mu[2] + sigma[2] * g) * flag
    flag = (3 / 5 <= u) & (u <= 4 / 5)
    x += (mu[3] + sigma[3] * g) * flag
    flag = (4 / 5 <= u) & (u <= 1)
    x += (mu[4] + sigma[4] * g) * flag
    return x


if __name__ == "__main__":
    pass
