import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
def myrand_gmm(n, fill=0.0): # n は生成するデータの個数
    x=np.zeros(n)
    g=np.random.randn(n)
    u=np.random.rand(n)
    mu = np.array([1.0, 2.0, 3.0]) # 各ガウス分布の平均値．値を変えていろいろ試す． これは混合数 m=3 の場合
    sigma = np.array([0.1, 0.3, 0.5]) # 各分布の標準偏差．値を変えていろいろ試す．
    flag=(0<=u) & (u<1/3) # この例は，各分布から 1/3 の確率でデータが出現する場合．
    x = (mu[0] + sigma[0]*g)*flag
    flag=(1/3<=u) & (u<2/3)
    x += (mu[1] + sigma[1]*g)*flag
    flag=(2/3<=u) & (u<=1)
    x += (mu[2] + sigma[2]*g)*flag
    return x

if __name__ == "__main__":
    n = 1000 # 標本数（サンプル数）．
    x = myrand_gmm(n) # 乱数の生成．実験に使うデータを生成する関数名に書き換える．
    m = 3 # 混合数．この値を変えて実験する．
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.1, 0.3, 0.5])
    sigma2 = sigma*sigma
    w = np.ones(m)/m
    xx = np.arange(0,5,0.01) #0 から 5 まで、0.01 間隔．
    y0 = norm.pdf(xx, mu[0], np.sqrt(sigma2[0] ) )
    y1 = norm.pdf(xx, mu[1], np.sqrt(sigma2[1] ) )
    y2 = norm.pdf(xx, mu[2], np.sqrt(sigma2[2] ) )
    y = w[0]*y0 + w[1]*y1 + w[2]* y2
    plt.plot(xx,y,color='r')
    plt.hist(x, bins='auto', density=True)
    plt.show()