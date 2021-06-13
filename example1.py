import numpy as np
import matplotlib.pyplot as plt
from kalman import KalmanFilter


if __name__ == '__main__':
    """In this example we have a constant variable of interest that we have 
    noisy measurements of. 

    Thus the process equations are -

    x_{k+1} = x_k
    y_{k} = x_k + v_k

    Thus the Kalman Filter can be used to compute the unbiased estimate of the 
    variable of interest. 
    """




    P0 = 100 #Initial Estimate of 
    R = 100
    x0 = 0
    kf = KalmanFilter([1],[0],[1],[0],[R],[x0],[P0])
    xHist = [x0]
    pHist = [P0]
    yHist = []
    noise = 50
    
    temp = 25

    N = 100
    for t in range(N):
        z = temp + noise*(np.random.random() - 0.5)
        x,P = kf.update([z])
        xHist.append(x)
        pHist.append(P)
        yHist.append(z)

    xHist = np.array(xHist)
    pHist = np.array(pHist)

    plt.plot(xHist,label = 'Predicted Mean')
    plt.plot(xHist + pHist,linestyle='dashed', label = "68% Confidence Upper Interval")
    plt.plot(xHist - pHist,linestyle='dashed',label = "68% Confidence Lower Interval")
    plt.scatter(list(range(1,N+1)),yHist,marker = "+",label = "Noisy Observations")
    plt.legend()
    plt.grid()
    plt.show()
        

    