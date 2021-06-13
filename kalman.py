
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:

    def __init__(self,phi,gamma,M,Q,R,x0,P0):        
        """This class implements a Kalman Filter for the Discrete time System 
        ___________________________________   
        
        x_{k+1} = phi * x_k + gamma * w_k
        y_{k} =  M * x_k + v_k
        ___________________________________

        where w_k and v_k are the process and measurement noise respectively. 
        Both the noise vectors are assumed to be i.i.d. Gaussian noise processes with 
        0 mean and covariance matrices Q and R respectively.

        Let 
        n be the dimension of the state vector
        m be the dimension of the w_k vector


        Args:
            phi (numpy array of size n * n): [description]
            gamma ([type]): [description]
            M ([type]): [description]
            Q ([type]): [description]
            R ([type]): [description]
            x0 ([type]): [description]
            P0 ([type]): [description]
        """
        self.x = np.array(x0)
        self.P = np.array(P0)

        self.Q = np.array(Q)
        self.phi = np.array(phi)
        self.gamma = np.array(gamma)
        self.M = np.array(M)

        self.R = np.array(R)

    

    def update(self,z):
        """[summary]

        Args:
            z ([type]): [description]
        """


        y = np.array(z)
        x_k_1_plus =  self.x
        P_k_1_plus =  self.P

        # Before Measurement
        x_k_minus = self.phi*x_k_1_plus
        P_k_minus = self.phi * P_k_1_plus * np.transpose(self.phi) + self.gamma * self.Q * np.transpose(self.gamma)  
		
		#Kalman Gain Calculation
        K_num = P_k_minus * np.transpose(self.M)
        K_den = self.M * P_k_minus * np.transpose(self.M) + self.R

        if(K_den.shape[0] == 1):
            K = K_num/K_den
        else:
            K = K_num * np.linalg.inv(K_den)
        
		#After Measurement
        self.x = x_k_minus + K * (y - self.M * x_k_minus)
        self.P = P_k_minus - K * self.M * P_k_minus  

        return self.x,self.P

        