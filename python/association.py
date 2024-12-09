'''
Expectation-Maximization (EM) Algorithm

This is a test program for the Expectation-Maximization algorithm.
We first generate two sets of points from two normal multivariate
distributions and label them as cluster 1 and 2 (0 and 1). This will 
serve as our true value dataset.
We then reshuffle the labels to make the dataset ready for the algorithm.
Given the number of clusters, the algorithm is going to look at the data 
and estimate paramters for two normal distributions that the data is drawn 
from. These parameters include the means and covariances of those 
normal distributions.   
In the next step, the algorithm starts from initial guess for which we 
use a set of bad (far) guesses intentionally.
Now we run the EM algorithm and the algorithm starts from the initial guess
and iteratively converges to the solution. 
===================================================
Code: Reza Ahmadzadeh - 2023
===================================================
'''
import numpy as np
import matplotlib.pyplot as plt
from dynamic_EM import *


def generate_data(path):
    '''
    This function generates random data
    The points are drawn from two multivariate Normal Distributions 
    with parameters (mu, sigma)
    Input: 
        n: number of points in each cluster
    Output: 
        D: the generated dataset, size: 2n x 3 [x y label]
    '''

    data = np.loadtxt(path, delimiter=',')
    frame = 2 # detection and frame numbers begin at 1
    frame_data = data[data[:, 0] == frame]
    Param = {"sigma": [], "mu": []}
    line_idx = 0
    idx = 1
    mean0 = []
    sigma0 = []
    for k in frame_data[:,1]:
        if k == 1:#estimation
            mean1 = frame_data[line_idx, 2:5]
            sigma1 = frame_data[line_idx, 5:14]
            Param["sigma"].append(sigma1.reshape((3, 3)))
            Param["mu"].append(mean1)
            Param["lambd"] = np.full(np.sum(frame_data[:,1] ==1), 1 / np.sum(frame_data[:,1] ==1))
            idx +=1
        else:#measurement
            mean0.append(frame_data[line_idx, 2:5])
            sigma0.append(frame_data[line_idx, 5:14].reshape((3,3)))
        line_idx +=1

    mean0 = np.array(mean0)
    sigma0 = np.array(sigma0)
    D = np.hstack((mean0, np.zeros((len(mean0), 1))))

    return D, Param



def test():

    path = '/home/fbh/2024/FGMOT/try_gmm/em_data.txt'
    Data, Param = generate_data(path)

    # run EM to find the parameters 
    Data_f, Param_f = EM(Data, Param)

    print(type(Param_f), Param_f)

    # plot results
    plt.figure(figsize=(10,4))
    ax1 = plt.subplot(1,3,1)
    ax1.scatter(Data_r[:, 0], Data_r[:, 1], 10, 'b')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Raw Data')

    Data1 = Data[0:num_points, 0:2]
    Data2 = Data[num_points:2*num_points, 0:2]
    ax2 = plt.subplot(1,3,2)
    ax2.scatter(Data1[:, 0], Data1[:, 1], 10, 'g')
    ax2.scatter(Data2[:, 0], Data2[:, 1], 10, 'r')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('True Value')

    Data_f1 = Data_f[Data_f[:,2]==0, :]
    Data_f2 = Data_f[Data_f[:,2]==1, :]
    ax3 = plt.subplot(1,3,3)
    ax3.scatter(Data_f1[:, 0], Data_f1[:, 1], 10, 'g')
    ax3.scatter(Data_f2[:, 0], Data_f2[:, 1], 10, 'r')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Data Clustered by EM')
    plt.show()


if __name__=='__main__':
    test()