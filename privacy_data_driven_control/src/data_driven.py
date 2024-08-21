import numpy as np
import pandas as pd
import cvxpy as cp

from scipy import linalg as LA
from scipy.signal import cont2discrete

  

from typing import Tuple




class Plant():
    """
    This class models the behavior and data processing of a dynamic plant system as described in the paper.
    It provides methods for key generation, trajectory collection, data transformation, and controller post-processing.

    Attributes:
        As (np.ndarray): State transition matrix of the plant.
        Bs (np.ndarray): Input matrix of the plant.
        num_states (int): Number of states in the system.
        num_inputs (int): Number of inputs to the system.

    Methods:
        key_generation(n, m, key_range):
            Generates random matrices F1 and G1 for use as keys.

        random_matrix(row, col, key_range):
            Generates a random matrix of specified size and range.

        collecting_trajectories(T, input_range, initial_state_range, disturbance_range):
            Collects state, input, and disturbance trajectories over a given time horizon.

        transforming_data(X1, X0, U0, F1, G1):
            Transforms the collected data using given transformation matrices F1 and G1.

        post_processing_controller(F1, G1, eps_bar, key_range, Bs):
            Adjusts the transformation matrices F2 and G2 based on controller constraints.

        get_traject_bias(K, x0, T, T_injection, beta, ainf):
            Computes the trajectory of the system under a control law with bias injection.
    """


    def __init__(self, As: np.ndarray, Bs: np.ndarray):
        self.As = As
        self.Bs = Bs
        self.num_states = np.size(self.As, 1)
        self.num_inputs = np.size(self.Bs, 1)
    
    def key_generation(self, n: int, m: int, key_range: float) -> Tuple[np.ndarray, np.ndarray]:
        F1 = self.random_matrix(m, n, key_range)
        G1 = self.random_matrix(m, m, key_range)
        return F1, G1

    def random_matrix(self, row: int, col: int, key_range: float) -> np.ndarray:
        return (np.random.rand(row, col) * (2 * key_range)) - key_range


    def collecting_trajectories(self, T: int, input_range: float,
                                initial_state_range: float, disturbance_range:float = 0) -> Tuple[
                                    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


        U0 = self.random_matrix(self.num_inputs, T, input_range)
        X0 = self.random_matrix(self.num_states, T, initial_state_range)
        D0 = self.random_matrix(self.num_states, T, disturbance_range)


        X1 = self.As @ X0 + self.Bs @ U0 + D0

        return X1, X0, U0, D0



    def transforming_data(self, X1: np.ndarray, X0: np.ndarray,
                           U0: np.ndarray, F1: np.ndarray, G1: np.ndarray) -> Tuple[
                               np.ndarray, np.ndarray, np.ndarray]:
        X1_tilde = X1
        X0_tilde = X0
        Im = np.eye(np.size(G1, 0))
        I_G1_inv = LA.inv(Im + G1)

        V0 = (-(I_G1_inv) @ F1 @ (X0)) +((I_G1_inv) @ (U0))



        return X1_tilde, X0_tilde, V0



    def post_processing_controller(self, F1: np.ndarray, G1: np.ndarray,
                                    eps_bar: float, key_range: float, Bs: np.ndarray) -> Tuple[
                                        np.ndarray, np.ndarray]:

        con = eps_bar/(LA.norm(Bs,ord=2))

        deltaF2 = self.random_matrix(self.num_inputs, self.num_states, key_range)
        deltaG2 = self.random_matrix(self.num_inputs, self.num_inputs, key_range)

        deltaF2 = deltaF2 /LA.norm(deltaF2, ord=2)
        deltaG2 = deltaG2 /LA.norm(deltaG2, ord=2)

        G2 = G1
        F2 = F1 - 0.999 * con * deltaF2

        

        return F2, G2

    def get_traject_bias(self,  K: np.ndarray, x0: np.ndarray, T: int,
                          T_injection: int, beta: float, ainf: float) -> np.ndarray:

        X = np.zeros((self.num_states, T + 1))
        X[:, [0]] = x0
        a = np.zeros((self.num_inputs, 1))
        
        for i in range(T):
            u = K @ X[:,[i]]
            if i>= T_injection:
                a = (beta * a) + (1-beta) * ainf
                X[:,[i+1]] = self.As @ X[:,[i]] + self.Bs @ (u + a)
            else:
                X[:,[i+1]] = self.As @ X[:,[i]] + self.Bs @ (u)



        return X



class Cloud():
    """
    This class models the cloud-based computations as described in the paper.
    It provides methods for computing matrices for the Quadratic Matrix Inequality (QMI) set and 
    solving a feasibility problem using ellipsoid parameters. The feasibility problem is solved using 
    the CVXPY library, with MOSEK as the default solver for better precision.

    Methods:
        ellipsoid_parameters(X1, X0, U0, disturbance_bound):
            Computes the matrices bA, bB, bC, and bQ for the QMI set using input data and a disturbance bound.

        get_controller_cvxpy(bA, bB, bC, epsilon):
            Solves a feasibility problem using the computed ellipsoid parameters or those provided by the user. 
            This method uses CVXPY with MOSEK as the backend solver. Other solvers can be specified, but MOSEK 
            has been empirically found to yield better precision. Returns the solution status, and the 
            optimal matrices P and Y if a solution is found.
    """

    
    def ellipsoid_parameters(self, X1: np.ndarray, X0: np.ndarray, 
                             U0: np.ndarray, disturbance_bound: float) -> Tuple[
                                 np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        n = np.size(X1, 0)
        T = np.size(X1, 1)

        delta_delta_tran = n * (disturbance_bound**2) * T * np.eye(n)

        # Ellipsoid Parameters
        X0_U0 = np.vstack((X0, U0))
        bA = (X0_U0) @ (X0_U0).T
        bB = - (X0_U0) @ X1.T
        bC = X1 @ X1.T - delta_delta_tran


        # # Ellipsoid parameter second form

        bQ =  bB.T @ LA.inv(bA) @ bB - bC

        return bA, bB, bC, bQ
        


    def get_controller_cvxpy(self, bA: np.ndarray, bB: np.ndarray,
                              bC: np.ndarray, epsilon: float) -> Tuple[
                                  bool, np.ndarray, np.ndarray]:

        n =  np.size(bC, 1)
        m = np.size(bA, 1) - n

    
        P = cp.Variable((n,n), symmetric=True)
        Y = cp.Variable((m,n))

        bQ = bB.T @ LA.inv(bA) @ bB - bC
        constant1 = 2 * np.sqrt(n * LA.norm(bA, ord=2) * LA.norm(bQ, ord=2))
        constant2 = n * LA.norm(bA, ord=2)


        # The operator >> denotes matrix inequality.
        constraints = [P >> cp.multiply(1e-6, np.eye(n))]

        constraints += [cp.bmat([[P + bC - cp.multiply(epsilon, constant1 * np.eye(n))
                                - cp.multiply(epsilon**2, constant2 * np.eye(n)) ,
                                    np.zeros((n, n)), -bB.T ],
                                [np.zeros((n, n)), P, (-cp.vstack((P,Y))).T],
                                [-bB, -cp.vstack((P,Y)), bA]
                                ])>> cp.multiply(1e-6, np.eye(3*n + m))]


        prob = cp.Problem(cp.Maximize(1),
                        constraints)
        prob.solve(solver='MOSEK', eps=1e-6)

        return prob, P.value, Y.value





class BiasSignal():
    """
    This class handles the injection of bias signals into actuator data, based on the information shared by the Cloud.
    It determines the level of information the adversary (person injecting the bias) possesses, and based on that 
    information, calculates the magnitude of the injected bias signal.

    Attributes:
        As (np.ndarray): State transition matrix of the system.
        Bs (np.ndarray): Input matrix of the system.
        F1, F2, G1, G2 (np.ndarray): Keys for privacy method.
        K_bar (np.ndarray): Baseline feedback gain matrix.
        threshold (float): Threshold value used in bias magnitude calculation.
        K (np.ndarray): Calculated control gain based on F1, G1, and K_bar.

    Methods:
        a_inf(ad_info: str) -> np.ndarray:
            Core method that differentiates the bias injection process based on the level of information the adversary has.
            Depending on the information type (`ad_info`), it calls other methods to compute the magnitude of the injected signal.

        fun_ainf(Acl: np.ndarray, Bcl: np.ndarray) -> np.ndarray:
            Helper method that computes the magnitude of the injected bias signal based on closed-loop system matrices.
    """


    def __init__(self, As: np.ndarray, Bs: np.ndarray, F1: np.ndarray,
                  G1: np.ndarray, F2: np.ndarray, G2: np.ndarray, 
                  K_bar: np.ndarray, threshold: float):
        
        self.As = As
        self.Bs = Bs
        self.F1 = F1
        self.G1 = G1
        self.F2 = F2
        self.G2 = G2
        self.K_bar = K_bar
        self.threshold = threshold


        self.n = np.size(self.As, 0)
        self.m = np.size(self.Bs, 1)

     
        self.K = self.F1 + (np.eye(self.m) + self.G1) @ self.K_bar
     


    def a_inf(self, ad_info: str) -> np.ndarray:

        if ad_info == 'no_bias':
            ainf = 0
        
        elif ad_info =='perfect_info':
            K_star = self.F2 + (np.eye(self.m) + self.G2) @ self.K_bar

            Acl = self.As + self.Bs @ K_star
            Bcl = self.Bs

            ainf = self.fun_ainf(Acl, Bcl)

        elif ad_info =='use_of_estimation':

            Acl = self.As + self.Bs @ self.K
            Bcl =  self.Bs @ (np.eye(self.m) + self.G1)
            
            ainf = self.fun_ainf(Acl, Bcl)
            
        elif ad_info =='know_Bcl':
            Acl = self.As + self.Bs @ self.K

            Bcl = self.Bs
            
            ainf = self.fun_ainf(Acl, Bcl)

        elif ad_info =='know_norm_of_Bcl':

            Acl = self.As + self.Bs @ self.K


            Bg = self.Bs @ (np.eye(self.m) + self.G1)

            Bg_norm = LA.norm(Bg,ord=2)
            B_norm = LA.norm(self.Bs,ord=2)
   

            Bcl = (B_norm/Bg_norm) * Bg 



            ainf = self.fun_ainf(Acl, Bcl)


        return ainf

    def fun_ainf(self, Acl: np.ndarray, Bcl: np.ndarray) -> np.ndarray:

        Gxa = LA.inv(np.eye(self.n) - Acl) @ Bcl
        GXA = Gxa.T @ Gxa
        U, s, Vh = LA.svd(LA.sqrtm(GXA))
        ainf =  self.threshold * ((1/s[0]) * U[:, 0].reshape(-1, 1))
        return ainf


