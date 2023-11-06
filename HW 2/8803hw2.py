import scipy
import pandas as pd
import numpy as np
from numpy.linalg import norm
import cmath
import pprint

# Givens Rotation
def givens(a:int, b:int, hyperbolic:bool=False):
    if hyperbolic:
        r = np.sqrt(a**2 - b**2)
    else:
        r = np.sqrt(a**2 + b**2)
    c, s = a/r, b/r 
    return c, s


def permu_matrix(size:int, inverse:bool=False):
    if inverse:
        C = np.eye(size-2)
        C = np.vstack((np.zeros(size-2), C))
        C = np.hstack((C, np.asmatrix(np.zeros(size-1)).T))
        C[0, -1] = 1
        return C
    B = np.eye(size)
    B[0, 0], B[-1, -1] = 0, 0
    B[0, -1], B[-1, 0] = 1, 1
    return B


# QR factorization based on givens
def qr_givens(A:np.array, reduced:bool=False):
    m, n = A.shape
    Q = np.eye(m)
    R = np.copy(A)

    for j in range(n):
        for i in range(m-1, j, -1):
            c, s = givens(R[i-1,j], R[i,j])
            R[i-1, :], R[i, :] = c*R[i-1, :] + s*R[i, :], (-s)*R[i-1, :] + c*R[i, :]
            Q[:, i-1], Q[:, i] = c*Q[:, i-1] + s*Q[:, i], (-s)*Q[:, i-1] + c*Q[:, i]
    if reduced:
        return Q[:, :n], R[:n, :]
    else:
        return Q, R
    


####### Problems #######
####### 1 #######

def full_qr_del(A): # Full QR Downdating
    ##### Already know #####
    m, n = A.shape
    Q, R = qr_givens(A)
    ##### Already know #####
    
    B = permu_matrix(m)
    Q = B @ Q

    q1 = np.asmatrix(Q[0, :]).T
    J, _ = qr_givens(q1)
    Q_ = Q @ J
    q1_R = np.hstack((q1, R))
    R1 = (J.T @ q1_R)[1:, 1:]
    Q1 = Q_[1:, 1:]

    C = permu_matrix(m, inverse=True)
    Q1 = C @ Q1


    return Q1, R1


####### 2 #######
def reduced_qr_del(A): # Reduced QR Downdating
    ##### Already know #####
    m, n = A.shape
    Q1, R1 = qr_givens(A, reduced=True)
    ##### Already know #####

    B = permu_matrix(m)
    Q1 = B @ Q1
    
    e1 = np.asmatrix(np.eye(m)[:, 0]).T
    q1 = np.asmatrix(Q1[0, :]).T
    v = e1 - Q1 @ q1
    if norm(v) >= np.sqrt(1/2):
        v = v/norm(v)
    else:
        s = Q1.T @ v
        v_ = v - Q1 @ s
        if norm(v_) >= norm(v)/np.sqrt(2):
            v = v_/norm(v_)
        else:
            h = (Q1[1:, 1:]).nullspace()[0]
            h = h/norm(h)
            v = np.hstack((np.zeros(1), h))
    Q1_e1 = np.hstack((Q1, v))

    # givens rotation to eliminate first row
    #J = np.eye(n+1)
    R_0 = np.vstack((R1, np.zeros(n)))
    for i in range(n-1, -1, -1):
        c, s = givens(Q1_e1[0, n], Q1_e1[0, i])
        Q1_e1[:, n], Q1_e1[:, i] = c*Q1_e1[:, n] + s*Q1_e1[:, i], (-s)*Q1_e1[:, n] + c*Q1_e1[:, i]
        R_0[n, :], R_0[i, :] = c*R_0[n, :] + s*R_0[i, :], (-s)*R_0[n, :] + c*R_0[i, :]
    
    Q = Q1_e1[1: ,:-1]
    R = R_0[:-1, :]

    C = permu_matrix(m, inverse=True)
    Q = C @ Q

    return Q, R


####### 3 #######
def r_qr_del(A, method2=False): # Cholesky Downdating
    ##### Already know #####
    m, n = A.shape
    _, R = qr_givens(A)
    ##### Already know #####

    # results are real numbers
    if method2:
        import cmath
        a = np.asmatrix(1j*A[-1, :])
        R_ = np.vstack((R, a))
        for i in range(n):
            c, s = givens(R_[i,i], R_[m-1,i])
            R_[i, :], R_[m-1, :] = c*R_[i, :] + s*R_[m-1, :], (-s)*R_[i, :] + c*R_[m-1, :]
        return R_

    # results are complex numbers
    a = np.asmatrix(A[-1, :])
    R_ = np.vstack((R, a))
    for i in range(n):
        ch, sh = givens(R_[i,i], R_[m,i], hyperbolic=True)
        R_[i, :], R_[m, :] = ch*R_[i, :] + (-sh)*R_[m, :], (-sh)*R_[i, :] + ch*R_[m, :]
    return R_[:-2, :]


####### 4 #######
####### (1) #######
def Full_QR_Down(A, b):
    m, n = A.shape
    Q, R = full_qr_del(A)
    b = b[:-1]

    y = (Q.T @ b).T
    x = np.asarray(np.linalg.solve(R[:n, :], y[:n]))
    x = np.squeeze(np.asarray(x))
    return x


####### (2) #######
def Reduced_QR_Down(A, b): # Reduced QR Downdating
    A_b = np.hstack((A, np.asmatrix(b).T))
    Q, R = reduced_qr_del(A_b)
    R_, u = R[:-1, :-1], R[:-1, -1]

    x = np.linalg.solve(R_, u)
    return x


####### (3) #######
def R_Down(A, b):
    m, n = A.shape
    Q, _ = qr_givens(A[:-1, :])
    R = r_qr_del(A)
    b = b[:-1]

    y = (Q.T @ b).T
    x = np.linalg.solve(R[:n, :], y[:n])
    return x



####### artificial example #######
def matrix_gen(m, n):
    cond_P = 10000**2     # Condition number
    log_cond_P = np.log(cond_P)
    exp_vec = np.arange(-log_cond_P/n, log_cond_P * (n + 1)/(n * (n - 1)), log_cond_P/((n/2)*(n-1)))
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((np.random.rand(m, n) - 5.) * 200)
    V, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    return P


def test_qr_downdating(A):

    print("==============================Problem 1 & 2 & 3==============================")
    
    A_ = A[:-1, :]
    Q, R = np.linalg.qr(A)

    print("Random Matrix A:\n", A, "\n")
    print("Random Matrix A_:\n", A_, "\n")

    print("Q Matrix of A_:\n", Q, "\n")
    print("R Matrix of A_:\n", R, "\n")


    Q1, R1 = full_qr_del(A)
    Q2, R2 = reduced_qr_del(A)
    R3 = r_qr_del(A)

    print("-------------------Full QR Downdating-------------------")

    print("Q Matrix of A_:\n", Q1, "\n")
    print("R Matrix of A_:\n", R1, "\n")

    print("Q @ R =\n", Q1 @ R1, "\n")

    print("-------------------Reduced QR Downdating-------------------")

    print("Q Matrix of A_:\n", Q2, "\n")
    print("R Matrix of A_:\n", R2, "\n")

    print("Q @ R =\n", Q2 @ R2, "\n")


    print("-------------------Cholesky Downdating-------------------")

    print("R Matrix of A_:\n", R3, "\n")

    print("A_.T @ A_ =\n", A_.T @ A_, "\n")
    print("R.T @ R =\n", R3.T @ R3, "\n")


def test_least_square(A, b):

    print("==============================Problem 4==============================")

    print("Artificial Matrix A:\n", A, "\n")
    print("Artificial Vector b:\n", b, "\n")

    print("-------------------Solution Comparison-------------------")

    print("Numpy package:\n", np.linalg.lstsq(A[:-1, :], b[:-1])[0], "\n")
    print("Full QR Downdating:\n", Full_QR_Down(A, b), "\n")
    print("Reduced QR Downdating:\n", Reduced_QR_Down(A, b), "\n")
    print("Cholesky R Downdating:\n", R_Down(A, b), "\n")
    



####### Print Result #######
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.set_printoptions(precision=16, suppress=True, linewidth=200)
    
    Test_A = np.random.rand(5,3)
    test_qr_downdating(Test_A)

    # Gram-Schmidt breaks
    A = np.random.rand(6)
    A = np.asmatrix(A)
    for i in range(19):
        xi = A[-1, :] + 0.000000001
        A = np.vstack((A, xi))
    b = np.random.rand(20)
    test_least_square(A, b)

    # Deleted row dominating
    A = matrix_gen(20, 6)
    A[-1, :] = 10000000*A[-1, :]
    b = np.random.rand(20)
    #b[-1] = 10000000*b[-1]

    test_least_square(A, b)
    

