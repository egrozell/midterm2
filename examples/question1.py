import numpy as np
from linalg import linalg_interp as l

def main():
    parta_data = np.loadtxt('question1.txt')

    x = parta_data[:,0]
    y = parta_data[:,1]

    A = np.array(([[-1, 2,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [ 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 0, 1, 4, 1, 0, 0, 0, 0, 0, 0],
                   [ 0, 0, 0, 1, 4, 1, 0, 0, 0, 0, 0],
                   [ 0, 0, 0, 0, 1, 4, 1, 0, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 1, 4, 1, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 0],
                   [ 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0],
                   [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1],
                   [ 0, 0, 0, 0, 0, 0, 0, 0,-1, 2,-1]]),dtype=float)
    B = np.zeros(11,dtype=float)
    for i in range(1,10):
        a = (y[i-1]-y[i])/(x[i-1]-x[i])
        b = (y[i-2]-y[i-1])/(x[i-2]-x[i-1])
        B[i] = 3*(a-b)
    print(A)
    b = np.resize(B, (11,1))
    print(b)
    xn = l.gauss_iter_solve(A,b,alg='jacobi')
    x1 = np.array(xn[:,1])
    s = np.zeros(11,dtype=float)
    print(x1)
    t = 2015.25
    for j in range(1,10):
        ai = y[j]
        di= (x1[j+1]-x1[j])/(3*(x[j+1]-x[j]))
        bi = (y[j]-y[j+1])/(x[j]-x[j+1])-x1[j]*(x[j]-x[j+1])-di*(x[j]-x[j+1])**2
        ci = x1[j]
        s[j]=ai +bi*(t-x[j])+ci*(t-x[j])+di*(t-x[j])**2
    print(s)
    sum = 0
    for k in range(1,10):
        sum += s[k]
    avg = sum / 9

    print(avg)

if __name__ == "__main__":
    main()
