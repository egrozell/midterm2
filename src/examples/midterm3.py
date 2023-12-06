import numpy as np
import scipy
def main():
    a=15
    k = np.array(
           [
               [4*a,-a,0],
               [-a,3*a,-2*a],
               [0,-2*a,2*a],
           ],
           dtype=float,
       )
    det = np.linalg.det(k)
    print(k)
    print(det)
    g=9.81
    f = np.array([6*g,g,3*g])
    print(f)
    print ()

    u = np.linalg.solve(k, f)
    print(u)

if __name__ == "__main__":
    main()
