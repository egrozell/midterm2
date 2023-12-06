from midterms.q1  import (
        relerror
        )
import numpy as np
def main():

    K=4e11/0.275
    ma=3.68e4*4.15e-2
    k = np.array(
           [

               [ 3*K,-1*K    , 0, 0],
               [ 0  , (5/3)*K,-1*K, 0],
               [ 0  ,     0  , (12/5)*K,-2*K],
               [ 0, 0,0, (11/3)*K],
           ],
           dtype=float,
       )
    print(f"k.shape = {k.shape}")
    print(k)

    f = np.array([1*ma,(11/6)*ma,(21/10)*ma,(23/4)*ma])
    print(f"f.shape = {f.shape}")
    print(f)

    u = np.linalg.solve(k, f)
    print(f"u.shape = {u.shape}")
    print(u)


if __name__ == "__main__":
    main()
