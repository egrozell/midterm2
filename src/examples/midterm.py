from midterms.q1  import (
        relerror
        )
import numpy as np
def main():
    L=3.828e26
    L_delta=0.004e26
    a=0.306
    a_delta=0.001
    D=1.496e11
    D_delta=0.025e11
    stefan=5.670374419e-8
    L_error = relerror(L,L_delta)
    a_error = relerror(a,a_delta)
    D_error = relerror(D,D_delta)

    T_expected_k = (L*(1-a)**0.25)/(16*np.pi*stefan*D**2)
    T_expected_c=T_expected_k-272.15
    T_error_L = L_delta*(((1-a)**0.25)/(16*np.pi*stefan*D**2))
    T_error_a = a_delta*((-L*(0.25))/(16*np.pi*stefan*((1-a)**.75)*D**2))
    T_error_D = D_delta*((-2*L*(1-a)**.25)/(16*np.pi*stefan*(D**3)))
    T_approx =T_expected_k +T_error_L +T_error_a +T_error_D
    total_error=T_error_L +T_error_a +T_error_D
    T_relative= relerror(T_approx,T_expected_k)

    print(f"""
    b)
    i.    L error {L_error}
    ii.   a error {a_error}
    iii.  D error {D_error}
    iv. Expected value of T in Kelvin = {T_expected_k}
        Expected value of T in Celsius= {T_expected_c}
    v.  Total error in T    = {total_error}
    contribution of L error =  {T_error_L}
    contribution of a error = {T_error_a}
    contribution of D error = {T_error_D}
    vi. Relative error of T =  {T_relative}
    """
         )
    K=4e11/0.275
    ma=3.68e4*4.15e-2
    k = np.array(
           [

               [ 3*K,-1*K, 0, 0],
               [-1*K, 2*K,-1*K, 0],
               [ 0,-1*K, 3*K,-2*K],
               [ 0, 0,-2*K, 2*K],
           ],
           dtype=float,
       )
    print(f"k.shape = {k.shape}")
    print(k)

    f = np.array([1,1.5*ma,1*ma,4*ma])
    print(f"f.shape = {f.shape}")
    print(f)

    u = np.linalg.solve(k, f)
    print(f"u.shape = {u.shape}")
    print(u)
    a = np.array(
           [

               [ 3,-1, 0, 0],
               [-1, 2,-1, 0],
               [ 0,-1, 3,-2],
               [ 0, 0,-2, 2],
           ],
           dtype=float,
       )
    a_norm=np.linalg.norm(a)
    b = np.array(
           [

               [ .5,.5, 0.5, 0.5],
               [.5, 3/2,3/2, 3/2],
               [ .5,3/2,5/2,5/2],
               [ 1/2, 3/2,5/2, 3],
           ],
           dtype=float,
       )
    b_norm=np.linalg.norm(b)
    cn= a_norm*b_norm
    print(cn)

    c = np.array(
           [

               [0,0,0,1],
               [0,0,1,0],
               [0,1,0,0],
               [1,0,0,0],
           ],
           dtype=float,
       )
    c_norm=np.linalg.norm(c)
    d = np.array(
           [

               [1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [0,0,0,1],

           ],
           dtype=float,
       )
    d_norm=np.linalg.norm(d)
    cn1= c_norm*d_norm
    print(cn1)

if __name__ == "__main__":
    main()
