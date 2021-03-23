from sympy import integrals, simplify, cos
from sympy.abc import x


def Jacobi(k, n):
    pj = [x] * (n + 1)

    for j in range(n + 1):
        if j == 0:
            pj[j] = 1
        elif j == 1:
            pj[j] = (1 + k) * x
        else:
            tmp_3 = (j + 2 * k) * j
            tmp_1 = (j + k) * (2 * (j - 2) + 2 * k + 3)
            tmp_2 = (j + k) * (j + k - 1)
            pj[j] = (tmp_1 * x * pj[j - 1] - tmp_2 * pj[j - 2]) / tmp_3
    return pj


def Jacobi_value(k, n, y):
    vals = [0] * (n + 1)
    for j in range(n + 1):
        if j == 0:
            vals[j] = 1
        elif j == 1:
            vals[j] = (1 + k) * y
        else:
            tmp_3 = (j + 2 * k) * j
            tmp_1 = (j + k) * (2 * (j - 2) + 2 * k + 3)
            tmp_2 = (j + k) * (j + k - 1)
            vals[j] = (tmp_1 * y * vals[j - 1] - tmp_2 * vals[j - 2]) / tmp_3
    return vals


def base_funs(k, n):
    # находим массивы координатных функций Ф_i, i = 1, ... , n
    # в нашем случае массив от 0 до n невключительно, где соответственно Ф[0] = Ф_1
    phi = [x] * (n)
    dphi = [x] * (n)

    jacobs = Jacobi(k, n)
    djacobs = Jacobi(k - 1, n + 1)  # это не производные полиномо Якоби
    for i in range(n):
        phi[i] = (1 - x ** 2) * jacobs[i]
        phi[i] = simplify(phi[i])

        dphi[i] = (-2) * (i + 1) * (1 - x ** 2) ** (k - 1) * djacobs[i + 1]
        dphi[i] = simplify(dphi[i])

    return phi, dphi


def base_fun_ddphi(k, n):
    ddphi = [x] * (n)
    jacobs = Jacobi(k, n)
    djacobs = Jacobi(k - 1, n + 1)  # это не производные полиномо Якоби
    for i in range(n):
        tmp1 = (k - 1) * (1 - x ** 2) ** (k - 2) * djacobs[i + 1]
        tmp2 = (1 - x ** 2) ** (k - 1) * ((i + 1 + 2 * (k - 1) + 1) / 2) * jacobs[i]
        ddphi[i] = (-2) * (i + 1) * ( tmp1 + tmp2 )
        ddphi[i] = simplify(ddphi[i])
    return ddphi


def base_funs_values(k, n, y):
    phi = [0] * n
    dphi = [0] * n
    jacobs = Jacobi_value(k, n, y)
    djacobs = Jacobi_value(k - 1, n + 1, y)  # это не производные полиномо Якоби
    for i in range(n):
        phi[i] = (1 - y ** 2) * jacobs[i]
        dphi[i] = (-2) * (i + 1) * (1 - y ** 2) ** (k - 1) * djacobs[i + 1]
    return phi, dphi


def base_funs_val_ddphi(k, n, y):
    ddphi = [0] * n
    jacobs = Jacobi_value(k, n, y)
    djacobs = Jacobi_value(k - 1, n + 1, y)  # это не производные полиномо Якоби
    for i in range(n):
        tmp1 = (1 - y ** 2) ** (k - 2) * (k - 1) * djacobs[i + 1]
        tmp2 = (1 - y ** 2) ** (k - 1) * ((i + 1 + 2 * (k - 1) + 1) / 2) * jacobs[i]
        ddphi[i] = (-2) * (i + 1) * ( tmp1 + tmp2 )
    return ddphi



