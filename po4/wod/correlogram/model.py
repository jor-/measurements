import scipy.optimize
import numpy as np
import random
import math
import logging

from . import estimation
import measurements.util.calculate

class Correlation_Model_Base():

    def __init__(self, p_bounds, f_function, df_function, min_measurements=10, split_time=False):
        self.min_measurements = min_measurements
        self.d_dim = 4
        self.split_time = split_time
        self.p_bounds = p_bounds
        self.f_function = f_function
        self.df_function = df_function


    def copy(self):
        return Correlation_Model_Base(self.p_bounds, self.f_function, self.df_function, min_measurements=self.min_measurements)


    @property
    def p_dim(self):
        return len(self.__p_bounds)

    @property
    def p_bounds(self):
        return self.__p_bounds

    @p_bounds.setter
    def p_bounds(self, p_bounds):
        self.__p_bounds = p_bounds


    @property
    def correlations(self):
        try:
            self.__correlations
        except AttributeError:
            self.__correlations = estimation.get_correlation(min_measurements=self.min_measurements, discard_year=False)

        return self.__correlations


    @property
    def numbers(self):
        try:
            self.__numbers
        except AttributeError:
            self.__numbers = estimation.get_number(min_measurements=self.min_measurements, discard_year=False)

        return self.__numbers


    @property
    def directions(self):
        try:
            self.__directions
        except AttributeError:
            self.__directions = estimation.get_shifted_directions(min_measurements=self.min_measurements, discard_year=False)

        return self.__directions



    def _check_d(self, d):
        d = np.array(d)
        if d.shape == (self.d_dim,):
            return d
        else:
            raise ValueError('Direction has to be a vector of length %d, but its shape is %s.' % (self.d_dim , d.shape))

    def _check_p(self, p):
        p = np.array(p)
        if p.shape == (self.p_dim,):
            return p
        else:
            raise ValueError('Parameter has to be a vector of length %d, but its shape is %s.' % (self.p_dim, p.shape))


    def split_time_in_direction(self, d):
        n = len(d)

        x = np.empty(n+1)
        x[0:n] = d[0:n]
        x[n] = (1 - np.cos(d[0] * 2 * np.pi)) / 2
       # month = d[0] % 1
       # x[n] = min((month, 1 - month))
#         x[0] = math.floor(d[0])
#         x[1] = d[0] % 1
#         x[2:n+1] = d[1:n]

        return x


    def prepare_p_d(self, p, d):
        p = self._check_p(p)
        d = self._check_d(d)

        if self.split_time:
            d = self.split_time_in_direction(d)

        return (p, d)


    def f(self, p, d):
        p, d = self.prepare_p_d(p, d)
        return self.f_function(p, d)


    def df(self, p, d):
        p, d = self.prepare_p_d(p, d)
        return self.df_function(p, d)



    def F(self, p):
        F = 0

        directions = self.directions
        correlations = self.correlations
        numbers = self.numbers

        for i in range(len(directions)):
            F += numbers[i] * (self.f(p, directions[i]) - correlations[i]) ** 2

        return F


    def dF(self, p):
        dF = 0

        directions = self.directions
        correlations = self.correlations
        numbers = self.numbers

        for i in range(len(directions)):
            dF += numbers[i] * 2 * (self.f(p, directions[i]) - correlations[i]) * self.df(p, directions[i])

        return dF




    def get_random_p(self):
        p = ()
        for j in range(self.p_dim):
            p += (random.random(),)

        return p


    def get_random_d(self):
        d = ()
        for j in range(self.d_dim):
            d += (random.random(),)

        return d



    def check_grad(self):
        p = self.get_random_p()
        d = self.get_random_d()

        f_d = lambda p: self.f(p, d)
        df_d = lambda p: self.df(p, d)

        f_d_p = f_d(p)
        err_f_d = scipy.optimize.check_grad(f_d, df_d, p)
        F_p = self.F(p)
        err_F = scipy.optimize.check_grad(self.F, self.dF, p)

        rel_err = (err_f_d / f_d_p, err_F / F_p)

        return rel_err



    def optimize(self, method=None, number_of_starts=10, max_f_eval=1000, disp=False):
        if method is None:
            method = ('L-BFGS-B', 'TNC')
        elif isinstance(method, str):
            method = (method,)
        else:
            raise ValueError('Method has to be a string or a sequance of strings, but its type is: %s.' % type(method))

        p_dim = self.p_dim
        F_min = float('inf')
        p_min = None

        for current_method in method:
            for i in range(number_of_starts):
                ## get random p0
                p0 = self.get_random_p()

                ## optimize
                result = scipy.optimize.minimize(self.F, p0, jac=self.dF, bounds= self.p_bounds, options={'ftol':0.0, 'maxiter':max_f_eval, 'disp':disp}, method=current_method)

                ## check if better minima
                if result.success:
                    F = result.fun
                    if F < F_min:
                        F_min = F
                        p_min = result.x

        return p_min


## mult


class Correlation_Model_1(Correlation_Model_Base):
    # mult( p_i^d_i )

    def __init__(self, min_measurements=100, split_time=True):
        p_dim = 4
        if split_time:
            p_dim += 1

        eps = 10**(-8)
        p_bounds = ((eps, 1),) * p_dim

        def f_function(p, d):
            f = np.multiply.reduce(p ** d)

            return f

        def df_function(p, d):
            f = f_function(p, d)
            df = f * d / p

            return df

        Correlation_Model_Base.__init__(self, p_bounds, f_function, df_function, min_measurements=min_measurements, split_time=split_time)




class Correlation_Model_2(Correlation_Model_Base):
    # mult( 1 / (1 + p_i * d_i^j) )

    def __init__(self, order=1, min_measurements=100, split_time=True):
        p_dim = 4
        if split_time:
            p_dim += 1
        p_dim *= order
        p_bounds = ((0, None),) * p_dim

        def f_function(p, d):
            n = len(d)

            f = 1
            for i in range(order):
                k = i*n
                f *= np.multiply.reduce((1 + p[k:k+n] * d**(i+1))**(-1))

            return f

        def df_function(p, d):
            n = len(d)
            df = np.empty(p_dim, np.float)

            f = f_function(p, d)

            for i in range(order):
                k = i*n
                df[k:k+n] = f * (- d**(i+1)) / (1 + p[i*n:(i+1)*n] * d**(i+1))

            return df

        Correlation_Model_Base.__init__(self, p_bounds, f_function, df_function, min_measurements=min_measurements, split_time=split_time)



## sum


class Correlation_Model_3(Correlation_Model_Base):
    # sum( p_i^d_i ) / n

    def __init__(self, min_measurements=100, split_time=True):
        p_dim = 4
        if split_time:
            p_dim += 1

        eps = 10**(-8)
        p_bounds = ((eps, 1),) * p_dim

        def f_function(p, d):
            n = len(d)

            f = np.sum(p ** d)

            f /= n

            return f

        def df_function(p, d):
            n = len(d)

            df = d * p**d / p
            df /= n

            return df

        Correlation_Model_Base.__init__(self, p_bounds, f_function, df_function, min_measurements=min_measurements, split_time=split_time)




class Correlation_Model_4(Correlation_Model_Base):
    # sum( 1 / (1 + p_i_j * d_i^j) ) / (n * order)

    def __init__(self, order=1, min_measurements=100, split_time=True):
        d_dim = 4
        if split_time:
            d_dim += 1
        m = d_dim * order
        p_bounds = ((0, None),) * m

        def f_function(p, d):
            n = len(d)

            f = 0
            for i in range(order):
                k = i*n
                f += np.sum((1 + p[k:k+n] * d**(i+1))**(-1))

            f /= m

            return f

        def df_function(p, d):
            n = len(d)
            df = np.empty(len(p), np.float)

            for i in range(order):
                k = i*n
                df[k:k+n] = - d**(i+1) * (1 + p[k:k+n] * d**(i+1))**(-2)

            df /= m

            return df

        Correlation_Model_Base.__init__(self, p_bounds, f_function, df_function, min_measurements=min_measurements, split_time=split_time)


## sum extended by c


class Correlation_Model_5(Correlation_Model_Base):
    # (sum( c_i * p_i^d_i ) + (n - sum( c_i ))) / n

    def __init__(self, min_measurements=100, split_time=True):
        d_dim = 4
        if split_time:
            d_dim += 1

        eps = 10**(-8)
        p_bounds = ((eps, 1),) * (2 * d_dim)

        def f_function(p, d):
            n = len(d)

            f = np.sum(p[n:2*n] * p[:n] ** d[:n])
            f += n - np.sum(p[n:2*n])
            f /= n

            return f

        def df_function(p, d):
            n = len(d)
            df = np.empty(2*n)

            df[:n] = p[n:2*n] * d * p[:n]**d / p[:n]
            df[n:2*n] = p[:n]**d - 1

            df /= n

            return df

        Correlation_Model_Base.__init__(self, p_bounds, f_function, df_function, min_measurements=min_measurements, split_time=split_time)




class Correlation_Model_6(Correlation_Model_Base):
    # (sum( c_i_j / (1 + p_i_j * d_i^j) ) + (n * order - sum( c_i_j ))) / (n * order)

    def __init__(self, order=1, min_measurements=100, split_time=True):
        d_dim = 4
        if split_time:
            d_dim += 1
        m = d_dim * order
        p_bounds = ((0, None),) * m + ((0, 1),) * m

        def f_function(p, d):
            n = len(d)
            m = n * order

            f = 0
            for i in range(order):
                k = i*n
                f += np.sum(p[k+m:k+n+m] / (1 + p[k:k+n] * d**(i+1)))

            f += m - np.sum(p[m:2*m])

            f /= m

            return f

        def df_function(p, d):
            n = len(d)
            m = n * order
            df = np.empty(len(p), np.float)

            for i in range(order):
                k = i*n
                df[k:k+n] = - d**(i+1) * p[k+m:k+n+m] * (1 + p[k:k+n] * d**(i+1))**(-2)
                df[k+m:k+n+m] = 1 / (1 + p[k:k+n] * d**(i+1)) - 1

            df /= m

            return df

        Correlation_Model_Base.__init__(self, p_bounds, f_function, df_function, min_measurements=min_measurements, split_time=split_time)


## d squared


class Correlation_Model_7(Correlation_Model_Base):
    # mult( p_i^(d_i^2) )

    def __init__(self, min_measurements=100, split_time=True):
        p_dim = 4
        if split_time:
            p_dim += 1

        eps = 10**(-8)
        p_bounds = ((eps, 1),) * p_dim

        def f_function(p, d):
            f = np.multiply.reduce(p ** (d**2))

            return f

        def df_function(p, d):
            f = f_function(p, d)
            df = d**2 * f / p

            return df

        Correlation_Model_Base.__init__(self, p_bounds, f_function, df_function, min_measurements=min_measurements, split_time=split_time)


class Correlation_Model_8(Correlation_Model_Base):
    # mult( 1 / (1 + p_i * d_i^2) )

    def __init__(self, min_measurements=100, split_time=True):
        p_dim = 4
        if split_time:
            p_dim += 1
        p_bounds = ((0, None),) * p_dim

        def f_function(p, d):
            f = np.multiply.reduce((1 + p * d**2)**(-1))

            return f

        def df_function(p, d):
            f = f_function(p, d)
            df = f * (- d**2) / (1 + p * d**2)

            return df

        Correlation_Model_Base.__init__(self, p_bounds, f_function, df_function, min_measurements=min_measurements, split_time=split_time)




class Correlation_Model_9(Correlation_Model_Base):
    # sum( 1 / (1 + p_i_j * d_i^2) ) / (n)

    def __init__(self, min_measurements=100, split_time=True):
        d_dim = 4
        if split_time:
            d_dim += 1
        p_bounds = ((0, None),) * d_dim

        def f_function(p, d):
            n = len(d)
            f = np.sum((1 + p * d**2)**(-1))
            f /= n

            return f

        def df_function(p, d):
            n = len(d)
            df = - d**2 * (1 + p * d**2)**(-2)
            df /= n

            return df

        Correlation_Model_Base.__init__(self, p_bounds, f_function, df_function, min_measurements=min_measurements, split_time=split_time)




def extend_correlation_model_by_C(model):
    p_dim_old = model.p_dim
    p_bounds_old = model.p_bounds

    f_function_old = model.f_function
    df_function_old = model.df_function

    model.p_bounds = p_bounds_old + ((0,1),)

    def f_function(p, d):
        p_old = p[0:p_dim_old]
        f_old = f_function_old(p_old, d)

        c = p[p_dim_old]
        f = (1 - c) * f_old + c

        return f

    def df_function(p, d):
        p_old = p[0:p_dim_old]
        f_old = f_function_old(p_old, d)
        df_old = df_function_old(p_old, d)

        df = np.empty(p_dim_old + 1, np.float)
        c = p[p_dim_old]
        df[0:p_dim_old] = (1 - c) * df_old
        df[p_dim_old] = 1 - f_old

        return df

    model.f_function = f_function
    model.df_function = df_function

    return model







class Correlation_Model_Combined(Correlation_Model_Base):
    def __init__(self, model1, model2):
        p_bounds = model1.p_bounds + model2.p_bounds
        min_measurements = model1.min_measurements

        def split(p):
            p1 = p[0:model1.p_dim]
            p2 = p[model1.p_dim:len(p)]

            return (p1, p2)

        def f_function(p, d):
            (p1, p2) = split(p)

            f = model1.f(p1, d) * model2.f(p2, d)

            return f

        def df_function(p, d):
            (p1, p2) = split(p)

            df1 = model1.df(p1, d) * model2.f(p2, d)
            df2 = model1.f(p1, d) * model2.df(p2, d)
            df = np.concatenate((df1, df2))

            return df

        Correlation_Model_Base.__init__(self, p_bounds, f_function, df_function, min_measurements=min_measurements)



# class Correlation_Model():
#     def __init__(self, min_measurements=100):
# #         from .constants import P_1E_2E_OPT
# #
# #         correlation_model_1_E = extend_correlation_model_by_C(Correlation_Model_1(min_measurements=min_measurements))
# #         correlation_model_2_E = extend_correlation_model_by_C(Correlation_Model_2(min_measurements=min_measurements))
# #         self.correlation_model = Correlation_Model_Combined(correlation_model_1_E, correlation_model_2_E)
# #         self.p_opt = P_1E_2E_OPT
#         from .constants import P1E_OPT
#         self.correlation_model = Correlation_Model_1(min_measurements=min_measurements)
#         self.p_opt = P1E_OPT
#
#     def correlation(self, point_1, point_2):
#         from .constants import T_RANGE, X_RANGE
#
#         d = measurements.util.calculate.get_min_distance(point_1, point_2, t_range=T_RANGE, x_range=X_RANGE)
#         correlation = self.correlation_model.f(self.p_opt, d)
#         return correlation



class Correlation_Model():
    def __init__(self):
        from ..data.io import load_measurement_points

        self.logger = logging.getLogger(__name__)

        self.points = load_measurement_points()
#         self.correlation_model = Correlation_Model_Combined(extend_correlation_model_by_C(Correlation_Model_2()), Correlation_Model_6())
#         self.p_opt = np.array([  1.42467926e-02,   1.09009491e-01,   1.14544485e-01, 1.27031987e-02,   2.82689543e-01,   3.15936466e-01, 9.27357261e+01,   7.78565009e+04,   2.93241272e-01, 7.81488396e+04,   0.00000000e+00,   9.16141645e-01, 6.82714979e-01,   1.00000000e+00,   9.13380330e-02, 0.00000000e+00])
#         self.correlation_model = Correlation_Model_Combined(extend_correlation_model_by_C(Correlation_Model_8()), Correlation_Model_9())
#         self.p_opt = np.array([  1.16862407e-03,   2.02379355e-02,   1.29611737e-01, 8.47065220e-05,   4.61352239e-01,   4.69122777e-01, 5.08310357e+03,   2.24048931e+01,   1.36383989e-02, 7.30895852e-03,   0.00000000e+00])

        self.correlation_model = extend_correlation_model_by_C(Correlation_Model_1(split_time=False))
        self.p_opt = np.array([  1.00000000e-08,   5.37662216e-06,   1.00000000e-08, 8.70444360e-01,   1.07619669e-01])

    @property
    def n(self):
        return len(self.points.shape[0])


    def correlation(self, i, j):
        from .constants import T_RANGE, X_RANGE

        d = measurements.util.calculate.get_min_distance(self.points[i], self.points[j], t_range=T_RANGE, x_range=X_RANGE)
        correlation = self.correlation_by_distance(d)

        return correlation

    def correlation_by_distance(self, d):
        correlation = self.correlation_model.f(self.p_opt, d)

        return correlation