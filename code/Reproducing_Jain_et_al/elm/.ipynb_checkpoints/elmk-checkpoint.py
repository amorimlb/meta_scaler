# -*- coding: utf-8 -*-

"""
    elm.elmk
    --------

    This file contains ELMKernel classes and all developed methods.
"""

# # Python2 support
# from __future__ import unicode_literals
# from __future__ import division
# from __future__ import absolute_import
# from __future__ import print_function

from .mltools import MLTools
from .mltools import kfold_cross_validation
from .mltools import time_series_cross_validation
from .mltools import copy_doc_of

import numpy as np
import scipy.linalg

try:
    import optunity
except ImportError:
    _OPTUNITY_AVAILABLE = 0
else:
    _OPTUNITY_AVAILABLE = 1


import ast
import sys
# if sys.version_info < (3, 0):
#     import ConfigParser as configparser
# else:
#     import configparser


# # Find configuration file
# from pkg_resources import Requirement, resource_filename
# _ELMK_CONFIG = resource_filename(Requirement.parse("elm"), "elm/elmk.cfg")


class ELMKernel(MLTools):
    """
        A Python implementation of ELM Kernel defined by Huang[1].

        An ELM is a single-hidden layer feedforward network (SLFN) proposed by
        Huang  back in 2006, in 2012 the author revised and introduced a new
        concept of using kernel functions to his previous work.

        This implementation currently accepts both methods proposed at 2012,
        random neurons and kernel functions to estimate classifier/regression
        functions.

        Let the dimensionality "d" of the problem be the sum of "t" size (number of
        targets per pattern) and "f" size (number of features per pattern).
        So, d = t + f

        The data will be set as Pattern = (Target | Features).

        If database has *N* patterns, its size follows *Nxd*.


        Note:
            [1] Paper reference: G.-B. Huang, H. Zhou, X. Ding, and R. Zhang,
            “Extreme Learning Machine for Regression and Multiclass
            Classification,” IEEE Transactions on Systems, Man, and Cybernetics
            - Part B: Cybernetics,  vol. 42, no. 2, pp. 513-529, 2012.

        Attributes:
            output_weight (numpy.ndarray): a column vector (*Nx1*) calculated
                after training, represent :math:\\beta.
            training_patterns (numpy.ndarray): a matrix (*Nxd*) containing all
                patterns used for training.

                Need to save all training patterns to perform kernel
                calculation at testing and prediction phase.
            param_kernel_function (str): kernel function that will be used
                for training.
            param_c (float): regularization coefficient (*C*) used for training.
            param_kernel_params (list of float): kernel function parameters
                that will be used for training.

        Other Parameters:
            regressor_name (str): The name of classifier/regressor.
            available_kernel_functions (list of str): List with all available
                kernel functions.
            default_param_kernel_function (str): Default kernel function if
                not set at class constructor.
            default_param_c (float): Default parameter c value if
                not set at class constructor.
            default_param_kernel_params (list of float): Default kernel
                function parameters if not set at class constructor.

        Note:
            * **regressor_name**: defaults to "elmk".
            * **default_param_kernel_function**: defaults to "rbf".
            * **default_param_c**: defaults to 9.
            * **default_param_kernel_params**: defaults to [-15].

    """

    def __init__(self, params=[]):
        """
            Class constructor.

            Arguments:
                params (list): first argument (*str*) is an available kernel
                    function, second argument (*float*) is the coefficient
                    *C* of regularization and the third and last argument is
                    a list of arguments for the kernel function.

            Example:

                >>> import elm
                >>> params = ["linear", 5, []]
                >>> elmk = elm.ELMKernel(params)

        """
        super(self.__class__, self).__init__()

        self.regressor_name = "elmk"

        self.available_kernel_functions = ["rbf", "linear", "poly"]

        # Default parameters
        self.default_param_kernel_function = "rbf"
        self.default_param_c = 9
        self.default_param_kernel_params = [-15]

        # Initialized parameters values
        if not params:
            self.param_kernel_function = self.default_param_kernel_function
            self.param_c = self.default_param_c
            self.param_kernel_params = self.default_param_kernel_params
        else:
            self.param_kernel_function = params[0]
            self.param_c = params[1]
            self.param_kernel_params = params[2]

        self.output_weight = []
        self.training_patterns = []

    # ########################
    # Private Methods
    # ########################

    def _kernel_matrix(self, training_patterns, kernel_type, kernel_param,
                       test_patterns=None):

        """ Calculate the Omega matrix (kernel matrix).

            If test_patterns is None, then the training Omega matrix will be
            calculated. This matrix represents the kernel value from each
            pattern of the training matrix with each other. If test_patterns
            exists, then the test Omega matrix will be calculated. This
            matrix  represents the kernel value from each pattern of the
            training matrix with the patterns of test matrix.

            Arguments:
                training_patterns (numpy.ndarray): A matrix containing the
                    features from all training patterns.
                kernel_type (str): The type of kernel to be used e.g: rbf
                kernel_param (list of float): The parameters of the chosen
                    kernel.
                test_patterns (numpy.ndarray): An optional parameter used to
                 calculate the Omega test matrix.

            Returns:
                numpy.ndarray: Omega matrix

        """
        number_training_patterns = training_patterns.shape[0]

        if kernel_type == "rbf":
            if test_patterns is None:
                temp_omega = np.dot(
                    np.sum(training_patterns ** 2, axis=1).reshape(-1, 1),
                    np.ones((1, number_training_patterns)))

                temp_omega = temp_omega + temp_omega.conj().T

                omega = np.exp(
                    -(2 ** kernel_param[0]) * (temp_omega - 2 * (np.dot(
                        training_patterns, training_patterns.conj().T))))

            else:
                number_test_patterns = test_patterns.shape[0]

                temp1 = np.dot(
                    np.sum(training_patterns ** 2, axis=1).reshape(-1, 1),
                    np.ones((1, number_test_patterns)))
                temp2 = np.dot(
                    np.sum(test_patterns ** 2, axis=1).reshape(-1, 1),
                    np.ones((1, number_training_patterns)))
                temp_omega = temp1 + temp2.conj().T

                omega = \
                    np.exp(- (2 ** kernel_param[0]) *
                           (temp_omega - 2 * np.dot(training_patterns,
                                                    test_patterns.conj().T)))
        elif kernel_type == "linear":
            if test_patterns is None:
                omega = np.dot(training_patterns, training_patterns.conj().T)
            else:
                omega = np.dot(training_patterns, test_patterns.conj().T)

        elif kernel_type == "poly":
            # Power a**x is undefined when x is real and 'a' is negative,
            # so is necessary to force an integer value
            kernel_param[1] = round(kernel_param[1])

            if test_patterns is None:
                temp = np.dot(training_patterns, training_patterns.conj().T) + \
                       kernel_param[0]

                omega = temp ** kernel_param[1]
            else:
                temp = np.dot(training_patterns, test_patterns.conj().T) + \
                       kernel_param[0]

                omega = temp ** kernel_param[1]

        else:
            print("Error: Invalid or unavailable kernel function.")
            return

        return omega

    def _local_train(self, training_patterns, training_expected_targets,
                     params):

        # If params not provided, uses initialized parameters values
        if not params:
            pass
        else:
            self.param_kernel_function = params[0]
            self.param_c = params[1]
            self.param_kernel_params = params[2]

        # Need to save all training patterns to perform kernel calculation at
        # testing and prediction phase
        self.training_patterns = training_patterns

        number_training_patterns = self.training_patterns.shape[0]

        # Training phase

        omega_train = self._kernel_matrix(self.training_patterns,
                                          self.param_kernel_function,
                                          self.param_kernel_params)

        # print("fy")
        # self.output_weight = np.dot(
        #     scipy.linalg.inv((omega_train + np.eye(number_training_patterns) /
        #                       (2 ** self.param_c))), training_expected_targets)\
        #     .reshape(-1, 1)

        self.output_weight = np.linalg.solve(
            (omega_train + np.eye(number_training_patterns) /
             (2 ** self.param_c)),
            training_expected_targets).reshape(-1, 1)

        training_predicted_targets = np.dot(omega_train, self.output_weight)

        return training_predicted_targets

    def _local_test(self, testing_patterns, testing_expected_targets,
                    predicting):

        omega_test = self._kernel_matrix(self.training_patterns,
                                          self.param_kernel_function,
                                          self.param_kernel_params,
                                          testing_patterns)

        testing_predicted_targets = np.dot(omega_test.conj().T,
                                           self.output_weight)

        return testing_predicted_targets

    # ########################
    # Public Methods
    # ########################

    def get_available_kernel_functions(self):
        """
            Return available kernel functions.
        """

        return self.available_kernel_functions

    def print_parameters(self):
        """
            Print parameters values.
        """

        print()
        print("Regressor Parameters")
        print()
        print("Regularization coefficient: ", self.param_c)
        print("Kernel Function: ", self.param_kernel_function)
        print("Kernel parameters: ", self.param_kernel_params)
        self.print_cv_log()

    def search_param(self, database, dataprocess=None, path_filename=("", ""),
                     save=False, cv="ts", cv_nfolds=10, of="rmse", kf=None,
                     opt_f="cma-es", eval=50, print_log=True):
        """
            Search best hyperparameters for classifier/regressor based on
            optunity algorithms.

            Arguments:
                database (numpy.ndarray): a matrix containing all patterns
                    that will be used for training/testing at some
                    cross-validation method.
                dataprocess (DataProcess): an object that will pre-process
                    database before training. Defaults to None.
                path_filename (tuple): *TODO*.
                save (bool): *TODO*.
                cv (str): Cross-validation method. Defaults to "ts".
                cv_nfolds (int): Number of Cross-validation folds. Defaults
                    to 10.
                of (str): Objective function to be minimized at
                    optunity.minimize. Defaults to "rmse".
                kf (list of str): a list of kernel functions to be used by
                    the search. Defaults to None, this set all available
                    functions.
                opt_f (str): Name of optunity search algorithm. Defaults to
                    "cma-es".
                eval (int): Number of steps (evaluations) to optunity algorithm.

            Returns:


            Each set of hyperparameters will perform a cross-validation
            method chosen by param cv.


            Available *cv* methods:
                - "ts" :func:`mltools.time_series_cross_validation()`
                    Perform a time-series cross-validation suggested by Hydman.

                - "kfold" :func:`mltools.kfold_cross_validation()`
                    Perform a k-fold cross-validation.

            Available *of* function:
                - "accuracy", "rmse", "mape", "me".


            See Also:
                http://optunity.readthedocs.org/en/latest/user/index.html
        """

        if not _OPTUNITY_AVAILABLE:
            raise Exception("Please install 'deap' and 'optunity' library to \
                             perform search_param.")

        if kf is None:
            search_kernel_functions = self.available_kernel_functions
        elif type(kf) is list:
            search_kernel_functions = kf
        else:
            raise Exception("Invalid format for argument 'kf'.")

        if print_log:
            print(self.regressor_name)
            print("##### Start search #####")

#         config = configparser.ConfigParser()

#         if sys.version_info < (3, 0):
#             config.readfp(open(_ELMK_CONFIG))
#         else:
#             config.read_file(open(_ELMK_CONFIG))
        

        best_function_error = 99999.9
        temp_error = best_function_error
        best_param_c = 0
        best_param_kernel_function = ""
        best_param_kernel_param = []
        for kernel_function in search_kernel_functions:

            if sys.version_info < (3, 0):
                elmk_c_range = [(-15,15)]

                n_parameters = 1
                kernel_p_range = [(-15,15)]

            else:
                kernel_config = 'rbf'

                elmk_c_range = [(-15,15)]

                n_parameters = 1
                kernel_p_range = [(-15,15)]

            param_ranges = [[elmk_c_range[0][0], elmk_c_range[0][1]]]
            for param in range(n_parameters):
                    param_ranges.append([kernel_p_range[param][0],
                                         kernel_p_range[param][1]])

            def wrapper_0param(param_c):
                """
                    Wrapper for objective function.
                """

                if cv == "ts":
                    cv_tr_error, cv_te_error = \
                        time_series_cross_validation(self, database,
                                                     [kernel_function,
                                                      param_c,
                                                      list([])],
                                                     number_folds=cv_nfolds,
                                                     dataprocess=dataprocess)

                elif cv == "kfold":
                    cv_tr_error, cv_te_error = \
                        kfold_cross_validation(self, database,
                                               [kernel_function,
                                                param_c,
                                                list([])],
                                               number_folds=cv_nfolds,
                                               dataprocess=dataprocess)

                else:
                    raise Exception("Invalid type of cross-validation.")

                if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
                    util = 1 / cv_te_error.get(of)
                else:
                    util = cv_te_error.get(of)

                # print("c:", param_c, "util: ", util)
                return util

            def wrapper_1param(param_c, param_kernel):
                """
                    Wrapper for optunity.
                """

                if cv == "ts":
                    cv_tr_error, cv_te_error = \
                        time_series_cross_validation(self, database,
                                                     [kernel_function,
                                                      param_c,
                                                      list([param_kernel])],
                                                     number_folds=cv_nfolds,
                                                     dataprocess=dataprocess)

                elif cv == "kfold":
                    cv_tr_error, cv_te_error = \
                        kfold_cross_validation(self, database,
                                               [kernel_function,
                                                param_c,
                                                list([param_kernel])],
                                               number_folds=cv_nfolds,
                                               dataprocess=dataprocess)

                else:
                    raise Exception("Invalid type of cross-validation.")

                if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
                    util = 1 / cv_te_error.get(of)
                else:
                    util = cv_te_error.get(of)

                # print("c:", param_c, " gamma:", param_kernel, "util: ", util)
                return util

            def wrapper_2param(param_c, param_kernel1, param_kernel2):
                """
                    Wrapper for optunity.
                """

                if cv == "ts":
                    cv_tr_error, cv_te_error = \
                        time_series_cross_validation(self, database,
                                                     [kernel_function,
                                                      param_c,
                                                      list([param_kernel1,
                                                            param_kernel2])],
                                                     number_folds=cv_nfolds,
                                                     dataprocess=dataprocess)

                elif cv == "kfold":
                    cv_tr_error, cv_te_error = \
                        kfold_cross_validation(self, database,
                                               [kernel_function,
                                                param_c,
                                                list([param_kernel1,
                                                      param_kernel2])],
                                               number_folds=cv_nfolds,
                                               dataprocess=dataprocess)

                else:
                    raise Exception("Invalid type of cross-validation.")

                if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
                    util = 1 / cv_te_error.get(of)
                else:
                    util = cv_te_error.get(of)

                # print("c:", param_c, " param1:", param_kernel1,
                #       " param2:", param_kernel2, "util: ", util)
                return util

            if kernel_function == "linear":
                optimal_parameters, details, _ = \
                    optunity.minimize(wrapper_0param,
                                      solver_name=opt_f,
                                      num_evals=eval,
                                      param_c=param_ranges[0])

            elif kernel_function == "rbf":
                optimal_parameters, details, _ = \
                    optunity.minimize(wrapper_1param,
                                      solver_name=opt_f,
                                      num_evals=eval,
                                      param_c=param_ranges[0],
                                      param_kernel=param_ranges[1])

            elif kernel_function == "poly":
                optimal_parameters, details, _ = \
                    optunity.minimize(wrapper_2param,
                                      solver_name=opt_f,
                                      num_evals=eval,
                                      param_c=param_ranges[0],
                                      param_kernel1=param_ranges[1],
                                      param_kernel2=param_ranges[2])
            else:
                raise Exception("Invalid kernel function.")

            # Save best kernel result
            if details[0] < temp_error:
                temp_error = details[0]

                if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
                    best_function_error = 1 / temp_error
                else:
                    best_function_error = temp_error

                best_param_kernel_function = kernel_function
                best_param_c = optimal_parameters["param_c"]

                if best_param_kernel_function == "linear":
                    best_param_kernel_param = []
                elif best_param_kernel_function == "rbf":
                    best_param_kernel_param = [optimal_parameters["param_kernel"]]
                elif best_param_kernel_function == "poly":
                    best_param_kernel_param = \
                        [optimal_parameters["param_kernel1"],
                         round(optimal_parameters["param_kernel2"])]
                else:
                    raise Exception("Invalid kernel function.")

                # print("best: ", best_param_kernel_function,
                #       best_function_error, best_param_c, best_param_kernel_param)

            if print_log:
                if of == "accuracy" or of == "hr" or of == "hr+" or of == "hr-":
                    print("Kernel function: ", kernel_function,
                          " best cv value: ", 1/details[0])
                else:
                    print("Kernel function: ", kernel_function,
                          " best cv value: ", details[0])

        # ELM attribute
        self.param_c = best_param_c
        self.param_kernel_function = best_param_kernel_function
        self.param_kernel_params = best_param_kernel_param

        params = [self.param_kernel_function,
                  self.param_c,
                  self.param_kernel_params]

        # MLTools Attribute
        self.has_cv = True
        self.cv_name = cv
        self.cv_error_name = of
        self.cv_best_error = best_function_error
        self.cv_best_params = params

        if print_log:
            print("##### Search complete #####")
            self.print_parameters()

        return params

    @copy_doc_of(MLTools._ml_train)
    def train(self, training_matrix, params=[]):
        return self._ml_train(training_matrix, params)

    @copy_doc_of(MLTools._ml_test)
    def test(self, testing_matrix, predicting=False):
        return self._ml_test(testing_matrix, predicting)

    @copy_doc_of(MLTools._ml_predict)
    def predict(self, horizon=1):

        return self._ml_predict(horizon)

    @copy_doc_of(MLTools._ml_train_it)
    def train_it(self, database_matrix, params=[], dataprocess=None,
                 sliding_window=168, k=1, search=False):
        return self._ml_train_it(database_matrix, params, dataprocess,
                                 sliding_window, k, search)

    @copy_doc_of(MLTools._ml_predict_it)
    def predict_it(self, horizon=1, dataprocess=None):
        return self._ml_predict_it(horizon, dataprocess)

