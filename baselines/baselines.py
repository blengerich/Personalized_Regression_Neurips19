import numpy as np
np.set_printoptions(precision=3)
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression, LogisticRegression
import time

from mixture import LinearRegressionsMixture
from mixture.logistic_regression_mixtures import LogisticRegressionsMixture
from utils import *
from functions import *
from Lasso import Lasso


def logistic_lasso(X, Y, lam, max_iters=1000, lr=1e-2, tol=1e-4, verbosity=100, silent=False):
    if not silent:
        print("Fitting Logistic Regression with Lasso Reg.")
    t = time.time()
    f = logistic_loss
    f_prime = logistic_loss_prime
    rho_beta = lambda beta: lam*lasso_penalty(beta, np.zeros_like(beta))
    rho_beta_prime = lambda beta: lam*lasso_derivative(beta, np.zeros_like(beta))

    N = len(X)
    P = len(X[0])
    prev_loss = np.inf
    beta_hat = np.zeros((P))
    for iteration in range(max_iters):
        if not silent:
            print("Iteration {} of {}".format(iteration+1, max_iters), end='\r')
        grad_beta = np.zeros((P))
        for i in range(len(X)):
            grad_beta += f_prime(X[i], Y[i], beta_hat)
        grad_beta /= N
        grad_beta += rho_beta_prime(beta_hat)

        loss1 = np.mean([f(X[i], Y[i], beta_hat) for i in range(N)])
        loss2 = rho_beta(beta_hat)
        loss = loss1 + loss2
        if not silent and iteration % verbosity == 0:
            print("Estimate at Iteration {}\n{}".format(iteration, beta_hat))
            print("Loss at Iteration     {}: {}".format(iteration, loss))
        if loss > 1e8:
            if not silent:
                print("Diveraged at iteration {}".format(iteration))
            break
        if loss > prev_loss:
            if not silent:
                print("Reached local min at iteration {}".format(iteration))
            break
        prev_loss = loss
        prev_beta_hat = beta_hat.copy()
        beta_hat -= lr*grad_beta

    return np.array([prev_beta_hat.T for i in range(N)])


def linear_lasso(X, Y, lam, max_iters=1000, lr=1e-2, tol=1e-4, verbosity=100):
	# Population Estimator
    print("Fitting Linear Regression with Lasso Reg.")
    t = time.time()
    f = linear_loss
    f_prime = linear_loss_prime
    rho_beta = lambda beta: lam*lasso_penalty(beta, np.zeros_like(beta))
    rho_beta_prime = lambda beta: lam*lasso_derivative(beta, np.zeros_like(beta))

    N = len(X)
    P = len(X[0])
    prev_loss = np.inf
    beta_hat = np.zeros((P))
    for iteration in range(max_iters):
        print("Iteration {} of {}".format(iteration+1, max_iters), end='\r')
        grad_beta = np.zeros((P))
        for i in range(len(X)):
            grad_beta += f_prime(X[i], Y[i], beta_hat)
        grad_beta /= N
        grad_beta += rho_beta_prime(beta_hat)

        loss1 = np.mean([f(X[i], Y[i], beta_hat) for i in range(N)])
        loss2 = rho_beta(beta_hat)
        loss = loss1 + loss2
        if iteration % verbosity == 0:
            print("Estimate at Iteration {}\n{}".format(iteration, beta_hat))
            print("Loss at Iteration     {}: {}".format(iteration, loss))
        if loss > 1e8:
            print("Diveraged at iteration {}".format(iteration))
            break
        if loss > prev_loss:
            print("Reached local min at iteration {}".format(iteration))
            break
        prev_loss = loss
        prev_beta_hat = beta_hat.copy()
        beta_hat -= lr*grad_beta

    return np.array([prev_beta_hat.T for i in range(N)])


def mixture_model_logistic(X, Y, n_classes, lam, n_restarts=5, verbosity=100,
    init_lr=4e-3, n_iterations=2000, eps=1e-3):
    # Mixture Model
    print("Fitting Mixture Model for Logistic Regression")
    N = len(X)
    t = time.time()
    mixture = LogisticRegressionsMixture(X, Y,
        K=n_classes, fit_intercept=False)
    verbose = verbosity > 0
    mixture.train(epsilon=eps, lam=lam,
        iterations=n_iterations, random_restarts=n_restarts,
        verbose=verbose, silent=False, init_lr=init_lr)
    mixture_beta = mixture.w.T    # KxP
    mixture_beta = mixture_beta[np.argmax(mixture.gamma, axis=1)]
    print("-Took {:.2f} seconds".format(time.time() - t))
    return mixture_beta, np.argmax(mixture.gamma, axis=1)


def mixture_model_linear(X, Y, n_classes, lam, fit_intercept=False,
    n_restarts=5, max_iters=500, init_lr=1e-3,
    verbose=False, silent=False, init_beta=None):
    # Mixture Model
    print("Fitting Mixture Model for Linear Regression")
    t = time.time()
    mixture = LinearRegressionsMixture(X, Y, K=n_classes, fit_intercept=fit_intercept,
        init_w=init_beta)
    mixture.train(epsilon=1e-5, lam=lam, iterations=max_iters,
        init_lr=init_lr, random_restarts=n_restarts, verbose=verbose, silent=silent)
    mixture_beta = mixture.w.T    # KxP
    mixture_beta_samples = []
    for i in range(len(X)):
        best_assignment = np.argmax(mixture.gamma[i, :])
        mixture_beta_samples.append(mixture_beta[best_assignment].copy())
    mixture_beta = np.array(mixture_beta_samples)
    print("Took {:.2f} seconds".format(time.time() - t))
    return mixture_beta, np.argmax(mixture.gamma, axis=1)


def vc_logistic(X, Y, U, lam, lr, verbosity=50, tol=1e-8,
    max_iters=2000, n_restarts=1, init_Z=None, lr_decay=1-1e-6):
    """ Assumes that beta = U Z. """
    print("Fitting Varying Coefficients with Logistic Output.")
    t = time.time()
    N = X.shape[0]
    assert(U.shape[0] == N)
    U = np.hstack((np.ones((N, 1)), U)) # prepend with column of ones for intercept
    P = X.shape[1]
    K = U.shape[1]

    if init_Z is None:
        Z = np.zeros((K, P))#np.random.normal(0, 1e-5, size=(P, K))
    else:
        Z = init_Z

    prev_loss = np.inf
    initial_lr = lr
    initial_patience = 10

    rho_Z = lambda Z: lam*np.linalg.norm(Z, ord=1)
    rho_Z_prime = lambda Z: lam*np.sign(Z)

    best_loss = np.inf
    best_beta_hat = None
    best_Z = Z.copy()

    for restart in range(n_restarts):
        t = time.time()
        print("Restart {} of {}".format(restart+1, n_restarts))
        if init_Z is None:
            Z = np.zeros((K, P))#np.random.normal(0, 1e-5, size=(P, K))
        else:
            Z = init_Z
        prev_loss = np.inf
        patience = initial_patience
        lr = initial_lr

        for iteration in range(max_iters):
            beta_hat_vc = np.array([U[i].dot(Z).T for i in range(N)])
            loss1 = np.mean([logistic_loss(X[i], Y[i], beta_hat_vc[i]) for i in range(N)])
            loss2 = rho_Z(Z)
            loss = loss1 + loss2
            if loss > 1e10:
                print("Diverged at iteration:{}".format(iteration))
                break

            if iteration % verbosity == 0:
                print("Iteration: {:d} Total Loss:{:.3f} Pred:{:.3f} l1:{:.3f}".format(
                    iteration, loss, loss1, loss2))
            lr *= lr_decay
            if loss > prev_loss - tol:
                patience -= 1
                if patience <= 0:
                    print("Reached local minimum at iteration {:d}.".format(iteration))
                    beta_hat_vc = beta_prev
                    break

            # Record previous values
            beta_prev = beta_hat_vc.copy()
            prev_loss = loss

            # Calculate gradients
            grad_Z = rho_Z_prime(Z)
            for i in range(N):
                grad_Z += 1 * (np.expand_dims(X[i], 1).dot(np.expand_dims(U[i], 0))*(
                    np.exp(X[i].dot(U[i].dot(Z))) /
                    (1 + np.exp(X[i].dot(U[i].dot(Z)))) - Y[i])).T
            Z -= lr*grad_Z
        print("Took {:.3f} seconds.".format(time.time() - t))

        # Don't really need this since it is convex loss.
        if loss < best_loss:
            print("** New best solution **")
            best_loss = loss
            best_beta_hat = beta_hat_vc.copy()
            best_Z = Z.copy()

    print("Took {:.2f} seconds".format(time.time() - t))
    return best_beta_hat, best_Z


def vc_linear(X, Y, U, lam, lr,
    verbosity=50, tol=1e-5, initial_patience=0,
    max_iters=2000):
    """ Assumes that beta = U Z. """
    lr_decay = 1-1e-6
    print("Fitting Varying Coefficients with Linear Output.")
    t = time.time()
    N = X.shape[0]
    assert(U.shape[0] == N)
    U = np.hstack((np.ones((N, 1)), U)) # prepend with column of ones for intercept
    P = X.shape[1]
    K = U.shape[1]

    Z = np.random.normal(0, 1e-3, size=(P, K))

    prev_loss = np.inf
    initial_lr = lr

    rho_Z = lambda Z: lam*np.linalg.norm(Z, ord=1)
    rho_Z_prime = lambda Z: lam*np.sign(Z)

    n_restarts = 1
    best_loss = np.inf
    best_beta_hat = None

    for restart in range(n_restarts):
        t = time.time()
        print("Restart {} of {}".format(restart+1, n_restarts))
        Z = np.random.normal(0, 1, size=(K, P))
        grad_Z = np.zeros_like(Z)
        prev_loss = np.inf
        patience = initial_patience
        lr = initial_lr

        for iteration in range(max_iters):
            Z -= lr*grad_Z
            beta_hat_vc = np.array([U[i].dot(Z).T for i in range(N)])
            loss1 = np.mean([linear_loss(X[i], Y[i], beta_hat_vc[i]) for i in range(N)])
            #calc_prediction_error(Y, beta_hat_vc, X, N)
            loss2 = rho_Z(Z)
            loss = loss1 + loss2
            if loss > 1e10 and iteration > 0:
                print("Diverged at iteration:{}".format(iteration))
                break

            if iteration % verbosity == 0:
                print("Iteration: {:d} Total Loss:{:.3f} Pred:{:.3f} l1:{:.3f}".format(
                    iteration, loss, loss1, loss2))
            lr *= lr_decay
            if loss > prev_loss - tol:
                patience -= 1
                if patience < 0:
                    print("Reached local minimum at iteration {:d}.".format(iteration))
                    beta_hat_vc = beta_prev
                    break

            # Record previous values
            beta_prev = beta_hat_vc.copy()
            prev_loss = loss

            # Calculate gradients
            grad_Z = rho_Z_prime(Z)
            for i in range(N):
                grad_Z -= (Y[i] - X[i].dot(beta_hat_vc[i]))*np.expand_dims(U[i], 1).dot(np.expand_dims(X[i], 0))

        print("Took {:.3f} seconds.".format(time.time() - t))

        # Don't really need this since it is convex loss.
        if loss < best_loss:
            print("** New best solution **")
            best_loss = loss
            best_beta_hat = beta_hat_vc.copy()
            best_Z = Z.copy()

    print("Took {:.2f} seconds".format(time.time() - t))
    return best_beta_hat, best_Z