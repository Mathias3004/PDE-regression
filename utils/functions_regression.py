import numpy as np
import time
import itertools 

def FiniteDiff(u, dx, d, bc="periodic"):
    """
    Takes dth derivative from 1D array u_i using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3

    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    d = order of the derivative
    bc = boundary condition

    Returns: dth derivative D^n(u)/dx^n.

    # Note that numpy.roll([u], 1) cyclically shifts entries in the array forward
    # numpy.roll([u1, u2, u3, ..], 1) = [uN, u1, u2, u3, ...]
    """

    n = u.size
    ux = np.zeros(n, dtype=np.complex64)
    if d == 0:
        return u

    if d == 1:
        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

        if bc == "open":
            ux[0] = (-3. / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
            ux[n - 1] = (3. / 2 * u[n - 1] - 2 * u[n - 2] + u[-3] / 2) / dx
        return ux

    if d == 2:
        ux = (np.roll(u, -1) + np.roll(u, 1) - 2 * u) / dx ** 2

        if bc == "open":
            ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
            ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[-3] - u[n - 4]) / dx ** 2
        return ux

    if d == 3:
        ux = (
            np.roll(u, -2) / 2 - np.roll(u, -1) + np.roll(u, 1) - np.roll(u, 2) / 2
        ) / dx ** 3

        if bc == "open":
            ux[0] = (
                -2.5 * u[0] + 9 * u[1] - 12 * u[2] + 7 * u[3] - 1.5 * u[4]
            ) / dx ** 3
            ux[1] = (
                -2.5 * u[1] + 9 * u[2] - 12 * u[3] + 7 * u[4] - 1.5 * u[5]
            ) / dx ** 3
            ux[n - 1] = (
                2.5 * u[n - 1]
                - 9 * u[n - 2]
                + 12 * u[-3]
                - 7 * u[n - 4]
                + 1.5 * u[n - 5]
            ) / dx ** 3
            ux[n - 2] = (
                2.5 * u[n - 2]
                - 9 * u[-3]
                + 12 * u[n - 4]
                - 7 * u[n - 5]
                + 1.5 * u[n - 6]
            ) / dx ** 3
        return ux

    if d > 3:
        return FiniteDiff(FiniteDiff(u, dx, 2, bc), dx, d - 2, bc)


def FiniteDiff_t(u, dt, d):
    """
    Takes time derivative using 2nd order finite difference scheme.
    Edges are processed separately by imposing "open boundary" condition
    at t=0 and t=tf.
    This allows to get 2nd order precision vs 1st order precision at the start/end time steps.

    Input:
    u(t_i) = data to be differentiated (1D array)
    dt = time spacing.  Assumes uniform spacing.
    """

    ut = FiniteDiff(u, dt, d, bc="open")
    return ut

def TotalFiniteDiff(u, dx, d, bc="periodic"):
    """Calculate d-th order spatial derivative at all time points"""
    assert len(u.shape) == 2
    m, n = u.shape
    Du = np.zeros((m, n), dtype=u.dtype)
    for i in range(m):
        Du[i, :] = FiniteDiff(u[i, :], dx, d, bc=bc)
    return Du


def TotalFiniteDiff_t(u, dt, d=1, bc=""):
    """Calculate 1st order time derivative at all spatial points"""
    assert len(u.shape) == 2
    m, n = u.shape
    Du = np.zeros((m, n), dtype=u.dtype)
    for i in range(n):
        Du[:, i] = FiniteDiff_t(u[:, i], dt, d)
    return Du

def MultiFiniteDiff(u, dx, d, axis, bc="periodic"):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3

    Input:
    u(x,y) = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    axis = 0 -> derivative over x coordinate
    axis = 1 -> derivative over y coordinate
    """

    if d == 0:
      return u

    ux = np.zeros(u.shape, dtype=np.complex64)
    n = u.shape[1]
    # assert u.shape[1] == u.shape[2], "Support only square arrays"
    if d == 1:
        ux = (np.roll(u, 1, axis=axis) - np.roll(u, -1, axis=axis)) / (2 * dx)

        if bc == "open":
            if axis == 0:
                ux[0, :] = (-3.0 / 2 * u[0, :] + 2 * u[1, :] - u[2, :] / 2) / dx
                ux[-1, :] = (3.0 / 2 * u[-1, :] - 2 * u[-2, :] + u[-3, :] / 2) / dx
            if axis == 1:
                ux[:, 0] = (-3.0 / 2 * u[:, 0] + 2 * u[:, 1] - u[:, 2] / 2) / dx
                ux[:, -1] = (3.0 / 2 * u[:, -1] - 2 * u[:, -2] + u[:, -3] / 2) / dx
        return ux

    if d == 2:
        ux = (np.roll(u, 1, axis=axis) + np.roll(u, -1, axis=axis) - 2 * u) / dx ** 2
        if bc == "open":
            if axis == 0:
                ux[0, :] = (2 * u[0, :] - 5 * u[1, :] + 4 * u[2, :] - u[3, :]) / dx ** 2
                ux[-1, :] = (
                    2 * u[-1, :] - 5 * u[-2, :] + 4 * u[-3, :] - u[-4, :]
                ) / dx ** 2
            if axis == 1:
                ux[:, 0] = (2 * u[:, 0] - 5 * u[:, 1] + 4 * u[:, 2] - u[:, 3]) / dx ** 2
                ux[:, -1] = (
                    2 * u[:, -1] - 5 * u[:, -2] + 4 * u[:, -3] - u[:, -4]
                ) / dx ** 2

        return ux

    if d == 3:
        if bc == "open":
            raise NotImplementedError()

        ux = (
            np.roll(u, 2, axis=axis) / 2
            - np.roll(u, 1, axis=axis)
            + np.roll(u, -1, axis=axis)
            - np.roll(u, -2, axis=axis) / 2
        ) / dx ** 3
        return ux

    if d > 3:
        return MultiFiniteDiff(MultiFiniteDiff(u, dx, 2, axis, bc), dx, d - 2, axis, bc)

def MultiFiniteDiff_t(u, dt, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3

    Input:
    u = data to be differentiated
    dt = time spacing.  Assumes uniform spacing
    """

    # TODO: use FiniteDiff_t function here
    m = u.shape[0]  # t-axis
    ut = np.zeros(u.shape, dtype=np.complex64)

    if d == 1:
        ut[0] = (u[1] - u[0]) / dt
        for i in range(1, m - 1):
            ut[i] = (u[i + 1] - u[i - 1]) / (2 * dt)
        ut[-1] = (u[-1] - u[-2]) / dt
        return ut

    if d == 2:
        for i in range(1, m - 1):
            ut[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dt ** 2

        ut[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dt ** 2
        ut[m - 1] = (2 * u[m - 1] - 5 * u[m - 2] + 4 * u[m - 3] - u[m - 4]) / dt ** 2
        return ut

    if d > 2:
        raise NotImplementedError()


def MultiFourierDiff(u, dx, d, axis):
    """
    Takes dth derivative data using Spectral method (Fourier)

    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """

    m, n, n = u.shape
    q = np.fft.fftfreq(n, dx) * 2 * np.pi
    uhat = np.fft.fft(u, axis=axis)
    ux = np.fft.ifft((1j * q) ** d * uhat, axis=axis)

    return ux

def build_custom_Theta(
    data,
    derivatives,
    derivatives_description,
    data_description=[],
    add_constant_term=True,
):
    """
    builds a matrix with columns representing polynoimials up to degree P of all variables

    This is used when we subsample and take all the derivatives point by point or if there is an
    extra input (Q in the paper) to put in.

    input:
        data: column 0 is U, and columns 1:end are Q
        derivatives: a bunch of derivatives of U and maybe Q, should start with a column of ones
        derivatives_description: description of what derivatives have been passed in
        P: max power of polynomial function of U to be included in Theta

    returns:
        Theta = Theta(U,Q)
        descr = description of what all the columns in Theta are
    """

    n, m = data.shape
    if len(derivatives) > 0:
        m, d2 = derivatives.shape
        if n != m:
            raise Exception("dimension error")
    if data_description != []:
        d = data.shape[1]
        if len(data_description) != d:
            raise Exception("data descrption error")

    # First column of Theta is just ones.
    if add_constant_term:
        Theta = np.ones((n, 1), dtype=np.complex64)
        descr = [""]
    else:
        Theta = np.array([], dtype=np.complex64).reshape((n, 0))
        descr = []
    # Add "u"-part into Theta
    if len(data) > 0:
        Theta = np.hstack([Theta, data])
        descr += data_description
    # Add the derivaitves part into Theta
    if len(derivatives) > 0:
        for D in range(0, derivatives.shape[1]):
            Theta = np.hstack([Theta, (derivatives[:, D]).reshape(n, 1)])
        descr += derivatives_description
    return Theta, descr

def BruteForceL0(
    R,
    Ut,
    descr,
    lam_l2=1e-5,
    l0_penalty=None,
    num_l0_steps=30,
    split=0.8,
    verbose=False,
    seed=0,
    add_l2_loss=False,
    lhs_descr='u_t',
    fixed_columns=[]
):
    """
    This function trains a predictor using Brute force search over subsets of all possible 2^M combinations of terms.

    It runs over different values of tolerance and trains predictors on a training set, then evaluates them
    using a loss function on a holdout set.
    """
    start = time.time()
    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(seed)  # for consistancy
    n, _ = R.shape
    train = np.random.choice(n, int(n * split), replace=False)
    test = np.setdiff1d(np.arange(n), train)
    TrainR = R[train, :]
    TestR = R[test, :]
    TrainY = Ut[train, :]
    TestY = Ut[test, :]
    X = TrainR
    D = TrainR.shape[1]
    F = len(fixed_columns)

    train_log = {}
    # assert D-F <= 15, "Brute force search is supported for <= 15 candidate terms"
    if l0_penalty is None:
        l0_penalty = 0.001 * np.linalg.cond(R)
    if verbose:
        print("Brute Force search with l0_penalty", l0_penalty)
    # Get the standard least squares estimator
    w_best = np.linalg.lstsq(X, TrainY)[0]
    err_best = np.linalg.norm(
        TestY - TestR.dot(w_best), 2
    ) + l0_penalty * np.count_nonzero(w_best)
    if add_l2_loss:
        err_best += lam_l2 * np.linalg.norm(w_best)

    best_pde = print_pde(w_best, descr, lhs_descr=lhs_descr, verbose=False)

    for i, indexes in enumerate(itertools.product([True, False], repeat=D-F)):

        # add the fixed indices as True on right postitions
        indexes = list(indexes)
        for key in fixed_columns:
          ind = descr.index(key)
          indexes.insert(ind,True)
        indexes = tuple(indexes)

        if np.all(1 - np.array(indexes)):  # all indexes == False (no terms in fit)
            continue
        if verbose:
            print(f'iteration {i} from total {2**(D-F)}')
        # iterate over all possible subsets of terms
        w = np.zeros((D, 1), dtype=np.complex64)
        X = TrainR[:, list(indexes)]
        non_zero_w = np.count_nonzero(w)

        if lam_l2 != 0:
            # import pdb; pdb.set_trace()
            w[list(indexes)] = np.linalg.lstsq(
                X.T.dot(X) + lam_l2 * np.eye(X.shape[1]), X.T.dot(TrainY)
            )[0]
            err = np.linalg.norm(
                TestY - TestR.dot(w), 2
            ) + l0_penalty * np.count_nonzero(w)
        else:
            w[list(indexes)] = np.linalg.lstsq(TrainR[:, list(indexes)], TrainY)[0]
            err = np.linalg.norm(
                TestY - TestR.dot(w), 2
            ) + l0_penalty * np.count_nonzero(w)
        if add_l2_loss:
            err += lam_l2 * np.linalg.norm(w)

        # Has the accuracy improved?
        if err < err_best:
            err_best = err
            w_best = w
            best_pde = print_pde(w_best, descr, lhs_descr=lhs_descr, verbose=False)

        if verbose:
            print("Current PDE")
            print_pde(w, descr)
            print(
                f"Error {err}, Error best: {err_best}, l0_penalty: {l0_penalty}, pde: "
            )
            print("Best PDE")
            print_pde(w_best, descr, lhs_descr=lhs_descr)

    train_log["err_best"] = err_best
    train_log["best_pde"] = best_pde
    train_log["best_xi"] = w_best
    print_pde(w_best, descr, lhs_descr=lhs_descr)
    end = time.time()
    print("Time elapsed", end - start)
    return w_best, train_log

def BruteForceL0Scan(
    R, Ut, descr, l0_init=1e-8, l0_fin=0.1, lam_l2=1e-5, num_points=10, verbose=False
):
    l0_arr = np.logspace(np.log10(l0_init), np.log10(l0_fin), num=num_points)
    train_data = []
    results = {"l0_arr": l0_arr, "train_data": train_data}
    for l0 in l0_arr:
        w_best, train_log = BruteForceL0(R, Ut, descr, l0_penalty=l0, lam_l2=lam_l2)
        train_data.append(train_log)
    return results

def print_pde(w, rhs_description, lhs_descr="u_t", verbose=True):
    pde = lhs_descr + " = "
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + " + "
            pde = (
                pde
                + "(%05f %+05fi)" % (w[i].real, w[i].imag)
                + rhs_description[i]
                + "\n   "
            )
            first = False
    if verbose:
        print(pde)
    return pde

def LinearSearch(X, ut, descr, num_terms, verbose=0, lhs_descr="u_t", fixed_terms=[]):
  # A number of terms of the PDE can be fixed (the ones you are sure to include) 
  # and one by one terms of library are added and best one is selected

  collected_terms = fixed_terms
  collected_i = []
  remaining_terms = []
  remaining_i = []

  for ind, term in enumerate(descr):
    if term in collected_terms:
      collected_i.append( ind )
    else:
      remaining_i.append( ind )
      remaining_terms.append( term )

  if num_terms == 0:
    try_i = tuple(collected_i)
    result = BruteForceL0(X[:,try_i], ut, collected_terms, l0_penalty=0., lam_l2=0., 
                                  verbose=verbose, lhs_descr=lhs_descr, fixed_columns=collected_terms)
    return [result], collected_terms

  else:
    train_data = []
    n_terms = 0
    while n_terms < num_terms:
      err_best = np.inf
      for ind, term in enumerate(remaining_terms):
        try_i = tuple(collected_i + [ind])
        try_terms = collected_terms + [term]

        result = BruteForceL0(X[:,try_i], ut, try_terms, l0_penalty=0., lam_l2=0., 
                                    verbose=verbose, lhs_descr=lhs_descr, fixed_columns=try_terms)
        err = result[1]['err_best']
        if err < err_best:
          new_term = term
          new_ind = ind
          result_best = result 
          err_best = err
          try_best = try_terms

      # save result
      train_data.append(result_best)

      # for next search
      collected_i.append(new_ind)
      collected_terms.append(new_term)
      remaining_i.remove(new_ind)
      remaining_terms.remove(new_term)

      n_terms += 1

      # print
      print("Current error, including " + str(n_terms) + " terms: " + str(err_best))
      print("Recovered PDE:")
      print_pde(result_best[0], try_best, lhs_descr=lhs_descr)

    return train_data, collected_terms