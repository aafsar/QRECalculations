"""Trace a smooth parameterized curve using a predictor-corrector method.
"""
import math
import numpy as np
import scipy.linalg


def qr_decomp(b: np.array):
    q, b = scipy.linalg.qr(b, overwrite_a=True)
    return q.transpose(), b


def newton_step(q, b, u, y):
    # Expects q and b to be 2-D arrays, and u and y to be 1-D arrays
    # Returns steplength
    for k in range(b.shape[1]):
        y[k] -= np.dot(b[:k, k], y[:k])
        y[k] /= b[k, k]

    d = 0.0
    for k in range(b.shape[0]):
        s = np.dot(q[:-1, k], y)
        u[k] -= s
        d += s * s
    return math.sqrt(d)


class Bunch:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def solve_continuation(
        fun,
        t_span,
        x0,
        jac,
        first_step=0.03,
        min_step=1.0e-8,
        max_decel=1.1,
        maxiter=1000,
        tol=1.0e-4,
        crit=None,
        callback=None
) -> Bunch:
    """
    Solve a parameterized system of equations by numerical continuation.

    Parameters
    ----------
    fun : callable
        Left-hand side of the system.  Calling signature is
        ``fun(x)``, where `x` has shape (n+1,), consisting of the
        n variables of the system followed by the parameter.

    t_span : 2-tuple of floats
        Interval of solution (t0, tf).  The solver starts at t=t0
        and traces a path until it reaches a solution with parameter
        outside of (t0, tf).

    x0 : array_like, shape (n,)
        Initial solution value valid at the first element of `t_span`.

    jac : callable
        Jacobian of the system.  Calling signature is
        ``jac(x)`` following the same conventions as ``fun``.
        Returns an array with shape (n, n+1), with each row corresponding
        to one equation and each column one of the variables.

    first_step : float, optional
        Initial stepsize

    min_step : float, optional
        Minimum allowed step size.  If the step size falls below this
        length, the tracing terminates.

    max_decel : float, optional
        The maximum deceleration/acceleration rate for the step size.

    maxiter : int, optional
        Maximum number of iterations to perform at a step.

    tol : float, optional
        Tolerance for error estimate in corrector step.

    crit : callable, optional
        Criterion function.  The signature is `callback(xk) -> float`,
        where `xk` is the current point.  If provided, the tracing attempts
        to find a point on the curve where `crit(xk)` is zero.

    callback : callable, optional
        Function to call after each step.  The signature is `callback(xk)`,
        where `xk` is the accepted point.

    Returns
    -------
    Bunch object with the following fields defined:

    points : list
        The points computed along the parameterized curve.
    nfev : int
        Number of evaluations of the left-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    message : string
        Human-readable description of the termination reason.
    success : bool
        True if the solver reached an end of the interval

    References
    ----------
    .. [1] Allgower, Eugene L. and Georg, Kurt (1990)  *Introduction to
           Numerical Continuation Methods*.
    """
    # Maximum distance to curve
    max_dist = 0.4
    # Maximum contraction rate in corrector
    max_contr = 0.6
    # Perturbation to avoid cancellation in calculating contraction rate
    eta = 0.1

    x = np.append(x0, t_span[0])
    if callback:
        callback(x)
    # The last column of Q in the QR decomposition is the tangent vector
    t = qr_decomp(jac(x).transpose())[0][-1]
    h = first_step
    # Determines whether we are using Newton steplength (for zero-finding)
    newton = False
    # Set orientation of curve, so we are tracing into the interior of `t_span`
    omega = np.sign(t[-1]) * np.sign(t_span[1] - t_span[0])
    bunch = Bunch(nfev=0, njev=1, points=[x])

    while min(t_span) <= x[-1] <= max(t_span):
        accept = True

        if abs(h) <= min_step:
            # Stepsize below minimum tolerance; terminate.
            bunch.success = False
            bunch.message = f"Stepsize {abs(h)} less than minimum {min_step}."
            return bunch

        # Predictor step
        u = x + h * omega * t
        q, b = qr_decomp(jac(u).transpose())
        bunch.njev += 1

        disto = 0.0
        decel = 1.0 / max_decel  # deceleration factor
        for it in range(maxiter):
            y = fun(u)
            bunch.nfev += 1
            dist = newton_step(q, b, u, y)
            if dist >= max_dist:
                accept = False
                break
            decel = max(decel, math.sqrt(dist / max_dist) * max_decel)
            if it > 0:
                contr = dist / (disto + tol * eta)
                if contr > max_contr:
                    accept = False
                    break
                decel = max(decel, math.sqrt(contr / max_contr) * max_decel)
            if dist < tol:
                # Success; break out of iteration
                break
            disto = dist
        else:
            # We have run out of iterations; terminate.
            bunch.success = False
            bunch.message = f"Maximum number of iterations {maxiter} reached."
            return bunch

        if not accept:
            # Step was not accepted; take a smaller step and try again
            h /= max_decel
            continue  # back out to main loop to try again

        if not newton and crit:
            # Enter Newton mode if we have passed over a critical point
            # between the last point `x` and the new point `u`.
            newton = crit(x, t) * crit(u, q[-1]) < 0.0

        # Determine new stepsize
        if newton:
            # Newton-type steplength adaptation, secant method
            h *= -crit(u, q[-1]) / (crit(u, q[-1]) - crit(x, t))
        else:
            # Standard steplength adaptation
            h = abs(h / min(decel, max_decel))

        # Update with outcome of successful PC step
        if sum(t * q[-1]) < 0.0:
            # The orientation of the curve as determined by the QR
            # decomposition has changed.
            #
            # The original Allgower-Georg QR decomposition implementation
            # ensures the sign of the tangent is stable; when it is stable
            # the curve orientation switching is a sign of a bifurcation.
            # The QR decomposition implementation in scipy does not have
            # this property and therefore cannot be use for bifurcation
            # detection.
            omega = -omega
        x, t = u, q[-1]
        bunch.points.append(x)
        if callback:
            callback(x)

    bunch.success = True
    bunch.message = "Computation completed successfully."
    return bunch
