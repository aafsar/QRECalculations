import time

import numpy as np

import src.from_literature.pcjac as pcjac


def f(x: np.array) -> np.array:
    expx = np.exp(x[:-1])
    return np.array([
        x[0] - x[1] - x[4] * (4 * expx[2] - expx[3]),
        x[2] - x[3] - x[4] * (expx[1] - expx[0]),
        expx[0] + expx[1] - 1,
        expx[2] + expx[3] - 1
    ])


def jac(x: np.array) -> np.array:
    expx = np.exp(x[:-1])
    lam = x[-1]
    ret = np.zeros((4, 5))
    ret[0] = [
        1, -1, -4 * lam * expx[2], lam * expx[3], -4 * expx[2] + expx[3]
    ]
    ret[1] = [
        lam * expx[0], -lam * expx[1], 1, -1, -expx[1] + expx[0]
    ]
    ret[2] = [
        expx[0], expx[1], 0, 0, 0
    ]
    ret[3] = [
        0, 0, expx[2], expx[3], 0
    ]
    return ret


def callback(x):
    point = np.exp(x[:-1])
    print(x[-1], point)


def main():
    print("Solving Ochs game using Jacobian")
    x0 = np.array([np.log(0.5), np.log(0.5), np.log(0.5), np.log(0.5)])
    start = time.process_time()
    res = pcjac.solve_continuation(
        f,
        (0.0, 100.0),
        x0,
        jac,
        # callback=callback
    )
    print(
        f"Took {time.process_time() - start:.4f} for {res.nfev} function evaluations, {res.njev} Jacobian evaluations.")
    print(f"Computed {len(res.points)} points along curve.")
    print("Final point computed:")
    callback(res.points[-1])
    print()

if __name__ == "__main__":
    main()
