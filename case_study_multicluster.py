from murty_auction import murtys, auction, compute_number_of_possible_assos
import numpy as np


if __name__ == "__main__":
    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ])

    n, mpn = R.shape
    m = mpn - n

    R_LC = np.hstack((np.diag(R[:, m:])[:,None], R[:,:m]))

    prior_hypotheses_per_cluster: list[list[tuple[list[int], float]]] = [
        [
            (np.array([1, 2]), 0.5),
            (np.array([1, 3]), 0.5)
        ],
        [
            (np.array([4]), 0.5),
            (np.array([5]), 0.5)
        ]
    ]

    A = R.T

    existing_tracks = [
        [1, 2, 4],
        [1, 3, 4],
        [1, 2, 5],
        [1, 3, 5]
    ]
    existing_tracks = [np.array(ts) - 1 for ts in existing_tracks]
    all_tracks = np.arange(n)
    s = 0
    for ts in existing_tracks:
        Rsub = R[ts]
        nonexisting_tracks = np.delete(all_tracks, existing_tracks)
        Rsub = np.delete(R[ts], nonexisting_tracks + m, axis=1)
        A = Rsub.T
        s += compute_number_of_possible_assos(A)


    print(f"number of possible assos: {s}")

    assignments = auction(A)

    for t, j in enumerate(assignments):
        track = t + 1
        asso = j + 1
        if asso > m:
            asso = 0
        print(f"a({track}) = {asso}")