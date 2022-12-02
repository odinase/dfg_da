import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'

import scipy
import scipy.linalg as la


def ellipse(mu, P, s, n):
    thetas = np.linspace(0, 2*np.pi, n)
    ell = mu + s * (la.cholesky(P).T @ np.array([np.cos(thetas), np.sin(thetas)])).T
    return ell

def main():
    x1 = np.array([1.5, 1.2]) + np.array([0.1, -0.3])
    x2 = np.array([1.2, 1.5])

    d = x1 - x2
    dn = np.linalg.norm(d)
    d = d / dn

    # Standard deviations
    sx2 = np.array([dn/5.0, 1.2*dn])
    Vx2 = np.array([
        [d[0],  d[1]],
        [d[1], -d[0]]
    ])

    Sx2 = np.diag(sx2)
    Px2 = Vx2 @ Sx2 @ Vx2.T

    ell_x2 = ellipse(x2, Px2, 1, 200)

    sx1 = sx2[::-1]
    Vx1 = Vx2#[:, ::-1]

    Sx1 = np.diag(sx1)
    Px1 = Vx1 @ Sx1 @ Vx1.T

    ell_x1 = ellipse(x1, Px1, 1, 200)

    z1 = x2 + 0.25*d
    dp = np.array([d[1], -d[0]])
    z2 = x2 - 0.5*dp

    z = np.vstack((z1, z2))

    fig, ax = plt.subplots()

    ax.plot(*ell_x2.T, 'b')
    ax.plot(*x2, 'bo')
    ax.annotate(r'$\bm{x}^2$', tuple(x2), tuple(x2 + np.array([0.05, 0.0])), fontsize=12)

    ax.plot(*ell_x1.T, 'g')
    ax.plot(*x1, 'go')
    ax.annotate(r'$\bm{x}^1$', tuple(x1), tuple(x1 + np.array([0.05, 0.0])), fontsize=12)
    
    ax.plot(*z.T, 'rx')
    meas_names = [r'$\bm{z}^1$', r'$\bm{z}^2$']
    for zz, text in zip(z, meas_names):
        ax.annotate(text, tuple(zz), tuple(zz + np.array([0.05, 0.0])), fontsize=12)


    ax.grid(True, alpha=0.2)
    ax.set_xlim([0, 2.5])
    ax.set_ylim([0, 2.5])

    plt.show()

    fig.tight_layout()
    fig.savefig("./figures/association_mtt_example.pdf", bbox_inches = 'tight')


if __name__ == "__main__":
    main()