"""Draft README figures: training, unconditional denoising, SDA-steered denoising.

Layout follows Fig. 1 of Manshausen et al. (2024, JAMES, doi:10.1029/2024MS004505).
The fields are a stand-in: a Gaussian random field with an SSH-like spectrum, for
which the optimal denoiser is an exact Wiener filter. The sampling trajectories are
therefore real deterministic-DDIM and MMPS trajectories, just
with a closed-form denoiser instead of the trained SongUNet. Runs on CPU in seconds.

    python examples/figures/make_sketches.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent
N = 128
RNG = np.random.default_rng(7)

INK = "#222222"
LINE = "#555555"
PRIOR = "#3b6ea8"  # denoiser / prior
PRIOR_BG = "#eaf1f8"
OBS = "#c8642d"  # observations / guidance
OBS_BG = "#fbeee5"
CMAP = plt.get_cmap("RdBu_r").copy()
CMAP.set_bad("#ececec")
VLIM = 2.5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": LINE,
    "savefig.facecolor": "white",
})

# ---------------------------------------------------------------- toy prior

kx = np.fft.fftfreq(N)[:, None]
ky = np.fft.rfftfreq(N)[None, :]
K = np.sqrt(kx**2 + ky**2) * N
P = (K**2 + 3.0**2) ** -2.2  # SSH-like steep spectrum


def fft(x):
    return np.fft.rfft2(x, norm="ortho")


def ifft(X):
    return np.fft.irfft2(X, s=(N, N), norm="ortho")


def _unit_variance():
    x = np.fft.irfft2(np.sqrt(P) * np.fft.rfft2(np.random.default_rng(0).standard_normal((N, N)),
                                                 norm="ortho"), s=(N, N), norm="ortho")
    return x.var()


P /= _unit_variance()  # unit pixel variance, so sigma is in units of the field std


def prior_sample():
    x = ifft(np.sqrt(P) * fft(RNG.standard_normal((N, N))))
    return x / x.std()


def denoise(x, sigma):
    """Exact E[x0 | x_sigma] for the Gaussian prior (Wiener filter)."""
    return ifft(P / (P + sigma**2) * fft(x))


def covariance(v, sigma):
    """Cov[x0 | x_sigma] v = sigma^2 dD/dx v (Tweedie), as used by MMPS."""
    return ifft(sigma**2 * P / (P + sigma**2) * fft(v))


def swot_mask():
    """Two crossing SWOT passes: 2 swaths each with a nadir gap and a nadir track."""
    i, j = np.mgrid[0:N, 0:N]
    mask = np.zeros((N, N), bool)
    for slope, offset in [(0.22, 30), (-0.25, 98)]:
        d = (j - (slope * i + offset)) / np.sqrt(1 + slope**2)
        mask |= (np.abs(d) > 4) & (np.abs(d) < 15)
        mask |= np.abs(d) < 0.6  # nadir altimeter
    return mask


def mmps_denoise(x, sigma, mask, y, obs_std, iters=100):
    """Guided denoiser E[x0 | x_sigma, y] via MMPS (exact for a Gaussian prior)."""
    xhat = denoise(x, sigma)
    r = y - xhat[mask]

    def A(u):
        g = np.zeros((N, N))
        g[mask] = u
        return covariance(g, sigma)[mask] + obs_std**2 * u

    # conjugate gradients on (H V H^T + R) u = y - H xhat
    u = np.zeros_like(r)
    res = r.copy()
    p = res.copy()
    rs = res @ res
    for _ in range(iters):
        Ap = A(p)
        a = rs / (p @ Ap)
        u += a * p
        res -= a * Ap
        rs_new = res @ res
        if np.sqrt(rs_new) < 1e-6 * np.sqrt(r @ r):
            break
        p = res + rs_new / rs * p
        rs = rs_new
    g = np.zeros((N, N))
    g[mask] = u
    return xhat + covariance(g, sigma)


def ddim(steps=32, guide=None):
    """Deterministic DDIM in the EDM sigma parameterization. Returns per-step frames."""
    sigmas = np.geomspace(20.0, 0.01, steps)
    x = sigmas[0] * RNG.standard_normal((N, N))
    frames = []
    for s, s_next in zip(sigmas, np.append(sigmas[1:], 0.0)):
        xhat = denoise(x, s) if guide is None else mmps_denoise(x, s, *guide)
        frames.append((s, x.copy(), xhat))
        x = xhat + (s_next / s) * (x - xhat)
    return frames


def shown(x, sigma):
    """Noisy states are rescaled by 1/sqrt(1 + sigma^2) for display."""
    return x / np.sqrt(1 + sigma**2)


# ---------------------------------------------------------------- drawing helpers


def canvas(w, h, scale=0.82):
    fig = plt.figure(figsize=(w * scale, h * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def tile(ax, field, x, y, s, label=None, outline=None, below=None, z=2):
    ax.imshow(field, extent=(x, x + s, y, y + s), origin="lower", cmap=CMAP,
              vmin=-VLIM, vmax=VLIM, interpolation="bilinear", zorder=z)
    if outline is not None:
        ax.contour(np.linspace(x, x + s, N), np.linspace(y, y + s, N), outline.astype(float),
                   levels=[0.5], colors=[INK], linewidths=0.5, alpha=0.6, zorder=z + 0.1)
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=LINE, lw=0.8, zorder=z + 0.2))
    if label:
        ax.text(x + s / 2, y + s + 0.12, label, ha="center", va="bottom", fontsize=10)
    if below:
        ax.text(x + s / 2, y - 0.12, below, ha="center", va="top", fontsize=8.5, color=LINE)


def box(ax, x, y, w, h, text, fc, ec, fontsize=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                fc=fc, ec=ec, lw=1.2, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            linespacing=1.5, zorder=3)


def arrow(ax, start, end, color=LINE, rad=0.0, dashed=False, label=None, label_xy=None):
    ax.annotate("", xy=end, xytext=start, zorder=5, arrowprops=dict(
        arrowstyle="-|>", color=color, lw=1.3, mutation_scale=13, shrinkA=0, shrinkB=0,
        linestyle=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=9, color=color)


def group(ax, x, y, w, h, title):
    """Dashed frame with a tab label, for grouping the inputs or outputs of a diagram."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                fc="#f6f6f6", ec=LINE, lw=1.0, ls=(0, (4, 3)), zorder=1))
    ax.text(x + 0.2, y + h, title, ha="left", va="center", fontsize=9, fontweight="bold",
            color=LINE, zorder=6, bbox=dict(fc="white", ec="none", pad=1.5))


def panel_letter(ax, x, y, letter, title):
    ax.text(x, y, f"{letter})", fontsize=13, fontweight="bold", va="top")
    ax.text(x + 0.45, y - 0.02, title, fontsize=12, va="top")


# ---------------------------------------------------------------- a) training


def fig_training():
    x0 = prior_sample()
    eps = RNG.standard_normal((N, N))
    fig, ax = canvas(15, 5.0)
    panel_letter(ax, 0.2, 4.9, "a", "Training the prior (denoising score matching)")

    tile(ax, x0, 0.4, 0.9, 2.2, "training patch $x_0$", below="llc4320 SSH / SST")
    arrow(ax, (2.75, 2.0), (3.55, 2.0), label="add noise", label_xy=(3.15, 2.3))

    # stack of noise levels, back to front
    for k, s in enumerate([2.0, 0.8, 0.3]):
        off = 0.28 * (2 - k)
        tile(ax, shown(x0 + s * eps, s), 3.7 + off, 0.75 + off, 2.0, z=2 + k)
    ax.text(4.95, 3.55, r"$x_\sigma = x_0 + \sigma\,\epsilon$", ha="center", fontsize=10.5)
    ax.text(4.7, 0.6, r"$\sigma \sim p(\sigma)$, $\epsilon \sim \mathcal{N}(0, I)$",
            ha="center", va="top", fontsize=8.5, color=LINE)

    arrow(ax, (6.45, 2.0), (7.15, 2.0))
    box(ax, 7.2, 1.25, 2.5, 1.5, "denoiser\n" r"$D_\theta(x_\sigma, \sigma)$" "\nSongUNet",
        PRIOR_BG, PRIOR)
    arrow(ax, (9.75, 2.0), (10.35, 2.0))
    tile(ax, denoise(x0 + 0.8 * eps, 0.8), 10.45, 0.9, 2.2, r"estimate $\hat{x}_0$")
    arrow(ax, (12.8, 2.0), (13.2, 2.0))
    box(ax, 13.25, 1.4, 1.6, 1.2, "loss\n" r"$\|\hat{x}_0 - x_0\|^2$", "white", LINE, 9.5)

    ax.plot([1.5, 1.5, 14.05], [3.5, 4.2, 4.2], color=LINE, lw=1.3, ls=(0, (4, 3)))
    arrow(ax, (14.05, 4.2), (14.05, 2.65), dashed=True, color=LINE,
          label="target", label_xy=(8.0, 4.38))
    arrow(ax, (14.05, 1.35), (8.45, 1.2), rad=-0.3, dashed=True, color=PRIOR,
          label=r"update $\theta$", label_xy=(11.6, 0.2))
    fig.savefig(OUT / "a_training.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- b) and c) strips


def fig_strip(frames, name, letter, title, truth=None, obs=None, mask=None):
    idx = np.linspace(0, len(frames) - 1, 6).round().astype(int)
    extra = 1 if obs is not None else 0
    ncol = len(idx) + extra
    fig, axes = plt.subplots(2, ncol, figsize=(1.75 * ncol + 0.9, 4.4),
                             gridspec_kw=dict(wspace=0.06, hspace=0.2))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.74, bottom=0.03)
    if obs is not None:  # gap after the observation column for the row labels
        for a in axes[:, 0]:
            p = a.get_position()
            a.set_position([p.x0 - 0.035, p.y0, p.width, p.height])
    for a in axes.flat:
        a.set_xticks([])
        a.set_yticks([])
    if obs is not None:
        axes[0, 0].imshow(obs, origin="lower", cmap=CMAP, vmin=-VLIM, vmax=VLIM)
        axes[0, 0].set_title("observations $y$", fontsize=9.5, color=OBS)
        axes[1, 0].imshow(truth, origin="lower", cmap=CMAP, vmin=-VLIM, vmax=VLIM)
        axes[1, 0].set_title("truth (hidden)", fontsize=9.5, color=LINE)
        for a in axes[:, 0]:
            for sp in a.spines.values():
                sp.set_color(OBS)
    for c, i in enumerate(idx, start=extra):
        s, x, xhat = frames[i]
        axes[0, c].imshow(shown(x, s), origin="lower", cmap=CMAP, vmin=-VLIM, vmax=VLIM)
        axes[1, c].imshow(xhat, origin="lower", cmap=CMAP, vmin=-VLIM, vmax=VLIM)
        if mask is not None:
            axes[1, c].contour(mask.astype(float), [0.5], colors=[INK], linewidths=0.4, alpha=0.5)
        axes[0, c].set_title(rf"$\sigma = {s:.2g}$", fontsize=9.5)
    axes[0, extra].set_ylabel("noisy state $x_\\tau$", fontsize=10)
    axes[1, extra].set_ylabel(r"estimate $\hat{x}_0$", fontsize=10)
    if extra:
        axes[0, 0].set_ylabel(" ")

    x0 = axes[0, extra].get_position().x0
    x1 = axes[0, -1].get_position().x1
    fig.text(0.01, 0.97, f"{letter})", fontsize=13, fontweight="bold", va="top")
    fig.text(0.045, 0.965, title, fontsize=12, va="top")
    fig.add_artist(plt.matplotlib.patches.FancyArrowPatch(
        (x0, 0.855), (x1, 0.855), transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=13, lw=1.3, color=PRIOR if obs is None else OBS))
    fig.text((x0 + x1) / 2, 0.87, r"diffusion time $\tau$: 1 $\rightarrow$ 0", ha="center",
             va="bottom", fontsize=9.5)
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)


def gif(frames, name, truth=None, obs=None, mask=None):
    n = 4 if obs is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(2.05 * n, 2.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.12, wspace=0.05)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    kw = dict(origin="lower", cmap=CMAP, vmin=-VLIM, vmax=VLIM)
    s, x, xhat = frames[0]
    a_x, a_hat = axes[-2], axes[-1]
    im_x = a_x.imshow(shown(x, s), **kw)
    im_hat = a_hat.imshow(xhat, **kw)
    a_x.set_title("noisy state $x_\\tau$", fontsize=10)
    a_hat.set_title(r"estimate $\hat{x}_0$", fontsize=10)
    if obs is not None:
        axes[0].imshow(obs, **kw)
        axes[0].set_title("observations $y$", fontsize=10, color=OBS)
        axes[1].imshow(truth, **kw)
        axes[1].set_title("truth (hidden)", fontsize=10, color=LINE)
        a_hat.contour(mask.astype(float), [0.5], colors=[INK], linewidths=0.4, alpha=0.5)
    bar_bg = fig.add_axes([0.25, 0.05, 0.5, 0.025])
    bar_bg.set_xlim(0, 1)
    bar_bg.axis("off")
    bar_bg.add_patch(Rectangle((0, 0), 1, 1, fc="#e6e6e6", ec="none"))
    bar = Rectangle((0, 0), 0, 1, fc=OBS if obs is not None else PRIOR, ec="none")
    bar_bg.add_patch(bar)
    label = fig.text(0.5, 0.92, "", ha="center", fontsize=10)

    hold = 12
    order = list(range(len(frames))) + [len(frames) - 1] * hold

    def update(k):
        s, x, xhat = frames[k]
        im_x.set_data(shown(x, s))
        im_hat.set_data(xhat)
        bar.set_width((k + 1) / len(frames))
        label.set_text(rf"step {k + 1}/{len(frames)}    $\sigma = {s:.3g}$")
        return im_x, im_hat, bar, label

    anim = FuncAnimation(fig, update, frames=order, blit=False)
    anim.save(OUT / f"{name}.gif", writer=PillowWriter(fps=7), dpi=72)
    plt.close(fig)


# ---------------------------------------------------------------- c) SDA loop


def fig_sda_loop(frames, truth, obs, mask):
    s, x, xhat = frames[len(frames) // 3]
    fig, ax = canvas(18, 7.0)
    panel_letter(ax, 0.2, 6.9, "c", "Score-based data assimilation")
    S = 2.0

    # input: pure noise
    group(ax, 0.15, 3.2, 2.5, 2.95, "INPUT")
    tile(ax, shown(frames[0][1], frames[0][0]), 0.4, 3.4, S,
         r"pure noise $x_1 \sim \mathcal{N}(0, \sigma_\mathrm{max}^2 I)$")
    arrow(ax, (2.7, 4.4), (3.2, 4.4))

    # top row: prior prediction, mapped into observation space
    tile(ax, shown(x, s), 3.3, 3.4, S, r"noisy state $x_\tau$")
    arrow(ax, (5.45, 4.4), (6.1, 4.4))
    box(ax, 6.15, 3.75, 2.4, 1.3, "denoiser\n" r"$D_\theta(x_\tau, \tau)$", PRIOR_BG, PRIOR)
    arrow(ax, (8.6, 4.4), (9.25, 4.4))
    tile(ax, denoise(x, s), 9.35, 3.4, S, r"estimate $\hat{x}_0$", below="prior only")
    arrow(ax, (11.5, 4.4), (12.15, 4.4))
    box(ax, 12.2, 3.75, 2.3, 1.3, "observation\noperator $H$", OBS_BG, OBS)
    arrow(ax, (14.55, 4.4), (15.2, 4.4))
    tile(ax, np.ma.masked_where(~mask, denoise(x, s)), 15.3, 3.4, S, r"$H(\hat{x}_0)$")

    # comparison of the generated sample with the observations
    tile(ax, np.ma.masked_where(~mask, obs), 15.3, 0.5, S, below="observations $y$")
    cx, cy, r = 14.5, 1.5, 0.28
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=OBS, lw=1.3, zorder=6))
    ax.text(cx, cy, r"$\Sigma$", ha="center", va="center", fontsize=11, color=OBS, zorder=7)
    arrow(ax, (15.25, cy), (cx + r, cy), color=OBS)
    ax.text(15.0, cy + 0.12, "+", ha="center", va="bottom", fontsize=12, color=OBS)
    arrow(ax, (15.8, 3.3), (cx + 0.12, cy + r - 0.02), color=OBS)
    ax.text(14.95, 2.35, "−", ha="center", va="center", fontsize=13, color=OBS)
    ax.text(cx, cy - r - 0.12, "residual\n" r"$y - H(\hat{x}_0)$", ha="center", va="top",
            fontsize=9, color=OBS)
    arrow(ax, (cx - r, cy), (13.85, cy), color=OBS)
    box(ax, 10.3, 0.85, 3.5, 1.3, "likelihood score (MMPS)\n"
        r"$\nabla_{x_\tau} \log p(y \mid x_\tau)$", OBS_BG, OBS)
    arrow(ax, (10.25, cy), (8.65, cy), color=OBS)
    box(ax, 6.15, 0.85, 2.45, 1.3, "DDIM step\n" r"$x_\tau \rightarrow x_{\tau - \Delta\tau}$",
        "white", LINE)

    # loop back while tau > 0, exit to the output at tau = 0
    jx = 4.3
    ax.plot([6.1, jx], [cy, cy], color=LINE, lw=1.3)
    arrow(ax, (jx, cy), (jx, 3.3))
    ax.text(jx + 0.15, 2.4, "repeat\n" r"while $\tau > 0$", ha="left", va="center",
            fontsize=9, color=LINE)
    arrow(ax, (jx, cy), (2.7, cy))
    ax.text(3.4, cy + 0.12, r"$\tau = 0$", ha="center", va="bottom", fontsize=9, color=LINE)
    group(ax, 0.15, 0.05, 2.5, 2.75, "OUTPUT")
    tile(ax, frames[-1][2], 0.4, 0.5, S, below=r"posterior sample $x_0$")
    ax.add_patch(Rectangle((0.4, 0.5), S, S, fill=False, ec=OBS, lw=1.6, zorder=5))
    fig.savefig(OUT / "c_sda_loop.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_training()

    uncond = ddim()
    fig_strip(uncond, "b_unconditional_strip", "b", "Unconditional denoising: sampling from the prior")
    gif(uncond, "b_unconditional")

    truth = prior_sample()
    mask = swot_mask()
    obs_std = 0.05
    noisy = truth + obs_std * RNG.standard_normal((N, N))
    obs = np.ma.masked_where(~mask, noisy)
    sda = ddim(guide=(mask, noisy[mask], obs_std))
    fig_sda_loop(sda, truth, obs, mask)
    fig_strip(sda, "c_sda_strip", "c", "SDA-steered denoising: sampling conditioned on SWOT swaths",
              truth=truth, obs=obs, mask=mask)
    gif(sda, "c_sda", truth=truth, obs=obs, mask=mask)
    print("wrote", sorted(p.name for p in OUT.glob("*.[pg][ni][gf]")))
