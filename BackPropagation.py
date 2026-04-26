"""
Backpropagation Visualizer
==========================
Interactive step-by-step visualization of forward pass and backpropagation
on a simple XOR dataset using a 2-2-1 neural network.

Requirements:
    pip install numpy matplotlib

Usage:
    python backprop_visualizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.widgets import Button, Slider
import warnings
warnings.filterwarnings("ignore")

# ─── Colour Palette ────────────────────────────────────────────────────────────
BG          = "#0d1117"
PANEL       = "#161b22"
ACCENT1     = "#58a6ff"   # blue  – forward pass
ACCENT2     = "#f78166"   # red   – backward pass / gradient
ACCENT3     = "#3fb950"   # green – weight update
ACCENT4     = "#d2a8ff"   # purple – bias / activation
TEXT        = "#e6edf3"
TEXT_DIM    = "#8b949e"
BORDER      = "#30363d"
NEURON_BG   = "#21262d"


# ─── Sigmoid helpers ───────────────────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


# ─── Neural Network ────────────────────────────────────────────────────────────
class NeuralNet:
    def __init__(self):
        np.random.seed(42)
        self.W1 = np.array([[0.15, 0.25], [0.20, 0.30]])   # (2,2)
        self.b1 = np.array([[0.35, 0.35]])
        self.W2 = np.array([[0.40], [0.45]])                 # (2,1)
        self.b2 = np.array([[0.60]])
        self.lr = 0.5
        self.history = []

    def forward(self, X):
        self.X  = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, y):
        m   = y.shape[0]
        dL_da2 = -(y / (self.a2 + 1e-8) - (1 - y) / (1 - self.a2 + 1e-8))
        da2_dz2 = sigmoid_deriv(self.z2)
        self.delta2 = dL_da2 * da2_dz2

        self.dW2 = self.a1.T @ self.delta2 / m
        self.db2 = np.mean(self.delta2, axis=0, keepdims=True)

        da1_dz1 = sigmoid_deriv(self.z1)
        self.delta1 = (self.delta2 @ self.W2.T) * da1_dz1

        self.dW1 = self.X.T @ self.delta1 / m
        self.db1 = np.mean(self.delta1, axis=0, keepdims=True)

    def update(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

    def loss(self, y):
        eps = 1e-8
        return -np.mean(y * np.log(self.a2 + eps) + (1 - y) * np.log(1 - self.a2 + eps))

    def train_epoch(self, X, y):
        out  = self.forward(X)
        l    = self.loss(y)
        self.backward(y)
        self.update()
        self.history.append(l)
        return l


# ─── Dataset ───────────────────────────────────────────────────────────────────
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)   # XOR


# ─── Positions ─────────────────────────────────────────────────────────────────
# layer x-coords
LX = [0.12, 0.45, 0.78]
# neuron y-coords  [input layer, hidden layer, output layer]
LY = {
    "input":  [0.72, 0.28],
    "hidden": [0.72, 0.28],
    "output": [0.50],
}
NEURON_R = 0.055


# ─── Drawing helpers ───────────────────────────────────────────────────────────
def neuron_circle(ax, cx, cy, r, fill, edge, lw=2.5, alpha=1.0, zorder=4):
    c = Circle((cx, cy), r, fc=fill, ec=edge, lw=lw, alpha=alpha,
               transform=ax.transData, zorder=zorder)
    ax.add_patch(c)
    return c


def draw_arrow(ax, x0, y0, x1, y1, color, lw=1.8, alpha=0.8, zorder=2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>",
                                color=color,
                                lw=lw,
                                mutation_scale=12),
                alpha=alpha, zorder=zorder)


def draw_label(ax, x, y, txt, color=TEXT, fs=8, ha="center", va="center", zorder=6):
    ax.text(x, y, txt, color=color, fontsize=fs, ha=ha, va=va,
            fontfamily="monospace", zorder=zorder,
            bbox=dict(boxstyle="round,pad=0.15", fc=PANEL, ec="none", alpha=0.8))


# ─── Main visualizer class ─────────────────────────────────────────────────────
class BackpropVisualizer:
    def __init__(self):
        self.net   = NeuralNet()
        self.step  = 0          # 0=init, 1=forward, 2=loss, 3=backward, 4=update
        self.epoch = 0
        self.sample_idx = 0    # which XOR sample to highlight
        self.max_epochs = 200

        # Pre-train silently for loss curve
        self._pretrained_loss = []
        _net2 = NeuralNet()
        for _ in range(self.max_epochs):
            _net2.train_epoch(X, y)
            self._pretrained_loss.append(_net2.history[-1])

        self._build_figure()
        self._render()
        plt.show()

    # ── Layout ──────────────────────────────────────────────────────────────────
    def _build_figure(self):
        plt.rcParams.update({
            "figure.facecolor":  BG,
            "axes.facecolor":    PANEL,
            "axes.edgecolor":    BORDER,
            "text.color":        TEXT,
            "xtick.color":       TEXT_DIM,
            "ytick.color":       TEXT_DIM,
            "font.family":       "monospace",
        })

        self.fig = plt.figure(figsize=(16, 9), facecolor=BG)
        self.fig.canvas.manager.set_window_title("Backpropagation Visualizer")

        gs = gridspec.GridSpec(3, 3, figure=self.fig,
                               left=0.04, right=0.98,
                               top=0.93, bottom=0.12,
                               hspace=0.45, wspace=0.35)

        self.ax_net   = self.fig.add_subplot(gs[:, 0])          # network diagram
        self.ax_loss  = self.fig.add_subplot(gs[0, 1])           # loss curve
        self.ax_grad  = self.fig.add_subplot(gs[1, 1])           # gradient bar chart
        self.ax_data  = self.fig.add_subplot(gs[2, 1])           # dataset table
        self.ax_eq    = self.fig.add_subplot(gs[:, 2])           # equations panel

        for ax in [self.ax_net, self.ax_loss, self.ax_grad,
                   self.ax_data, self.ax_eq]:
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)

        self.ax_net.set_xlim(0, 1); self.ax_net.set_ylim(0, 1)
        self.ax_net.set_aspect("equal"); self.ax_net.axis("off")

        self.ax_eq.axis("off")

        # Title
        self.fig.text(0.5, 0.97,
                      "⚡  Backpropagation Visualizer  —  XOR Dataset",
                      ha="center", va="top", fontsize=15,
                      color=TEXT, fontweight="bold", fontfamily="monospace")

        # Buttons
        btn_color = PANEL
        self.btn_fwd  = Button(self.fig.add_axes([0.12, 0.03, 0.14, 0.055]),
                               "▶  Forward",  color=btn_color, hovercolor=BORDER)
        self.btn_back = Button(self.fig.add_axes([0.28, 0.03, 0.14, 0.055]),
                               "◀  Backward", color=btn_color, hovercolor=BORDER)
        self.btn_upd  = Button(self.fig.add_axes([0.44, 0.03, 0.14, 0.055]),
                               "⟳  Update",   color=btn_color, hovercolor=BORDER)
        self.btn_auto = Button(self.fig.add_axes([0.60, 0.03, 0.14, 0.055]),
                               "⏩  Auto-Train", color=btn_color, hovercolor=BORDER)
        self.btn_rst  = Button(self.fig.add_axes([0.76, 0.03, 0.10, 0.055]),
                               "↺  Reset",    color=btn_color, hovercolor=BORDER)

        for btn in [self.btn_fwd, self.btn_back, self.btn_upd,
                    self.btn_auto, self.btn_rst]:
            btn.label.set_color(TEXT)
            btn.label.set_fontsize(9)
            btn.label.set_fontfamily("monospace")

        self.btn_fwd.on_clicked(self._on_forward)
        self.btn_back.on_clicked(self._on_backward)
        self.btn_upd.on_clicked(self._on_update)
        self.btn_auto.on_clicked(self._on_auto)
        self.btn_rst.on_clicked(self._on_reset)

        # Sample slider
        self.ax_sl = self.fig.add_axes([0.12, 0.005, 0.38, 0.025])
        self.sl_sample = Slider(self.ax_sl, "Sample", 0, 3,
                                valinit=0, valstep=1,
                                color=ACCENT1, track_color=BORDER)
        self.sl_sample.label.set_color(TEXT)
        self.sl_sample.valtext.set_color(ACCENT1)
        self.sl_sample.on_changed(self._on_sample_change)

    # ── Event callbacks ─────────────────────────────────────────────────────────
    def _on_forward(self, _):
        self.step = 1
        xi = X[self.sample_idx:self.sample_idx+1]
        self.net.forward(xi)
        self._render()

    def _on_backward(self, _):
        if self.step < 1:
            return
        self.step = 2
        yi = y[self.sample_idx:self.sample_idx+1]
        self.net.backward(yi)
        self._render()

    def _on_update(self, _):
        if self.step < 2:
            return
        self.step = 3
        yi = y[self.sample_idx:self.sample_idx+1]
        l = self.net.loss(yi)
        self.net.history.append(l)
        self.net.update()
        self.epoch += 1
        self._render()

    def _on_auto(self, _):
        for _ in range(50):
            self.net.train_epoch(X, y)
            self.epoch += 1
        self.step = 3
        xi = X[self.sample_idx:self.sample_idx+1]
        yi = y[self.sample_idx:self.sample_idx+1]
        self.net.forward(xi)
        self.net.backward(yi)
        self._render()

    def _on_reset(self, _):
        self.net   = NeuralNet()
        self.step  = 0
        self.epoch = 0
        self._render()

    def _on_sample_change(self, val):
        self.sample_idx = int(val)
        if self.step >= 1:
            xi = X[self.sample_idx:self.sample_idx+1]
            self.net.forward(xi)
        if self.step >= 2:
            yi = y[self.sample_idx:self.sample_idx+1]
            self.net.backward(yi)
        self._render()

    # ── Render ──────────────────────────────────────────────────────────────────
    def _render(self):
        self._draw_network()
        self._draw_loss()
        self._draw_gradients()
        self._draw_dataset()
        self._draw_equations()
        self.fig.canvas.draw_idle()

    # ── Network diagram ─────────────────────────────────────────────────────────
    def _draw_network(self):
        ax = self.ax_net
        ax.cla()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.axis("off")

        step   = self.step
        net    = self.net
        xi     = X[self.sample_idx]
        yi     = float(y[self.sample_idx, 0])

        # Layer labels
        for lx, lbl in zip(LX, ["Input\nLayer", "Hidden\nLayer", "Output\nLayer"]):
            ax.text(lx, 0.96, lbl, ha="center", va="top", color=TEXT_DIM,
                    fontsize=7.5, fontfamily="monospace")

        # ── Weights W1 (input→hidden) ────────────────────────────────────────
        w1_max = np.abs(net.W1).max() + 1e-8
        for i, iy in enumerate(LY["input"]):
            for j, hy in enumerate(LY["hidden"]):
                w   = net.W1[i, j]
                col = ACCENT2 if (step == 2 and abs(net.dW1[i,j]) > 0.001) else ACCENT1
                alpha = 0.3 + 0.7 * abs(w) / w1_max
                lw    = 0.8 + 2.5 * abs(w) / w1_max
                draw_arrow(ax, LX[0], iy, LX[1], hy, col, lw=lw, alpha=alpha)
                mx, my = (LX[0]+LX[1])/2, (iy+hy)/2
                ax.text(mx, my + 0.03, f"w={w:.3f}", color=col,
                        fontsize=5.5, ha="center", fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.1", fc=PANEL, ec="none", alpha=0.7))

        # ── Weights W2 (hidden→output) ───────────────────────────────────────
        w2_max = np.abs(net.W2).max() + 1e-8
        for j, hy in enumerate(LY["hidden"]):
            w   = net.W2[j, 0]
            col = ACCENT2 if (step == 2 and abs(net.dW2[j,0]) > 0.001) else ACCENT1
            alpha = 0.3 + 0.7 * abs(w) / w2_max
            lw    = 0.8 + 2.5 * abs(w) / w2_max
            draw_arrow(ax, LX[1], hy, LX[2], LY["output"][0], col, lw=lw, alpha=alpha)
            mx, my = (LX[1]+LX[2])/2, (hy+LY["output"][0])/2
            ax.text(mx, my + 0.03, f"w={w:.3f}", color=col,
                    fontsize=5.5, ha="center", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.1", fc=PANEL, ec="none", alpha=0.7))

        # ── Input neurons ────────────────────────────────────────────────────
        for i, (iy, xval) in enumerate(zip(LY["input"], xi)):
            edge = ACCENT1 if step >= 1 else BORDER
            neuron_circle(ax, LX[0], iy, NEURON_R, NEURON_BG, edge)
            ax.text(LX[0], iy + NEURON_R + 0.03, f"x{i+1}", ha="center",
                    color=TEXT_DIM, fontsize=7, fontfamily="monospace")
            ax.text(LX[0], iy, f"{xval:.0f}", ha="center", va="center",
                    color=ACCENT1, fontsize=10, fontweight="bold",
                    fontfamily="monospace")

        # ── Hidden neurons ───────────────────────────────────────────────────
        for j, hy in enumerate(LY["hidden"]):
            if step >= 1:
                z   = float(net.z1[0, j])
                a   = float(net.a1[0, j])
                edge = ACCENT4
                ax.text(LX[1], hy, f"{a:.3f}", ha="center", va="center",
                        color=ACCENT4, fontsize=7.5, fontweight="bold",
                        fontfamily="monospace")
            else:
                z = a = 0.0
                edge = BORDER
                ax.text(LX[1], hy, "h", ha="center", va="center",
                        color=TEXT_DIM, fontsize=9, fontfamily="monospace")

            neuron_circle(ax, LX[1], hy, NEURON_R, NEURON_BG, edge)
            ax.text(LX[1], hy + NEURON_R + 0.03, f"h{j+1}", ha="center",
                    color=TEXT_DIM, fontsize=7, fontfamily="monospace")
            if step >= 1:
                ax.text(LX[1], hy - NEURON_R - 0.04,
                        f"z={z:.3f}", ha="center", color=TEXT_DIM,
                        fontsize=5.5, fontfamily="monospace")

            # Gradient annotation on backward
            if step >= 2:
                d = float(net.delta1[0, j])
                ax.text(LX[1] - 0.12, hy, f"δ={d:.4f}",
                        color=ACCENT2, fontsize=5.5, fontfamily="monospace",
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc=PANEL, ec=ACCENT2,
                                  lw=0.8, alpha=0.9))

        # ── Output neuron ────────────────────────────────────────────────────
        oy = LY["output"][0]
        if step >= 1:
            pred = float(net.a2[0, 0])
            edge = ACCENT3 if step >= 3 else ACCENT1
            ax.text(LX[2], oy, f"{pred:.3f}", ha="center", va="center",
                    color=edge, fontsize=9, fontweight="bold",
                    fontfamily="monospace")
        else:
            pred = 0.0
            edge = BORDER
            ax.text(LX[2], oy, "ŷ", ha="center", va="center",
                    color=TEXT_DIM, fontsize=10, fontfamily="monospace")

        neuron_circle(ax, LX[2], oy, NEURON_R, NEURON_BG, edge)
        ax.text(LX[2], oy + NEURON_R + 0.03, "ŷ", ha="center",
                color=TEXT_DIM, fontsize=7, fontfamily="monospace")

        if step >= 1:
            loss_val = -( yi * np.log(pred + 1e-8) + (1 - yi) * np.log(1 - pred + 1e-8) )
            ax.text(LX[2], oy - NEURON_R - 0.06,
                    f"y={yi:.0f}  Loss={loss_val:.4f}", ha="center",
                    color=ACCENT2, fontsize=6, fontfamily="monospace")

        if step >= 2:
            d = float(net.delta2[0, 0])
            ax.text(LX[2] + 0.13, oy, f"δ={d:.4f}",
                    color=ACCENT2, fontsize=6, fontfamily="monospace",
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=PANEL, ec=ACCENT2,
                              lw=0.8, alpha=0.9))

        # ── Legend ───────────────────────────────────────────────────────────
        handles = [
            mpatches.Patch(color=ACCENT1, label="Forward pass"),
            mpatches.Patch(color=ACCENT2, label="Gradient / δ"),
            mpatches.Patch(color=ACCENT4, label="Activation"),
            mpatches.Patch(color=ACCENT3, label="Updated weight"),
        ]
        ax.legend(handles=handles, loc="lower center", ncol=2,
                  fontsize=6.5, framealpha=0.5, facecolor=PANEL,
                  edgecolor=BORDER, labelcolor=TEXT)

        # ── Step banner ──────────────────────────────────────────────────────
        banners = {
            0: ("READY",         TEXT_DIM),
            1: ("FORWARD PASS ▶", ACCENT1),
            2: ("BACKWARD PASS ◀", ACCENT2),
            3: ("WEIGHTS UPDATED ✓", ACCENT3),
        }
        banner_txt, banner_col = banners.get(step, ("", TEXT))
        ax.text(0.5, 0.02, banner_txt, ha="center", va="bottom",
                color=banner_col, fontsize=10, fontweight="bold",
                fontfamily="monospace", transform=ax.transAxes)

        ax.set_title(f"Network  |  Epoch {self.epoch}  |  Sample {self.sample_idx+1}/4  "
                     f"→  [{int(xi[0])},{int(xi[1])}]→{int(yi)}",
                     color=TEXT, fontsize=8, pad=4)

    # ── Loss curve ──────────────────────────────────────────────────────────────
    def _draw_loss(self):
        ax = self.ax_loss
        ax.cla()
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        ax.plot(self._pretrained_loss, color=BORDER, lw=1.2,
                linestyle="--", alpha=0.5, label="Full-run reference")

        if self.net.history:
            ax.plot(self.net.history, color=ACCENT1, lw=2, label="Current run")
            ax.scatter([len(self.net.history)-1], [self.net.history[-1]],
                       color=ACCENT3, s=50, zorder=5)
            ax.text(len(self.net.history)-1, self.net.history[-1],
                    f"  {self.net.history[-1]:.4f}",
                    color=ACCENT3, fontsize=7, va="center",
                    fontfamily="monospace")

        ax.set_xlabel("Epoch", color=TEXT_DIM, fontsize=7)
        ax.set_ylabel("BCE Loss", color=TEXT_DIM, fontsize=7)
        ax.set_title("Training Loss", color=TEXT, fontsize=8)
        ax.tick_params(colors=TEXT_DIM, labelsize=6.5)
        ax.legend(fontsize=6.5, facecolor=PANEL, edgecolor=BORDER,
                  labelcolor=TEXT, loc="upper right")
        ax.set_xlim(0, self.max_epochs)
        ax.set_ylim(0, max(self._pretrained_loss[:5]) * 1.05)

    # ── Gradient bar chart ──────────────────────────────────────────────────────
    def _draw_gradients(self):
        ax = self.ax_grad
        ax.cla()
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        if self.step < 2:
            ax.text(0.5, 0.5, "Run Forward + Backward\nto see gradients",
                    ha="center", va="center", color=TEXT_DIM,
                    fontsize=8, fontfamily="monospace", transform=ax.transAxes)
            ax.set_title("Gradients ∂L/∂W", color=TEXT, fontsize=8)
            ax.axis("off")
            return

        net  = self.net
        labels = (["dW1[0,0]","dW1[0,1]","dW1[1,0]","dW1[1,1]"] +
                  ["dW2[0,0]","dW2[1,0]"] +
                  ["db1[0]","db1[1]","db2[0]"])
        values = (list(net.dW1.flatten()) +
                  list(net.dW2.flatten()) +
                  list(net.db1.flatten()) + list(net.db2.flatten()))

        colors = [ACCENT2 if v < 0 else ACCENT1 for v in values]
        bars = ax.barh(labels, values, color=colors, height=0.6, alpha=0.85)

        for bar, v in zip(bars, values):
            ax.text(v + (0.001 if v >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}",
                    ha="left" if v >= 0 else "right",
                    va="center", color=TEXT, fontsize=5.5,
                    fontfamily="monospace")

        ax.axvline(0, color=BORDER, lw=1)
        ax.set_title("Gradients ∂L/∂W  (red=neg, blue=pos)", color=TEXT, fontsize=7.5)
        ax.tick_params(colors=TEXT_DIM, labelsize=6)
        ax.set_xlabel("Gradient value", color=TEXT_DIM, fontsize=7)

    # ── Dataset table ────────────────────────────────────────────────────────────
    def _draw_dataset(self):
        ax = self.ax_data
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.axis("off")

        ax.set_title("XOR Dataset", color=TEXT, fontsize=8, pad=4)

        headers = ["x₁", "x₂", "y (target)"]
        col_w   = [0.25, 0.25, 0.35]
        row_h   = 0.18

        # Header row
        for ci, (h, cw) in enumerate(zip(headers, col_w)):
            ax.text(sum(col_w[:ci]) + cw/2, 0.92, h, ha="center", va="top",
                    color=ACCENT1, fontsize=8, fontweight="bold",
                    fontfamily="monospace", transform=ax.transAxes)

        for ri, (xi, yi) in enumerate(zip(X, y)):
            y_pos = 0.74 - ri * row_h
            bg_col = ACCENT1 + "22" if ri == self.sample_idx else PANEL

            # Row highlight
            ax.add_patch(FancyBboxPatch((0.02, y_pos - 0.06), 0.96, row_h - 0.02,
                                        boxstyle="round,pad=0.01",
                                        fc=bg_col, ec=ACCENT1 if ri == self.sample_idx else BORDER,
                                        lw=1.5 if ri == self.sample_idx else 0.5,
                                        transform=ax.transAxes, zorder=1))
            row_data = [int(xi[0]), int(xi[1]), int(yi[0])]
            for ci, (val, cw) in enumerate(zip(row_data, col_w)):
                col = ACCENT3 if ri == self.sample_idx else TEXT
                ax.text(sum(col_w[:ci]) + cw/2, y_pos + 0.03, str(val),
                        ha="center", va="center", color=col,
                        fontsize=9, fontweight="bold" if ri == self.sample_idx else "normal",
                        fontfamily="monospace", transform=ax.transAxes, zorder=2)

        if self.step >= 1:
            pred = float(self.net.a2[0, 0])
            ax.text(0.5, 0.04, f"Current prediction: {pred:.4f}",
                    ha="center", va="bottom", color=ACCENT4,
                    fontsize=7.5, fontfamily="monospace", transform=ax.transAxes)

    # ── Equations panel ──────────────────────────────────────────────────────────
    def _draw_equations(self):
        ax = self.ax_eq
        ax.cla(); ax.axis("off")

        step = self.step
        net  = self.net

        def box(title, lines, y_start, title_col=ACCENT1):
            ax.text(0.05, y_start, title, color=title_col, fontsize=8.5,
                    fontweight="bold", fontfamily="monospace", transform=ax.transAxes)
            for i, (txt, col) in enumerate(lines):
                ax.text(0.08, y_start - 0.055 - i * 0.055, txt,
                        color=col, fontsize=7.5, fontfamily="monospace",
                        transform=ax.transAxes)

        ax.set_title("Step-by-Step Equations", color=TEXT, fontsize=8, pad=4)

        # ── Always-visible network description ──────────────────────────────
        box("Architecture", [
            ("2 → 2 → 1   (σ activations)", TEXT_DIM),
            ("Loss: Binary Cross-Entropy",   TEXT_DIM),
        ], 0.97)

        # ── Forward pass ────────────────────────────────────────────────────
        if step >= 1:
            z1 = net.z1[0]; a1 = net.a1[0]
            z2 = float(net.z2[0,0]); a2 = float(net.a2[0,0])
            box("Forward Pass ▶", [
                (f"z₁¹ = x·W1 + b1  = [{z1[0]:.3f}, {z1[1]:.3f}]", ACCENT1),
                (f"a₁  = σ(z₁)      = [{a1[0]:.3f}, {a1[1]:.3f}]", ACCENT4),
                (f"z₂  = a₁·W2 + b2 = {z2:.4f}",                   ACCENT1),
                (f"ŷ   = σ(z₂)      = {a2:.4f}",                   ACCENT4),
            ], 0.80, ACCENT1)
        else:
            box("Forward Pass ▶", [
                ("z¹ = X·W1 + b1",   TEXT_DIM),
                ("a¹ = σ(z¹)",       TEXT_DIM),
                ("z² = a¹·W2 + b2",  TEXT_DIM),
                ("ŷ  = σ(z²)",       TEXT_DIM),
            ], 0.80, TEXT_DIM)

        # ── Loss ────────────────────────────────────────────────────────────
        if step >= 1:
            yi   = float(y[self.sample_idx, 0])
            pred = float(net.a2[0, 0])
            l    = -(yi * np.log(pred+1e-8) + (1-yi) * np.log(1-pred+1e-8))
            box("Loss  L", [
                ("L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]", ACCENT2),
                (f"L = {l:.5f}  (y={yi:.0f}, ŷ={pred:.4f})", ACCENT2),
            ], 0.57, ACCENT2)
        else:
            box("Loss  L", [
                ("L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]", TEXT_DIM),
            ], 0.57, TEXT_DIM)

        # ── Backward pass ───────────────────────────────────────────────────
        if step >= 2:
            d2 = float(net.delta2[0,0])
            d1 = net.delta1[0]
            box("Backward Pass ◀", [
                ("δ² = (∂L/∂ŷ)·σ'(z²)",              ACCENT2),
                (f"   = {d2:.5f}",                     ACCENT2),
                ("δ¹ = (δ²·W2ᵀ)·σ'(z¹)",             ACCENT2),
                (f"   = [{d1[0]:.5f}, {d1[1]:.5f}]",  ACCENT2),
                (f"∂L/∂W2 = [{net.dW2[0,0]:.5f},",    TEXT_DIM),
                (f"          {net.dW2[1,0]:.5f}]",      TEXT_DIM),
                (f"∂L/∂W1 = ...",                       TEXT_DIM),
            ], 0.44, ACCENT2)
        else:
            box("Backward Pass ◀", [
                ("δ² = (∂L/∂ŷ)·σ'(z²)",   TEXT_DIM),
                ("δ¹ = (δ²·W2ᵀ)·σ'(z¹)", TEXT_DIM),
                ("∂L/∂W  = δ·aᵀ",         TEXT_DIM),
            ], 0.44, TEXT_DIM)

        # ── Weight update ────────────────────────────────────────────────────
        if step >= 3:
            box("Weight Update ✓", [
                ("W ← W − η · ∂L/∂W",                       ACCENT3),
                (f"η (lr) = {net.lr}",                       ACCENT3),
                (f"W1[0]= [{net.W1[0,0]:.4f}, {net.W1[0,1]:.4f}]", ACCENT3),
                (f"W1[1]= [{net.W1[1,0]:.4f}, {net.W1[1,1]:.4f}]", ACCENT3),
                (f"W2   = [{net.W2[0,0]:.4f}, {net.W2[1,0]:.4f}]", ACCENT3),
            ], 0.20, ACCENT3)
        else:
            box("Weight Update", [
                ("W ← W − η · ∂L/∂W", TEXT_DIM),
                ("η (learning rate) = 0.5", TEXT_DIM),
            ], 0.20, TEXT_DIM)

        # ── Status badge ─────────────────────────────────────────────────────
        status = {0: "⬤  READY",
                  1: "⬤  FORWARD DONE",
                  2: "⬤  GRADIENTS READY",
                  3: "⬤  WEIGHTS UPDATED"}
        scols  = {0: TEXT_DIM, 1: ACCENT1, 2: ACCENT2, 3: ACCENT3}
        ax.text(0.5, 0.03, status.get(step, ""), ha="center", va="bottom",
                color=scols.get(step, TEXT), fontsize=9,
                fontweight="bold", fontfamily="monospace",
                transform=ax.transAxes)


# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   Backpropagation Visualizer")
    print("="*60)
    print("  Dataset : XOR  (4 samples)")
    print("  Network : 2 → 2 → 1  (sigmoid)")
    print()
    print("  Controls:")
    print("    ▶  Forward    – run forward pass on selected sample")
    print("    ◀  Backward   – compute gradients via backprop")
    print("    ⟳  Update     – apply gradient descent step")
    print("    ⏩  Auto-Train – run 50 full epochs automatically")
    print("    ↺  Reset      – restart from random weights")
    print("    [Slider]      – choose which XOR sample to inspect")
    print("="*60 + "\n")
    BackpropVisualizer()