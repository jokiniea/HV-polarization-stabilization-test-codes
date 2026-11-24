import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

t_FPGA_ps = 62500
t_detector_ps = 1e9
t_sweep_ps = 2e9
N_H = 9
N_V = 9
test_repetitions = 20
test_time = 1e10
test_n_data = int(2 * test_repetitions * test_time / t_detector_ps)

if min(t_FPGA_ps, t_detector_ps, t_sweep_ps, N_H, N_V) <= 0:
    raise ValueError("Aika-asetusten ja pyyhkäisypisteiden on oltava positiivisia")

fpga_samples_per_segment = t_sweep_ps / t_FPGA_ps
detector_bins_per_segment = t_sweep_ps / t_detector_ps
if not float(fpga_samples_per_segment).is_integer():
    raise ValueError("t_sweep_ps täytyy olla t_FPGA_ps:n kokonaislukukerrannainen")
if not float(detector_bins_per_segment).is_integer():
    raise ValueError("t_sweep_ps täytyy olla t_detector_ps:n kokonaislukukerrannainen")

fpga_samples_per_segment = int(fpga_samples_per_segment)
detector_bins_per_segment = int(detector_bins_per_segment)
segment_count = N_H * N_V
segment_duration_s = t_sweep_ps * 1e-12
detector_bin_s = t_detector_ps * 1e-12
trigger_num = int(700/1e9*t_detector_ps)


def plot_segment_series(ax, x, y, label, text_y):

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(y) == 0:
        ax.text(0.02, text_y, f"{label}: no data", transform=ax.transAxes)
        return

    ax.plot(x, y, ".", label=label)

    if len(x) > 1 and x[0] != x[-1]:
        fit = np.polyfit(x, y, 1)
        ax.plot(x, np.polyval(fit, x), "--", label=f"{label} fit")

    mean = np.mean(y)
    std = np.std(y)
    ax.text(
        0.02,
        text_y,
        f"{label}: mean={mean:.6g}, std={std:.6g}",
        transform=ax.transAxes,
    )

def extract_triggered(data, n_data, binwidth_ps, trig_row=2):
    """
    Palauttaa H, V ja ikkunan indeksit.
    data: Counterin data, muoto [kanava, bini], viimeinen rivi on triggerikanava
    n_data: odotettu datapisteiden määrä triggerin jälkeen
    offset: montako binia triggerin bini(n) jälkeen data alkaa (0 = samasta binistä)
    """
    if not np.any(data[trig_row] > trigger_num):
        raise RuntimeError("Kanavalla 8 ei havaittu liipaisua")

    trig_idx = int(np.argmax(data[trig_row] > trigger_num))
    first_idx = trig_idx
    last_idx = first_idx + n_data - 1          # viimeinen dataan kuuluva bini (mukaan lukien)

    if last_idx >= data.shape[1]:
        raise RuntimeError(
            f"Kasvata margin_s"
        )

    to_ms = binwidth_ps * 1e-12 * 1e3
    print(f"Liipaisu: bini {trig_idx} ({trig_idx * to_ms:.1f} ms)")
    print(f"Ensimmäinen dataan kuuluva indeksi: {first_idx}")
    print(f"Viimeinen dataan kuuluva indeksi:   {last_idx}")
    print(f"Datapisteitä: {n_data}, ikkuna {first_idx * to_ms:.1f}...{(last_idx + 1) * to_ms:.1f} ms")

    tail = data[:trig_row, last_idx + 1:]
    n_tail_bins = tail.shape[1]
    n_tail_counts = int(tail.sum())
    print(f"Ikkunan jälkeen {n_tail_bins} binia, joissa yhteensä {n_tail_counts} tapahtumaa (hylätty)")

    H = data[0, first_idx:last_idx + 1].astype(float)
    V = data[1, first_idx:last_idx + 1].astype(float)
    return H, V, first_idx, last_idx


sweep_path = Path("HV_stablization_code_test_sweep_data.csv")
with sweep_path.open(encoding="utf-8") as csv_file:
    columns = csv_file.readline().strip().split(",")

data = np.loadtxt(sweep_path, delimiter=",", skiprows=1)

output_dir = Path(__file__).with_name("segment_plots")
output_dir.mkdir(parents=True, exist_ok=True)

raw_data_path = Path("HV_stabilization_code_test_data.csv")
with raw_data_path.open(encoding="utf-8") as csv_file:
    raw_columns = csv_file.readline().strip().split(",")
raw_data = np.loadtxt(raw_data_path, delimiter=",", skiprows=1)
raw_t = raw_data[:, 0]
raw_channel_1 = raw_data[:, 1]
raw_channel_2 = raw_data[:, 2]
raw_trigger = raw_data[:, 3]
raw_hv_sum = raw_channel_1 + raw_channel_2
raw_hv_sum[raw_hv_sum == 0] = np.nan

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(raw_t, raw_channel_1 / raw_hv_sum, "-", label=f"H normalized")
ax.plot(raw_t, raw_channel_2 / raw_hv_sum, "-", label=f"V normalized")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized counts")
ax.set_title("Raw stabilization data")
ax.grid(True)
trigger_ax = ax.twinx()
trigger_ax.patch.set_visible(False)
trigger_ax.plot(
    raw_t,
    raw_trigger,
    color="red",
    linewidth=1.5,
    label=raw_columns[3],
    zorder=10,
)
trigger_ax.set_ylabel("Trigger counts")
trigger_ax.set_zorder(ax.get_zorder() + 1)
ax.legend(loc="upper left")
trigger_ax.legend(loc="upper right")
fig.tight_layout()
plt.show(block=True)
fig.savefig(output_dir / "raw_data.png", dpi=150)
plt.close(fig)

full_H = data[:, columns.index("H")].astype(float)
full_V = data[:, columns.index("V")].astype(float)
trigger_counts = data[:, 3]
sweep_t = data[:, 0]

full_hv_sum = full_H + full_V
full_hv_sum[full_hv_sum == 0] = np.nan
full_H_norm = full_H / full_hv_sum
full_V_norm = full_V / full_hv_sum

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(sweep_t, full_H_norm,"-", label="H/(H+V)",markersize=4)
ax.plot(sweep_t, full_V_norm,"-", label="V/(H+V)",markersize=4)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized value")
ax.set_title("Normalized polarization: full measurement")
ax.set_ylim(0, 1)
trigger_ax = ax.twinx()
trigger_ax.patch.set_visible(False)
trigger_ax.plot(
    sweep_t,
    trigger_counts,
    color="red",
    linewidth=1.5,
    label="Trigger counts",
    zorder=10,
)
trigger_ax.set_ylabel("Trigger counts")
trigger_ax.set_zorder(ax.get_zorder() + 1)
ax.legend()
trigger_ax.legend(loc="upper right")
ax.grid(True)
fig.tight_layout()
plt.show(block=True)
fig.savefig(output_dir / "full_data_normalized.png", dpi=150)
plt.close(fig)

sweep_values_h = np.linspace(-12000, 12000, N_H, dtype=np.int16)
sweep_values_v = np.linspace(-12000, 12000, N_V, dtype=np.int16)
sweep_values = list(product(sweep_values_h, sweep_values_v))
n_data = segment_count * detector_bins_per_segment

channel_data = np.vstack((full_H, full_V, trigger_counts))
H, V, first_idx, last_idx = extract_triggered(
    channel_data, n_data, t_detector_ps
)
hv_sum = H + V
hv_sum[hv_sum == 0] = np.nan
H_norm = H / hv_sum
V_norm = V / hv_sum
t = sweep_t[first_idx:last_idx + 1] - sweep_t[first_idx]

if len(sweep_values) != segment_count:
    raise RuntimeError("Sweep-arvojen määrä ei vastaa N_H * N_V -arvoa")

for i in range(segment_count):
    start = i * detector_bins_per_segment
    stop = start + detector_bins_per_segment
    segment_time = t[start:stop]
    H_segment = H[start:stop]
    V_segment = V[start:stop]
    H_norm_segment = H_norm[start:stop]
    V_norm_segment = V_norm[start:stop]
    HV_sum_segment = H_segment + V_segment

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    plot_segment_series(axes[0], segment_time, H_segment, "H", 0.95)
    plot_segment_series(axes[0], segment_time, V_segment, "V", 0.82)
    axes[0].set_ylabel("Counts")
    axes[0].set_title("H and V counts")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    plot_segment_series(axes[1], segment_time, H_norm_segment, "H/(H+V)", 0.95)
    plot_segment_series(axes[1], segment_time, V_norm_segment, "V/(H+V)", 0.82)
    axes[1].set_ylabel("Normalized value")
    axes[1].set_title("Normalized polarization")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    plot_segment_series(axes[2], segment_time, HV_sum_segment, "H+V", 0.95)
    axes[2].set_ylabel("Counts")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Total counts")
    axes[2].legend(loc="best")
    axes[2].grid(True)
    axes[2].set_xlim(i * segment_duration_s, (i + 1) * segment_duration_s)

    fig.suptitle(f"Segment {i + 1}/{segment_count}, sym: {sweep_values[i]}")
    fig.tight_layout()
    fig.savefig(output_dir / f"segment_{i + 1:03d}.png", dpi=150)
    plt.close(fig)

for old_plot in output_dir.glob("segment_*.png"):
    segment_number = old_plot.stem.partition("_")[2]
    if segment_number.isdigit() and int(segment_number) > segment_count:
        old_plot.unlink()

print(
    f"Segmentin kesto: {segment_duration_s:g} s "
    f"({detector_bins_per_segment} detector-biniä, "
    f"{fpga_samples_per_segment} FPGA-näytettä)"
)
print(f"Saved {segment_count} segment plots to {output_dir}")

data = np.loadtxt(
    "HV_stabilization_code_test_data.csv",
    delimiter=",",
    skiprows=1
)

n_data = test_n_data
channel_data = data[:, 1:4].T
H, V, first_idx, last_idx = extract_triggered(
    channel_data, n_data, t_detector_ps, trig_row=2
)
t = data[first_idx:last_idx + 1, 0]
hv_sum = H + V
hv_sum[hv_sum == 0] = np.nan
H_norm = H / hv_sum 
V_norm = V / hv_sum

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, H_norm, ".", label="H", markersize=4)
ax.plot(t, V_norm, ".", label="V", markersize=4)

for line_time in np.arange(t[0], t[-1], detector_bin_s):
    ax.axvline(line_time, color="gray", linewidth=0.2, alpha=0.5)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Counts")
ax.set_title("Test data: H and V")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(output_dir / "test_optimal_S_values.png", dpi=150)
plt.close(fig)
