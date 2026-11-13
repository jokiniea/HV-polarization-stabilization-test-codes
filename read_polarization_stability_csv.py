import numpy as np
import matplotlib.pyplot as plt

# Lue data
data = np.loadtxt(
    "HV_modulator_polarization_stability_3.csv",
    delimiter=",",
    skiprows=1
)

# Sarakkeet
t = data[:, 0]
H = data[:, 1]
V = data[:, 2]
H_over_HplusV = data[:, 3]
V_over_HplusV = data[:, 4]

# Trendiviivat
H_fit = np.polyfit(t, H, 1)
V_fit = np.polyfit(t, V, 1)

H_trend = np.polyval(H_fit, t)
V_trend = np.polyval(V_fit, t)

# Trendit H/(H+V) ja V/(H+V)
mask = np.isfinite(H_over_HplusV)
Hnorm_fit = np.polyfit(t[mask],H_over_HplusV[mask],1)
Hnorm_trend = np.polyval(Hnorm_fit, t)

mask = np.isfinite(V_over_HplusV)
Vnorm_fit = np.polyfit(t[mask],V_over_HplusV[mask],1)
Vnorm_trend = np.polyval(Vnorm_fit, t)

print(Hnorm_trend.min(), Hnorm_trend.max())

# Trendi H+V
HV_sum = H + V
HVsum_fit = np.polyfit(t, HV_sum, 1)
HVsum_trend = np.polyval(HVsum_fit, t)

# Piirrä
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# H ja V
# Kuvaaja 1
ax[0].plot(t, H, label="H")
ax[0].plot(t, H_trend, "--", label="H trend")

ax[0].plot(t, V, label="V")
ax[0].plot(t, V_trend, "--", label="V trend")
ax[0].legend()

# Kuvaaja 2
ax[1].plot(t, H_over_HplusV, label="H/(H+V)")
ax[1].plot(t, Hnorm_trend, "--", label="H/(H+V) trend")

ax[1].plot(t, V_over_HplusV, label="V/(H+V)")
ax[1].plot(t, Vnorm_trend, "--", label="V/(H+V) trend")
ax[1].legend()

# Kuvaaja 3
ax[2].plot(t, HV_sum, label="H+V")
ax[2].plot(t, HVsum_trend, "--", label="H+V trend")
ax[2].legend()

plt.tight_layout()
plt.show()

mask = np.isfinite(H_over_HplusV)
t = t[mask]
H_over_HplusV= H_over_HplusV[mask]

for T in [0.1, 0.2, 0.5, 1.0]:
    mask = t <= T

    x = H_over_HplusV[mask]

    mean = np.mean(x)
    stderr = np.std(x) / np.sqrt(len(x))

    print(
        f"{T:4.1f} s : {mean:.5f} ± {stderr:.5f}"
    )