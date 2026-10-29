from matplotlib import pyplot as plt
import numpy as np
from time import sleep

from Swabian import TimeTagger

# Yhteys TimeTaggeriin
tagger = TimeTagger.createTimeTaggerNetwork('130.230.124.213')
tagger.setInputDelay(1, 0)
tagger.setInputDelay(2, 0)

# 1000 ns binit
counter = TimeTagger.Counter(tagger=tagger,
    channels=[1, 2],  # H, V
    binwidth=int(1e10),
    n_values=int(1e4)
)
sleep(100)
# Hae data
data = counter.getData()

# Kanavat
H = data[0].astype(float)
V = data[1].astype(float)

hv_sum = H + V

hv_sum[hv_sum == 0] = np.nan

# Stokes-parametrit
S1 = (H - V) / hv_sum 

print(np.mean(H+V))

# Aika-akseli sekunteina
t = counter.getIndex() / 1e12
t = t - t[0]

# Tallenna data CSV-tiedostoon
output = np.column_stack([
    t,
    H,
    V,
    H / hv_sum,
    V / hv_sum,
    S1
])

np.savetxt(
    "HV_modulator_polarization_stability_3.csv",
    output,
    delimiter=",",
    header="time_s,H,V,H_over_HplusV,V_over_HplusV,S1",
    comments=""
)

print("Data tallennettu tiedostoon polarization_stability.csv")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(t, H, label="H")
ax.plot(t, V, label="V")
ax.plot(t, H/hv_sum, label="H/H+V")
ax.plot(t, V/hv_sum, label="V/H+V")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized polarization")
ax.set_title("Polarization stability")
ax.grid(True)
ax.legend()

plt.show()