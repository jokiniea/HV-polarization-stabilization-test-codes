from matplotlib import pyplot as plt
import numpy as np

from Swabian import TimeTagger

# Yhteys TimeTaggeriin
tagger = TimeTagger.createTimeTaggerNetwork('130.230.124.213')

# 100 ns binit
counter = TimeTagger.Counter(tagger=tagger,
    channels=[1, 3, 7, 8],  # H, V, A, D
    binwidth=int(1e5),
    n_values=int(5e7)
)

# Hae data
data = counter.getData()

# Kanavat
H = data[0].astype(float)
V = data[1].astype(float)
A = data[2].astype(float)
D = data[3].astype(float)

hv_sum = H + V
ad_sum = A + D

hv_sum[hv_sum == 0] = np.nan
ad_sum[ad_sum == 0] = np.nan

# Normalisoidut todennakoisyydet
PH = H / hv_sum
PV = V / hv_sum

PA = A / ad_sum
PD = D / ad_sum

# Stokes-parametrit
S1 = PH - PV
S2 = PA - PD

# Aika-akseli sekunteina
t = counter.getIndex() / 1e12

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(t, S1, label="S1 = P(H)-P(V)")
ax.plot(t, S2, label="S2 = P(A)-P(D)")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized polarization")
ax.set_title("Polarization stability")

ax.grid(True)
ax.legend()

plt.show()