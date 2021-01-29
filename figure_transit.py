import numpy as np
from matplotlib.patches import Circle, Wedge
import matplotlib.pyplot as plt


plt.figure(1, (8,5))
plt.clf()
plt.subplots_adjust(0.01, 0.01, 0.99, 0.99)
ax = plt.subplot(111)
ax.set_aspect('equal')
# Star:
r = 3.0
circle = Circle((0.25-r, 0.5), r, color="gold")
ax.add_artist(circle)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)

# Planet:
r = 0.25
x = 1.0
y = 0.5
circle = Circle((x, y), r, color="k", lw=2)
ax.add_artist(circle)
# Layers:
nlayers = 5
width = 0.07
c = np.linspace(0.4, 0.8, nlayers)
for i in np.arange(nlayers):
  ax.add_artist(Wedge((x, y), r+i*width, 0, 360, width=width, color=str(c[i])))

b = r + width
x0 = x - np.sqrt((r+3*width)**2 - b**2)
x1 = x - np.sqrt((r+2*width)**2 - b**2)

plt.plot([0.24, 2.0], [b+y, b+y], "b", lw=1.0, dashes=(8,1))
plt.plot([x0, x], [b+y, y], "r", lw=1.0)
plt.plot([x1, x], [b+y, y], "r", lw=1.0)
#plt.plot([0.25, 1.95], [y, y], "r", lw=1.0)
plt.plot([x, x], [y, y+r+width], "r", lw=1.5)
#plt.plot([0.4, 0.4], [y, y+r+width], "r", lw=2)
plt.plot([x0, x1], [b+y, b+y], "b", lw=2.2)

ax.text(0.03, 0.65, "Host star", fontsize=14)
ax.text(x+0.005, 0.63, "$b$", color='w', fontsize=15)
ax.text(x-0.23, 0.66, r"$r_{i}$", color='w', fontsize=15)
ax.text(x-0.15, 0.73, r"$r_{i+1}$", color='w', fontsize=15)
ax.text(0.25, y+r+0.09, r"Raypath $(s)$", fontsize=14, color="b")
ax.text(x0+0.02, y+r+0.09, r"$s_i(b)$", fontsize=14, color="b")
ax.text(1.3, y+r+0.1, r"To Earth", fontsize=14)
ax.annotate('', xy=(1.48, y+r+0.08),  xycoords='data',
                xytext=(-70, 0), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"))

plt.xlim(0, 1.5)
plt.ylim(0.4, 1.1)
plt.show()
plt.savefig("plots/transit_geometry.pdf")

