import numpy as np 
import pandas as pd
import string
from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection

def plot_corr_ellipses(data, ax=None, **kwargs):

    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    #print w , h , a 
    print M

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec


'''
letters = string.uppercase[0:10]
df = pd.DataFrame(dict(( (k, np.random.random(10)+ord(k)-65) for k in letters)))
data = df.corr()

#print data 

fig, ax = plt.subplots(1, 1)
#m = plot_corr_ellipses(data, ax=ax, cmap='Greens')
#m = plot_corr_ellipses(data, ax=ax, cmap='seismic', clim=[-1, 1])
m = plot_corr_ellipses(data, ax=ax, cmap='Greys')#, clim=[-1, 1])

cb = fig.colorbar(m)
cb.set_label('Correlation coefficient')
ax.margins(0.1)

plt.show()
'''
