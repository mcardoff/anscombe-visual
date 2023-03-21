"""Visualize the Anscombe Quartet."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def _line(x, m, b):
    return m*x + b


def r_square(ydata, fdata):
    """Return the r^2 coefficient."""
    residuals = ydata - fdata
    ssr = np.sum(residuals**2)
    sst = np.sum((ydata-np.mean(ydata))**2)
    return (1-(ssr/sst))


def main():
    """Import data, visualize using pandas."""
    # import data
    data = pd.read_csv('data.csv')

    # fit data
    p1, e1 = curve_fit(_line, data['x1'], data['y1'])
    p2, e2 = curve_fit(_line, data['x2'], data['y2'])
    p3, e3 = curve_fit(_line, data['x3'], data['y3'])
    p4, e4 = curve_fit(_line, data['x4'], data['y4'])

    # add to dataframe
    data['y1line'] = _line(data['x1'], p1[0], p1[1])
    data['y2line'] = _line(data['x2'], p2[0], p2[1])
    data['y3line'] = _line(data['x3'], p3[0], p3[1])
    data['y4line'] = _line(data['x4'], p4[0], p4[1])

    # calculate coefficient of determination
    r1 = r_square(data['y1'], data['y1line'])
    r2 = r_square(data['y2'], data['y2line'])
    r3 = r_square(data['y3'], data['y3line'])
    r4 = r_square(data['y4'], data['y4line'])

    # set up figure
    fig, axes = plt.subplots(nrows=2, ncols=2)

    # Raw Data
    data.plot(x='x1', y='y1', kind='scatter', ax=axes[0, 0], color='red')
    data.plot(x='x2', y='y2', kind='scatter', ax=axes[0, 1], color='green')
    data.plot(x='x3', y='y3', kind='scatter', ax=axes[1, 0], color='blue')
    data.plot(x='x4', y='y4', kind='scatter', ax=axes[1, 1], color='orange')

    # Fits
    data.plot(x='x1', y='y1line', kind='line', ax=axes[0, 0], color='red')
    data.plot(x='x2', y='y2line', kind='line', ax=axes[0, 1], color='green')
    data.plot(x='x3', y='y3line', kind='line', ax=axes[1, 0], color='blue')
    data.plot(x='x4', y='y4line', kind='line', ax=axes[1, 1], color='orange')

    print(f"Set 1 params: {p1}, r**2: {r1}")
    print(f"Set 2 params: {p2}, r**2: {r2}")
    print(f"Set 3 params: {p3}, r**2: {r3}")
    print(f"Set 4 params: {p4}, r**2: {r4}")

    # add text to plots
    axes[0, 0].text(6, 4, f"p: [{p1[0]:0.2f}, {p1[1]:0.2f}], r2: {r1:0.2f}")
    axes[0, 1].text(6, 3, f"p: [{p2[0]:0.2f}, {p2[1]:0.2f}], r2: {r2:0.2f}")
    axes[1, 0].text(6, 5, f"p: [{p3[0]:0.2f}, {p3[1]:0.2f}], r2: {r3:0.2f}")
    axes[1, 1].text(9, 5, f"p: [{p4[0]:0.2f}, {p4[1]:0.2f}], r2: {r4:0.2f}")

    # plt.savefig("~/school/SP23/PHYS019b/figs/anscombe.png")
    plt.show()


if __name__ == "__main__":
    main()
