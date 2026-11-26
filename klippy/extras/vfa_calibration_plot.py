# Графики калибровки VFA. Выполняется в chroot после остановки движений.
# This file may be distributed under the terms of the GNU GPLv3 license.
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HARMONICS = (1, 2, 4)


def plot(folder, names):
    with open(os.path.join(folder, 'calibration.json'), encoding='utf-8') as f:
        meta = json.load(f)
    for name in names:
        files = sorted(glob.glob(os.path.join(folder, name + '_alternative-*Hz.csv')))
        files += sorted(glob.glob(os.path.join(folder, name + '_search_calibrate.csv')))
        for path in files:
            data = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))
            fig, axes = plt.subplots(3, 3, figsize=(12, 9))
            for td, row in zip(HARMONICS, axes):
                for direction, color in ((1, 'tab:blue'), (0, 'tab:orange')):
                    points = data[(data['tdx'] == td) & (data['dir'] == direction)]
                    if not len(points):
                        continue
                    label = 'Forward' if direction else 'Reverse'
                    phase = points[np.isin(points['round'], [1, 2, 4])]
                    for stage, marker in ((1, 'o'), (2, 's'), (4, '^')):
                        subset = phase[phase['round'] == stage]
                        row[0].scatter(np.degrees(subset['phase']) % 360,
                                       subset['residual'], s=14, marker=marker,
                                       color=color, alpha=.7,
                                       label='%s R%d' % (label, stage))
                    amp = points[points['round'] == 3]
                    row[1].scatter(amp['amp'], amp['residual'], color=color,
                                   s=18, label=label)
                    bare = points[points['round'] == 0]
                    if len(bare):
                        row[1].axhline(bare['residual'][0], color=color,
                                       linestyle=':', alpha=.6)
                    stages = []
                    values = []
                    for stage in (0, 5, 6):
                        subset = points[points['round'] == stage]
                        if len(subset):
                            stages.append(stage)
                            values.append(subset['residual'][-1])
                    row[2].plot(stages, values, 'o-', color=color, label=label)
                row[0].set_title('TD%d — phase probes' % td)
                row[0].set_xlabel('Phase, deg')
                row[1].set_title('TD%d — amplitude probes' % td)
                row[1].set_xlabel('Compensation amplitude')
                row[2].set_title('TD%d — measured residual' % td)
                row[2].set_xticks([0, 5, 6], ['Bare', 'Individual', 'Common'])
                for ax in row:
                    ax.set_ylabel('Acceleration, mm/s²')
                    ax.set_ylim(bottom=0)
                    ax.grid(alpha=.25)
                    if ax.get_legend_handles_labels()[0]:
                        ax.legend(fontsize=7)
            fig.suptitle('Z-Mod VFA / T%d / %s' %
                         (meta['tool'], os.path.basename(path)[:-4]))
            fig.text(.5, .015, 'Common = tested candidate, not necessarily saved. '
                     'Actual applied values: calibration.json (MCU units).',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .035, 1, .95))
            fig.savefig(path[:-4] + '.png', dpi=120)
            plt.close(fig)

        sweep = os.path.join(folder, name + '_sweep.csv')
        if not os.path.isfile(sweep):
            continue
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
        checks = sorted(glob.glob(os.path.join(folder, name + '_check-*.csv')))
        before = np.atleast_1d(np.genfromtxt(sweep, delimiter=',', names=True))
        for td, ax in zip(HARMONICS, axes):
            field = 'td%d_amp' % td
            ax.plot(before['vel'], before[field], 'k.-', label='Bare / forward')
            for path in checks:
                data = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))
                label = os.path.basename(path)[len(name) + 7:-4]
                ax.plot(data['vel'], data[field],
                        '--' if label.endswith('dir0') else '-', label=label)
            ax.set_title('TD%d' % td)
            ax.set_ylabel('Acceleration, mm/s²')
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel('Toolhead speed, mm/s')
        fig.suptitle('Z-Mod VFA / T%d / %s / resonance sweep' %
                     (meta['tool'], name))
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(sweep[:-4] + '.png', dpi=120)
        plt.close(fig)


if __name__ == '__main__':
    plot(sys.argv[1], sys.argv[2:])
