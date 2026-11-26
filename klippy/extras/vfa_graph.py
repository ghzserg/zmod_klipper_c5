# Диагностика VFA для Creator 5 / Creator 5 Pro.
# This file may be distributed under the terms of the GNU GPLv3 license.
import csv
import json
import logging
import math
import os
import subprocess
import sys
import tempfile

HARMONICS = (1, 2, 4)
MOTORS = ('stepper_x', 'stepper_y')
MAX_FREQ = 700.
CAUTION_FREQ = 400.
MAX_SPEED = 400.
MAX_ACCEL = 5000.
BED_MAX = 255.
MARGIN = 10.
SETTLE = 0.05
MIN_DURATION = 0.25
MIN_CYCLES = 20


def speeds_for(gcmd, max_velocity):
    low = gcmd.get_float('MIN_SPEED', 20., minval=10., maxval=MAX_SPEED)
    high = gcmd.get_float('MAX_SPEED', min(MAX_SPEED, max_velocity),
                          minval=10., maxval=min(MAX_SPEED, max_velocity))
    step = gcmd.get_float('STEP', 10., minval=5., maxval=100.)
    if high <= low:
        raise gcmd.error("MAX_SPEED must exceed MIN_SPEED")
    count = int((high - low) / step)
    return [low + i * step for i in range(count + 1)] + (
        [high] if low + count * step < high - 1.e-6 else [])


def spectrum(samples, start, end, vel, accel, distance, freqs, angle,
             nominal_rate):
    import numpy as np
    # Анализируем только равномерное движение, без разгона и торможения.
    ramp = vel / accel
    if abs(end - start - (distance / vel + ramp)) > 0.03:
        return None
    data = np.asarray(samples)
    if data.ndim != 2 or data.shape[1] < 4 or len(data) < 128:
        return None
    lo, hi = start + ramp + SETTLE, end - ramp - SETTLE
    data = data[(data[:, 0] >= lo) & (data[:, 0] <= hi)]
    if len(data) < 128 or not np.isfinite(data).all():
        return None
    duration = data[-1, 0] - data[0, 0]
    dt = np.diff(data[:, 0])
    if duration < MIN_DURATION or np.any(dt <= 0.):
        return None
    fs = (len(data) - 1) / duration
    if (abs(fs - nominal_rate) > .1 * nominal_rate
            or np.max(dt) > 1.5 / fs
            or data[0, 0] - lo > 2. / fs
            or hi - data[-1, 0] > 2. / fs):
        return None
    usable = [i for i, f in enumerate(freqs)
              if MIN_CYCLES <= f * duration
              and f <= min(MAX_FREQ, .45 * fs)]
    result = [(float('nan'),) * 3 for _ in freqs]
    if usable:
        times = data[:, 0] - data[0, 0]
        # Совместная аппроксимация гармоник на известных частотах.
        columns = [np.ones(len(times)), times / duration - .5]
        for i in usable:
            phase = 2. * np.pi * freqs[i] * times
            columns.extend((np.sin(phase), np.cos(phase)))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x, y = data[:, 1], data[:, 2]
        signals = np.column_stack((x * cos_a + y * sin_a,
                                   -x * sin_a + y * cos_a))
        design = np.column_stack(columns)
        coeff, _, rank, _ = np.linalg.lstsq(design, signals, rcond=None)
        if rank != design.shape[1] or not np.isfinite(coeff).all():
            return None
        for k, i in enumerate(usable):
            along, across = np.hypot(coeff[2 + 2*k], coeff[3 + 2*k])
            result[i] = (float(along), float(across), math.hypot(along, across))
    return fs, duration, len(data), result


def sample(tester, gcmd, toolhead, vel, accel, freqs):
    start_xy, end_xy = tester.move
    dx, dy = end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]
    distance = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    missing = (float('nan'), float('nan'), 0,
               [(float('nan'),) * 3 for _ in freqs])
    if distance / vel - vel / accel < 2. * SETTLE + MIN_DURATION + .002:
        return missing
    chip = tester.printer.lookup_object(tester.td_config[tester.stepper]['accel_chip'])
    if not any(f <= min(MAX_FREQ, .45 * chip.data_rate) for f in freqs):
        return missing
    for _ in range(3):
        toolhead.manual_move(start_xy, min(150., vel))
        toolhead.wait_moves()
        client = chip.start_internal_client()
        try:
            toolhead.dwell(.2)
            start = toolhead.get_last_move_time()
            toolhead.manual_move(end_xy, vel)
            end = toolhead.get_last_move_time()
            toolhead.dwell(.2)
        finally:
            try:
                client.finish_measurements()
            finally:
                client.is_finished = True
        values = spectrum(client.get_samples(), start, end, vel, accel,
                          distance, freqs, angle, chip.data_rate)
        if values is not None:
            return values
    gcmd.respond_info("VFA: invalid acceleration samples at %.1f mm/s" % vel)
    return missing


def apply_damping(motors, values):
    for name, motor in motors.items():
        for h in HARMONICS:
            motor.set_resonance_damping(h, *values[name][h])


def restore_damping(motors, values):
    errors = []
    for name, motor in motors.items():
        for h in HARMONICS:
            try:
                motor.set_resonance_damping(h, *values[name][h])
            except Exception as exc:
                logging.exception("VFA: failed to restore %s TD%d", name, h)
                errors.append(exc)
    if errors:
        raise errors[0]


def measure(tester, gcmd):
    printer = tester.printer
    reactor = printer.get_reactor()
    now = reactor.monotonic()
    toolhead = printer.lookup_object('toolhead')
    status = toolhead.get_status(now)
    stats = printer.lookup_object('print_stats', None)
    pause = printer.lookup_object('pause_resume', None)
    if (stats is None or stats.get_status(now)['state'] in ('printing', 'paused')
            or (pause is not None and pause.get_status(now)['is_paused'])):
        raise gcmd.error("VFA_GRAPH is unavailable during printing or pause")
    t = gcmd.get_int('T', 0, minval=0, maxval=3)
    axis = gcmd.get('AXIS', 'XY').upper()
    if axis not in ('X', 'Y', 'XY'):
        raise gcmd.error("AXIS must be X, Y or XY")
    speeds = speeds_for(gcmd, status['max_velocity'])
    repeat = gcmd.get_int('REPEAT', 3, minval=1, maxval=10)
    color = printer.lookup_object('zmod_color', None)
    if color is None or color._get_active_extruder(gcmd) == -2:
        raise gcmd.error("VFA: invalid tool sensor state")
    absolute = printer.lookup_object('gcode_move').get_status(now)['absolute_coordinates']
    selected = False
    try:
        tester.gcode.run_script_from_command("G90\n_T_IN T=%d" % t)
        selected = True
        tester.gcode.run_script_from_command(
            "_SET_GCODE_OFFSET_FAST X=0 Y=0 Z=0 FROM=VFA_GRAPH\nM400")
        current = toolhead.get_status(reactor.monotonic())
        if not all(a in current['homed_axes'] for a in 'xyz'):
            raise gcmd.error("VFA: XYZ not homed after tool selection")
        if current['position'][2] < 100.:
            toolhead.manual_move([None, None, 100.], 20.)
        toolhead.wait_moves()
        # _T_IN меняет ускорение; тест использует исходный лимит.
        toolhead.set_max_velocities(None, status['max_accel'], None,
                                    status['minimum_cruise_ratio'])
        folder, names = measure_selected(tester, gcmd, t, axis, speeds, repeat)
    finally:
        try:
            if selected and not printer.is_shutdown():
                current = toolhead.get_status(reactor.monotonic())
                if all(a in current['homed_axes'] for a in 'xyz'):
                    tester.gcode.run_script_from_command("_T_OUT NO_Z=0\nM400")
        finally:
            toolhead.set_max_velocities(None, status['max_accel'], None,
                                        status['minimum_cruise_ratio'])
            if not absolute and not printer.is_shutdown():
                tester.gcode.run_script_from_command("G91")
    render(tester, gcmd, folder, names)


def measure_selected(tester, gcmd, t, axis, speeds, repeat):
    printer = tester.printer
    now = printer.get_reactor().monotonic()
    toolhead = printer.lookup_object('toolhead')
    status = toolhead.get_status(now)
    attached = []
    undocked = []
    for i in range(1, 5):
        grab = printer.lookup_object('gcode_button extruder_grab%d' % i, None)
        dock = printer.lookup_object('gcode_button extruder_pos%d' % i, None)
        if grab is None or dock is None:
            raise gcmd.error("VFA_GRAPH requires Creator 5 tool sensors")
        if grab.get_status(now)['state'] == 'PRESSED':
            attached.append(i)
        if dock.get_status(now)['state'] == 'RELEASED':
            undocked.append(i)
    if attached != [t + 1] or attached != undocked:
        raise gcmd.error("VFA: requested tool T%d is not securely attached" % t)
    motors = {name: printer.lookup_object('mclib ' + name) for name in MOTORS}
    if any(m.set_resonance_damp_cmd is None for m in motors.values()):
        raise gcmd.error("VFA: MCU is not ready")
    configured = {name: dict(m.configured_damping) for name, m in motors.items()}
    previous = {name: dict(m.current_damping) for name, m in motors.items()}
    disabled = {name: {h: (0, 0, 0) for h in HARMONICS} for name in MOTORS}
    modes = {'before': disabled}
    if any(configured[name][h][0] for name in MOTORS for h in HARMONICS):
        modes['after'] = configured
    else:
        gcmd.respond_info("VFA: no saved compensation; measuring baseline only")
    names = [name for name in MOTORS if name[-1].upper() in axis]
    accel = min(MAX_ACCEL, tester.move_accel, status['max_accel'])
    if tester.kin != 'corexy':
        raise gcmd.error("VFA_GRAPH requires CoreXY kinematics")
    x_min = max(0., tester.x_min, status['axis_minimum'][0]) + MARGIN
    x_max = min(BED_MAX, tester.x_max, status['axis_maximum'][0]) - MARGIN
    y_min = max(0., tester.y_min, status['axis_minimum'][1]) + MARGIN
    y_max = min(BED_MAX, tester.y_max, status['axis_maximum'][1]) - MARGIN
    side = min(x_max - x_min, y_max - y_min)
    if side <= 0.:
        raise gcmd.error("VFA: no safe diagonal inside configured limits")
    # Равные проекции диагонали нужны для движения одного мотора CoreXY.
    cx, cy = (x_min + x_max) / 2., (y_min + y_max) / 2.
    x0, x1 = cx - side / 2., cx + side / 2.
    y0, y1 = cy - side / 2., cy + side / 2.
    paths = {}
    for name in names:
        start, end = ((x0, y0), (x1, y1)) if name == 'stepper_x' else (
            (x0, y1), (x1, y0))
        paths[name, 1] = ((start, end), math.sqrt(2.))
        paths[name, 0] = ((end, start), math.sqrt(2.))
    root = os.path.join(os.path.dirname(printer.get_start_args()['config_file']),
                        'mod_data', 'vfa')
    try:
        os.makedirs(root, exist_ok=True)
        folder = tempfile.mkdtemp(prefix='vfa_T%d_' % t, dir=root)
    except OSError as exc:
        raise gcmd.error("VFA: cannot create output directory: %s" % exc)
    metadata = {'tool': t, 'repeat': repeat, 'accel': accel,
                'speeds': speeds, 'max_frequency_hz': MAX_FREQ,
                'caution_frequency_hz': CAUTION_FREQ, 'margin_mm': MARGIN,
                'min_duration_s': MIN_DURATION, 'min_cycles': MIN_CYCLES,
                'settle_s': SETTLE, 'input_shaper': 'unchanged',
                'damping_mcu_units': modes,
                'previous_damping_mcu_units': previous,
                'paths': {name: {direction: paths[name, direction][0]
                                 for direction in (1, 0)}
                          for name in names}}
    with open(os.path.join(folder, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    toolhead.wait_moves()
    try:
        toolhead.set_max_velocities(None, accel, None, 0.)
        for name in names:
            tester.stepper = name
            tester.rotation_dis = tester.td_config[name]['rotation_dis']
            filename = os.path.join(folder, name + '.csv')
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['speed', 'direction', 'repeat', 'mode', 'td',
                                 'frequency_hz', 'frequency_band',
                                 'sample_rate_hz', 'duration_s',
                                 'samples', 'along_mm_s2', 'across_mm_s2',
                                 'total_mm_s2'])
                for direction in (1, 0):
                    tester.move, tester.vel_ratio = paths[name, direction]
                    for index, vel in enumerate(speeds):
                        freqs = [vel / tester.rotation_dis * tester.vel_ratio * 50 * h
                                 for h in HARMONICS]
                        for trial in range(repeat):
                            order = list(modes)
                            if (index + trial + direction) % 2 == 0:
                                order.reverse()
                            for label in order:
                                gcmd.respond_info(
                                    "VFA %s %s DIR=%d %.1f mm/s %d/%d"
                                    % (label, name, direction, vel, trial + 1, repeat))
                                apply_damping(motors, modes[label])
                                toolhead.dwell(.2)
                                toolhead.wait_moves()
                                fs, duration, count, amps = sample(
                                    tester, gcmd, toolhead, vel, accel, freqs)
                                for h, freq, values in zip(HARMONICS, freqs, amps):
                                    band = ('unmeasured' if not math.isfinite(values[2])
                                            else 'caution' if freq > CAUTION_FREQ
                                            else 'normal')
                                    writer.writerow([vel, direction, trial + 1, label,
                                                     h, freq, band, fs, duration, count]
                                                    + list(values))
                            f.flush()
    finally:
        try:
            if not printer.is_shutdown():
                try:
                    toolhead.wait_moves()
                finally:
                    restore_damping(motors, previous)
        finally:
            toolhead.set_max_velocities(None, status['max_accel'], None,
                                        status['minimum_cruise_ratio'])
    return folder, names


def render(tester, gcmd, folder, names):
    printer = tester.printer
    reactor = printer.get_reactor()
    # Не передаём пути и библиотеки Python заводской прошивки в chroot.
    env = os.environ.copy()
    for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONHOME', 'PYTHONPATH'):
        env.pop(key, None)
    env['PATH'] = '/bin:/sbin:/usr/bin:/usr/sbin'
    plot_folder = os.path.join('/opt/config/mod_data/vfa',
                               os.path.basename(folder))
    try:
        proc = subprocess.Popen([
            '/bin/sh', '/usr/data/zmod/zmod/.shell/chroot.sh',
            'python3', '/usr/data/zmod/klipper/klippy/extras/vfa_graph.py',
            plot_folder] + names, env=env)
    except OSError as exc:
        raise gcmd.error("VFA: CSV saved in %s; plot failed: %s" % (folder, exc))
    deadline = reactor.monotonic() + 120.
    try:
        while proc.poll() is None:
            if printer.is_shutdown() or reactor.monotonic() >= deadline:
                raise gcmd.error("VFA: graph generation interrupted; CSV: " + folder)
            reactor.pause(reactor.monotonic() + .1)
        if proc.returncode:
            raise gcmd.error("VFA: plot failed; CSV saved in " + folder)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    gcmd.respond_info("VFA graphs and CSV: " + folder)


def plot(folder, names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    for name in names:
        data = np.atleast_1d(np.genfromtxt(
            os.path.join(folder, name + '.csv'), delimiter=',', names=True,
            dtype=None, encoding='utf-8'))
        modes = [('before', 'tab:red')]
        if np.any(data['mode'] == 'after'):
            modes.append(('after', 'tab:green'))
        speeds = np.unique(data['speed'])
        for suffix, displacement in (('', False), ('_displacement', True)):
            fig, axes = plt.subplots(3, 3, sharex=True, figsize=(12, 9))
            for h, row in zip(HARMONICS, axes):
                for field, ax, title in zip(
                        ('total_mm_s2', 'along_mm_s2', 'across_mm_s2'), row,
                        ('XY magnitude', 'Along move', 'Across move')):
                    for direction, style in ((1, '-'), (0, '--')):
                        curves = {label: [] for label, _ in modes}
                        for speed in speeds:
                            subset = data[(data['td'] == h)
                                          & (data['direction'] == direction)
                                          & (data['speed'] == speed)]
                            pairs = []
                            for trial in np.unique(subset['repeat']):
                                values = []
                                for label, _ in modes:
                                    rows = subset[(subset['repeat'] == trial)
                                                  & (subset['mode'] == label)]
                                    if len(rows) != 1:
                                        break
                                    value = rows[field][0]
                                    if displacement:
                                        value *= 1000. / (2. * np.pi
                                                          * rows['frequency_hz'][0])**2
                                    values.append(value)
                                if len(values) == len(modes) and np.isfinite(values).all():
                                    pairs.append(values)
                            medians = (np.median(pairs, axis=0) if pairs
                                       else [float('nan')] * len(modes))
                            for (label, _), value in zip(modes, medians):
                                curves[label].append(value)
                        for label, color in modes:
                            ax.plot(speeds, curves[label], style, color=color,
                                    marker='.', markersize=3,
                                    label='%s / DIR=%d' % (label, direction))
                    harmonic = data[data['td'] == h]
                    hz_per_speed = harmonic['frequency_hz'][0] / harmonic['speed'][0]
                    left = max(speeds[0], CAUTION_FREQ / hz_per_speed)
                    right = min(speeds[-1], MAX_FREQ / hz_per_speed)
                    if right > left:
                        ax.axvspan(left, right, color='orange', alpha=.15)
                    ax.set_title('TD%d — %s' % (h, title), fontsize=10)
                    ax.grid(True, alpha=.25)
                    ax.set_ylim(bottom=0)
                    if not any(np.isfinite(line.get_ydata()).any() for line in ax.lines):
                        ax.text(.5, .5, 'No valid data', transform=ax.transAxes,
                                ha='center', va='center', color='gray')
                row[0].set_ylabel('µm (estimated)' if displacement else 'mm/s²')
            axes[0, 0].legend(fontsize=7, ncol=2)
            for ax in axes[-1]:
                ax.set_xlabel('Toolhead speed (mm/s)')
            fig.suptitle('VFA — ' + name + ' / ' + os.path.basename(folder).split('_')[1]
                         + '\nMedian of matched repeats; orange: >400 Hz, attenuated amplitudes')
            fig.tight_layout()
            fig.savefig(os.path.join(folder, name + suffix + '.png'), dpi=120)
            plt.close(fig)


if __name__ == '__main__':
    plot(sys.argv[1], sys.argv[2:])
