# Диагностика VFA для Creator 5 / Creator 5 Pro.
# This file may be distributed under the terms of the GNU GPLv3 license.
import csv
import logging
import math
import os
import subprocess
import sys
import tempfile

HARMONICS = (1, 2, 4)
MOTORS = ('stepper_x', 'stepper_y')
MAX_FREQ = 700.
SETTLE = 0.05


def speeds_for(gcmd, max_velocity):
    low = gcmd.get_float('MIN_SPEED', 20., minval=10., maxval=300.)
    high = gcmd.get_float('MAX_SPEED', min(300., max_velocity),
                          minval=10., maxval=min(300., max_velocity))
    step = gcmd.get_float('STEP', 10., minval=5., maxval=100.)
    if high <= low:
        raise gcmd.error("MAX_SPEED must exceed MIN_SPEED")
    count = int((high - low) / step)
    return [low + i * step for i in range(count + 1)] + (
        [high] if low + count * step < high - 1.e-6 else [])


def spectrum(samples, start, end, vel, accel, distance, freqs, fft):
    import numpy as np
    # Анализируем только равномерное движение, без разгона и торможения.
    ramp = vel / accel
    if abs(end - start - (distance / vel + ramp)) > 0.03:
        return None
    data = np.asarray(samples)
    if data.ndim != 2 or len(data) < 128:
        return None
    data = data[(data[:, 0] >= start + ramp + SETTLE)
                & (data[:, 0] <= end - ramp - SETTLE)]
    if len(data) < 128 or not np.isfinite(data).all():
        return None
    duration = data[-1, 0] - data[0, 0]
    if duration <= 0. or np.any(np.diff(data[:, 0]) <= 0.):
        return None
    fs = (len(data) - 1) / duration
    if np.max(np.diff(data[:, 0])) > 2.5 / fs:
        return None
    usable = [i for i, f in enumerate(freqs)
              if 8. <= f * duration and f <= min(MAX_FREQ, .43 * fs)]
    result = [float('nan')] * len(freqs)
    if usable:
        wanted = [freqs[i] for i in usable]
        x, _ = fft(data[:, 0], data[:, 1], wanted, fs)
        y, _ = fft(data[:, 0], data[:, 2], wanted, fs)
        for i, ax, ay in zip(usable, x, y):
            result[i] = math.hypot(ax, ay)
    return result


def sample(tester, gcmd, toolhead, vel, accel, fft):
    start_xy, end_xy = tester.move
    distance = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    if distance / vel - vel / accel <= 2. * SETTLE + .08:
        return [float('nan')] * len(HARMONICS)
    chip = tester.printer.lookup_object(tester.td_config[tester.stepper]['accel_chip'])
    freqs = [vel / tester.rotation_dis * tester.vel_ratio * 50 * h
             for h in HARMONICS]
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
                          distance, freqs, fft)
        if values is not None:
            return values
    raise gcmd.error("VFA: no valid acceleration samples at %.1f mm/s" % vel)


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


def measure(tester, gcmd, fft):
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
    if not all(axis in status['homed_axes'] for axis in 'xyz'):
        raise gcmd.error("Home XYZ before VFA_GRAPH")
    if status['position'][2] < 100.:
        raise gcmd.error("Move Z to at least 100 mm and clear the bed before VFA_GRAPH")
    # Проверяем датчики всех четырёх инструментов, не меняя инструмент.
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
    if len(attached) != 1 or attached != undocked:
        raise gcmd.error("Attach exactly one tool and dock all other tools")
    axis = gcmd.get('AXIS', 'XY').upper()
    if axis not in ('X', 'Y', 'XY'):
        raise gcmd.error("AXIS must be X, Y or XY")
    speeds = speeds_for(gcmd, status['max_velocity'])
    motors = {name: printer.lookup_object('mclib ' + name) for name in MOTORS}
    if any(m.set_resonance_damp_cmd is None for m in motors.values()):
        raise gcmd.error("VFA: MCU is not ready")
    configured = {name: dict(m.configured_damping) for name, m in motors.items()}
    previous = {name: dict(m.current_damping) for name, m in motors.items()}
    if not any(configured[name][h][0] for name in MOTORS for h in HARMONICS):
        raise gcmd.error("No saved VFA compensation; calibrate VFA first")
    names = [name for name in MOTORS if name[-1].upper() in axis]
    accel = min(tester.move_accel, status['max_accel'])
    # Используем штатные траектории калибровки и машинные координаты.
    tester.move_params = None
    paths = {}
    for name in names:
        tester.stepper = name
        tester.rotation_dis = tester.td_config[name]['rotation_dis']
        for direction in (1, 0):
            tester.update_move(name, direction)
            if not all(tester.x_min <= x <= tester.x_max
                       and tester.y_min <= y <= tester.y_max
                       and status['axis_minimum'][0] <= x <= status['axis_maximum'][0]
                       and status['axis_minimum'][1] <= y <= status['axis_maximum'][1]
                       for x, y in tester.move):
                raise gcmd.error("VFA test path outside configured limits")
            paths[name, direction] = (tester.move, tester.vel_ratio)
    root = os.path.join(os.path.dirname(printer.get_start_args()['config_file']),
                        'mod_data', 'vfa')
    try:
        os.makedirs(root, exist_ok=True)
        folder = tempfile.mkdtemp(prefix='vfa_', dir=root)
    except OSError as exc:
        raise gcmd.error("VFA: cannot create output directory: %s" % exc)
    toolhead.wait_moves()
    try:
        toolhead.set_max_velocities(None, accel, None, 0.)
        for label, values in (
                ('before', {name: {h: (0, 0, 0) for h in HARMONICS}
                            for name in MOTORS}),
                ('after', configured)):
            apply_damping(motors, values)
            toolhead.dwell(.2)
            toolhead.wait_moves()
            for name in names:
                tester.stepper = name
                tester.rotation_dis = tester.td_config[name]['rotation_dis']
                filename = os.path.join(folder, name + '_' + label + '.csv')
                with open(filename, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['speed', 'direction', 'td1', 'td2', 'td4'])
                    for direction in (1, 0):
                        tester.move, tester.vel_ratio = paths[name, direction]
                        for vel in speeds:
                            gcmd.respond_info("VFA %s %s DIR=%d %.1f mm/s"
                                              % (label, name, direction, vel))
                            amps = sample(tester, gcmd, toolhead, vel, accel, fft)
                            writer.writerow([vel, direction] + amps)
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
    # Matplotlib запускается отдельным процессом после восстановления MCU.
    try:
        proc = subprocess.Popen(['python3', os.path.abspath(__file__),
                                 folder] + names)
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
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
        for label, color in (('before', 'tab:red'), ('after', 'tab:green')):
            data = np.atleast_1d(np.genfromtxt(
                os.path.join(folder, name + '_' + label + '.csv'),
                delimiter=',', names=True))
            for direction, style in ((1, '-'), (0, '--')):
                rows = data[data['direction'] == direction]
                for h, ax in zip(HARMONICS, axes):
                    ax.plot(rows['speed'], rows['td%d' % h], style,
                            color=color, label='%s / DIR=%d' % (label, direction))
        for h, ax in zip(HARMONICS, axes):
            ax.set_ylabel('TD%d (mm/s²)' % h)
            ax.grid(True, alpha=.25)
            ax.set_ylim(bottom=0)
        axes[0].legend(fontsize=8, ncol=2)
        axes[-1].set_xlabel('Toolhead speed (mm/s)')
        fig.suptitle('VFA — ' + name + ' (gaps: unmeasurable)')
        fig.tight_layout()
        fig.savefig(os.path.join(folder, name + '.png'), dpi=120)
        plt.close(fig)


if __name__ == '__main__':
    plot(sys.argv[1], sys.argv[2:])
