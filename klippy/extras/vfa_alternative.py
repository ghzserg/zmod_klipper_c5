# Альтернативная калибровка VFA для Z-Mod.
# Алгоритм адаптирован из OpenCreator (Kalico, 7da732e09d4e213b7766482b6cfdf7e75ee9e191).
# Этот файл распространяется на условиях GNU GPLv3.
import math
import os
import numpy as np

SWEEP_SPACING = 0.08
RESONANCE_CANDIDATES = 3
STRONG_FRACTION = 0.5
TUNE_CANDIDATES = 2
RESONANCE_PROMINENCE = 1.5

def combine_direction_amps(amps, fits):
    lo, hi = min(amps), max(amps)
    if len(fits) == len(amps) and all(f is not None and f[2] > 0. for f in fits):
        c1 = sum(f[1] for f in fits)
        c2 = sum(f[2] for f in fits)
        return float(np.clip(-c1 / (2.0 * c2), lo, hi))
    return float(sum(amps) / len(amps))

def damping_helps(bare, final, min_gain=0.05):
    bare_e = sum(r * r for r in bare)
    final_e = sum(r * r for r in final)
    return bare_e > 0. and final_e <= bare_e * (1.0 - min_gain) ** 2

def sweep_speeds(vmin, vmax, spacing=SWEEP_SPACING, step=10.):
    if spacing > 0.:
        n = int(math.ceil(math.log(vmax / vmin) / math.log(1. + spacing))) + 1
        return [round(float(v), 2) for v in np.geomspace(vmin, vmax, n)]
    return [round(float(v), 2) for v in np.arange(vmin, vmax + step / 2., step)]

def resonance_candidates(amps, count=RESONANCE_CANDIDATES):
    n = len(amps)
    if n < 2:
        return []
    peaks = [i for i in range(n)
             if (i == 0 or amps[i] >= amps[i - 1])
             and (i == n - 1 or amps[i] >= amps[i + 1])]
    return sorted(peaks, key=lambda i: -amps[i])[:count]

def strong_resonances(vels, amps, regions, fraction=STRONG_FRACTION,
                      count=TUNE_CANDIDATES):
    vels = np.asarray(vels, dtype=float)
    amps = np.asarray(amps, dtype=float)
    if len(vels) < 3:
        return []
    floor = RESONANCE_PROMINENCE * float(np.median(amps))
    found = []
    for lo, hi in regions:
        idx = [i for i in range(len(vels)) if lo - 1e-6 <= vels[i] <= hi + 1e-6]
        if not idx:
            continue
        i = max(idx, key=lambda j: amps[j])
        if i == 0 or i == len(vels) - 1:
            continue
        peak_vel, peak_amp = vels[i], amps[i]
        coeff = np.polyfit(vels[i - 1:i + 2], amps[i - 1:i + 2], 2)
        if coeff[0] < 0:
            v = -coeff[1] / (2.0 * coeff[0])
            if vels[i - 1] <= v <= vels[i + 1]:
                peak_vel, peak_amp = v, float(np.polyval(coeff, v))
        if peak_amp < floor:
            continue
        if any(abs(peak_vel - v) < 0.05 * v for v, _ in found):
            continue
        found.append((float(peak_vel), float(peak_amp)))
    if not found:
        return []
    top = max(a for _, a in found)
    found = [f for f in found if f[1] >= fraction * top]
    return sorted(found, key=lambda f: -f[1])[:count]

def measure_sweep(tester, gcmd, speeds, rows):
    # Сохраняем проверенное измерительное окно Z-Mod и частоту дискретизации.
    for speed in speeds:
        vel = round(float(speed), 2)
        if vel in rows:
            continue
        freqs = [vel / tester.rotation_dis * tester.vel_ratio * 50 * t
                 for t in tester.tdx]
        for _ in range(5):
            samples, start, end = tester.do_move_and_accel_measure(vel)
            amps = tester.calculate_resonance_amps(
                samples, freqs, tester.move_angle, tester.axis, start, end, vel)
            if amps is not None and any(np.isfinite(amps)):
                rows[vel] = amps
                break
        else:
            raise gcmd.error("Invalid VFA sweep measurement at %.2f mm/s" % vel)
        gcmd.respond_info("  %.2f mm/s: %s" %
                          (vel, ", ".join("td%d=%.1f" % (t, a)
                                          for t, a in zip(tester.tdx, rows[vel]))))


def find_resonances(tester, gcmd, speeds, folder):
    tester.update_move(tester.stepper, 1)
    rows = {}
    measure_sweep(tester, gcmd, speeds, rows)
    mi = tester.tdx.index(tester.main_tdx)
    seen = sorted(v for v in rows if np.isfinite(rows[v][mi]))
    regions = []
    for i in resonance_candidates([rows[v][mi] for v in seen]):
        lo = seen[max(i - 1, 0)]
        hi = seen[min(i + 1, len(seen) - 1)]
        regions.append((lo, hi))
        measure_sweep(tester, gcmd, np.linspace(lo, hi, 9), rows)
    seen = sorted(v for v in rows if np.isfinite(rows[v][mi]))
    strong = strong_resonances(seen, [rows[v][mi] for v in seen], regions)
    tester.save_to_csv(os.path.join(folder, "%s_sweep.csv" % tester.stepper),
                       [[v] + rows[v] for v in sorted(rows)],
                       header=["vel"] + ["td%d_amp" % t for t in tester.tdx])
    return [v / tester.rotation_dis * tester.vel_ratio * 50 * tester.main_tdx
            for v, _ in strong]


def tune(tester, gcmd, freq, folder):
    motor, _ = tester.test_damping
    tester.res_freq = freq
    for tdx in tester.tdx:
        motor.set_resonance_damping(tdx, 0, 0, 0)
    pending, history = [], []
    for tdx in tester.tdx:
        amps, fits, bare = [], [], []
        phases = [0., 0.]
        for direction in (1, 0):
            amp, phase, rows, fit = tester.search_calibrate_one_direction(
                gcmd, tdx, direction, alternative=True)
            idx = int(not (direction ^ tester.dir_inverted))
            phases[idx] = phase
            amps.append(amp)
            fits.append(fit)
            bare.append(rows[0][3])
            history.extend([tdx, direction] + row for row in rows)
        used = [i for i, amp in enumerate(amps) if amp > 0.]
        amp = combine_direction_amps([amps[i] for i in used],
                                     [fits[i] for i in used]) if used else 0.
        # Общая амплитуда проверяется в обоих направлениях, даже если
        # одна из исходных вибраций была ниже порога поиска.
        residual = []
        if amp > 0.:
            for direction in (1, 0):
                idx = int(not (direction ^ tester.dir_inverted))
                tester.update_move(tester.stepper, direction)
                vel = freq / 50 / tdx * tester.rotation_dis / tester.vel_ratio
                value = tester._apply_comp_and_measure(
                    gcmd, vel, tdx, direction, amp, phases[idx])
                residual.append(value)
                history.append([tdx, direction, 6, amp, phases[idx], value])
            if not damping_helps(bare, residual):
                amp, phases = 0., [0., 0.]
                gcmd.respond_info("td%d: alternative damping rejected" % tdx)
        gcmd.respond_info("td%d: common amp=%.4f (individual %.4f/%.4f), "
                          "bare=%s, final=%s" %
                          (tdx, amp, amps[0], amps[1], bare, residual))
        tester.apply_calibration(tdx, amp, phases[0], phases[1], pending)
    tester.save_to_csv(os.path.join(folder, "%s_alternative-%.3fHz.csv" %
                                    (tester.stepper, freq)),
                       history, header=["tdx", "dir", "round", "amp", "phase", "residual"],
                       format_spec='%.4f')
    return pending, dict(motor.current_damping)


def check_sweep(tester, gcmd, speeds, folder, freq):
    result = []
    for direction in (1, 0):
        tester.update_move(tester.stepper, direction)
        rows = {}
        measure_sweep(tester, gcmd, speeds, rows)
        result.extend(rows[v] for v in sorted(rows))
        tester.save_to_csv(os.path.join(folder, "%s_check-%.3fHz-dir%d.csv" %
                                        (tester.stepper, freq, direction)),
                           [[v] + rows[v] for v in sorted(rows)],
                           header=["vel"] + ["td%d_amp" % t for t in tester.tdx])
    return np.asarray(result)


def calibrate(tester, gcmd, pending, folder):
    tester.prepare_test(gcmd)
    motor, _ = tester.test_damping
    configured = tester.res_freq
    toolhead = tester.printer.lookup_object('toolhead')
    try:
        # Заводской диапазон часто заканчивается на 120 мм/с.
        # Для альтернативы используем диапазон OpenCreator 20–300 мм/с.
        vmax = min(300., toolhead.max_velocity)
        if vmax < 20.:
            raise gcmd.error("VFA alternative requires max_velocity >= 20")
        speeds = sweep_speeds(20., vmax)
        for tdx in tester.tdx:
            motor.set_resonance_damping(tdx, 0, 0, 0)
        freqs = find_resonances(tester, gcmd, speeds, folder)
        # При отсутствии выраженного пика остаётся заводская частота.
        # TD1 требует самой высокой скорости. Не выбираем частоту,
        # недоступную для калибровки всех трёх гармоник на данной машине.
        tester.update_move(tester.stepper, 1)
        distance = np.hypot(tester.move[1][0] - tester.move[0][0],
                            tester.move[1][1] - tester.move[0][1])
        chip = tester.printer.lookup_object(
            tester.td_config[tester.stepper]['accel_chip'])
        usable = []
        for freq in freqs + [configured]:
            if not 0. < freq < min(700., .45 * chip.data_rate):
                continue
            valid = True
            for tdx in tester.tdx:
                vel = freq / 50 / tdx * tester.rotation_dis / tester.vel_ratio
                duration = distance / vel - vel / toolhead.max_accel - .1
                if (vel > toolhead.max_velocity
                        or duration < max(20. / freq, 130. / chip.data_rate)):
                    valid = False
                    break
            if valid:
                usable.append(freq)
        freqs = [f for f in freqs if f in usable]
        if not freqs:
            if configured not in usable:
                raise gcmd.error("No measurable VFA calibration frequency")
            freqs = [configured]
        gcmd.respond_info("%s: alternative frequencies %s Hz" %
                          (tester.stepper, ", ".join("%.1f" % f for f in freqs)))
        trials = []
        for freq in freqs:
            values, damping = tune(tester, gcmd, freq, folder)
            sweep = (check_sweep(tester, gcmd, speeds, folder, freq)
                     if len(freqs) > 1 else None)
            trials.append((freq, values, damping, sweep))
        scores = []
        if len(trials) > 1:
            # Сравниваем кандидатов по одинаковым измеряемым точкам.
            mask = np.logical_and.reduce([np.isfinite(t[3]) for t in trials])
            if not mask.any():
                raise gcmd.error("No comparable VFA sweep measurements")
            scores = [float(np.sum(t[3][mask] ** 2)) for t in trials]
            for trial, score in zip(trials, scores):
                gcmd.respond_info("%s: %.1f Hz score %.1f" %
                                  (tester.stepper, trial[0], score))
            winner = trials[int(np.argmin(scores))]
        else:
            winner = trials[0]
        freq, values, damping, _ = winner
        for tdx, params in damping.items():
            motor.set_resonance_damping(tdx, *params)
        pending.extend(values)
        gcmd.respond_info("%s: alternative selected %.1f Hz" %
                          (tester.stepper, freq))
        # Частота остаётся диагностической: старый алгоритм использует
        # исходную конфигурацию, а SAVE_CONFIG не конфликтует с include.
        with open(os.path.join(folder, "%s_alternative.info" % tester.stepper), "w") as f:
            f.write("selected_freq=%.3f\nconfigured_freq=%.3f\n" %
                    (freq, configured))
            for trial, score in zip(trials, scores):
                f.write("candidate=%.3f score=%.6f\n" % (trial[0], score))
    finally:
        tester.res_freq = configured
        tester.end_test(gcmd)
