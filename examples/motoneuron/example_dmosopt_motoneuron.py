import os
import sys
import logging

import click
import numpy as np
import yaml
from functools import partial
from numpy.random import default_rng
from neuron import h
from neuron import load_mechanisms
from mpi4py import MPI
from scipy import optimize, signal

from dmosopt import dmosopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_mechanisms(os.path.join(os.path.dirname(__file__), "mechanisms"))
h.load_file("stdrun.hoc")

pc = h.ParallelContext()
if hasattr(pc, "mpiabort_on_error"):
    pc.mpiabort_on_error(0)


class BRK:
    def __init__(self, params=None):
        if params is not None:
            params = params["BoothRinzelKiehn"]

        # Create sections
        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name="dend", cell=self)

        # hack: use default attribute to trigger detection
        self.hillock = self.dend

        # Initialize position coordinates
        self.x = self.y = self.z = 0

        # Create section lists
        self.sections = h.SectionList()
        self.all = h.SectionList()

        if params is not None:
            self.set_parameters(params)
        else:
            self.set_default_parameters()

        self.init_topology()
        self.geometry()
        self.biophys()

        # Add sections to lists
        for sec in [self.soma, self.dend]:
            self.all.append(sec)
        self.sections = list(self.all)

    def set_default_parameters(self):
        self.pp = 0.5  # proportion of area taken up by soma
        self.Ltotal = 400 / np.pi  # total length of compartments
        self.gc = 10.5  # mS/cm2; Ra in ohm-cm

        self.global_e_pas = -60
        self.soma_g_pas = 0.0001
        self.soma_gmax_Na = 0.00030
        self.soma_gmax_K = 0.00010
        self.soma_gmax_KCa = 0.0005
        self.soma_gmax_CaN = 0.00010

        self.soma_f_Caconc = 0.004
        self.soma_alpha_Caconc = 1
        self.soma_kCa_Caconc = 8

        self.dend_g_pas = 0.0001
        self.dend_gmax_CaN = 0.00010
        self.dend_gmax_CaL = 0.00010
        self.dend_gmax_KCa = 0.00015

        self.dend_f_Caconc = 0.004
        self.dend_alpha_Caconc = 1
        self.dend_kCa_Caconc = 8

        self.global_cm = 3
        self.global_diam = 10  # Default value
        self.cm_ratio = 1

    def set_parameters(self, params):
        self.pp = params.get("pp")
        self.Ltotal = params.get("Ltotal")
        self.gc = params.get("gc")

        self.global_diam = params.get("global_diam")
        self.global_cm = params.get("global_cm")
        self.cm_ratio = params.get("cm_ratio", 1.0)

        self.global_e_pas = params.get("e_pas", -60)
        self.soma_g_pas = params.get("soma_g_pas")
        self.soma_gmax_Na = params.get("soma_gmax_Na")
        self.soma_gmax_K = params.get("soma_gmax_K")
        self.soma_gmax_KCa = params.get("soma_gmax_KCa")
        self.soma_gmax_CaN = params.get("soma_gmax_CaN")

        self.soma_f_Caconc = params.get("soma_f_Caconc")
        self.soma_alpha_Caconc = params.get("soma_alpha_Caconc")
        self.soma_kCa_Caconc = params.get("soma_kCa_Caconc")

        self.dend_g_pas = params.get("dend_g_pas")
        self.dend_gmax_CaN = params.get("dend_gmax_CaN")
        self.dend_gmax_CaL = params.get("dend_gmax_CaL")
        self.dend_gmax_KCa = params.get("dend_gmax_KCa")

        self.dend_f_Caconc = params.get("dend_f_Caconc")
        self.dend_alpha_Caconc = params.get("dend_alpha_Caconc")
        self.dend_kCa_Caconc = params.get("dend_kCa_Caconc")

    def lambda_f(self, section, freq):
        if section.n3d() < 2:
            return 1e5 * np.sqrt(
                section.diam / (4 * np.pi * freq * section.Ra * section.cm)
            )

        x1 = section.arc3d(0)
        d1 = section.diam3d(0)
        lam = 0

        for i in range(1, section.n3d()):
            x2 = section.arc3d(i)
            d2 = section.diam3d(i)
            lam += (x2 - x1) / np.sqrt(d1 + d2)
            x1, d1 = x2, d2

        lam *= np.sqrt(2) * 1e-5 * np.sqrt(4 * np.pi * freq * section.Ra * section.cm)
        return section.L / lam

    def init_topology(self):
        self.dend.connect(self.soma(1), 0)

    def geometry(self):
        self.init_dx()
        self.init_diam()
        self.init_nseg()

    def init_dx(self):
        self.soma.L = self.pp * self.Ltotal
        self.dend.L = (1 - self.pp) * self.Ltotal

    def init_diam(self):
        self.soma.diam = self.global_diam
        self.dend.diam = self.global_diam

    def init_nseg(self, freq=100, d_lambda=0.1):
        for sec in [self.soma, self.dend]:
            nseg = (
                int((sec.L / (d_lambda * self.lambda_f(sec, freq)) + 0.9) / 2) * 2 + 1
            )
            sec.nseg = nseg

    def init_ic(self, v_init):
        h.finitialize(v_init)
        seg = self.soma(0.5)
        self.soma.ic_constant = -(seg.ina + seg.ik + seg.ica + seg.i_pas)

    def biophys(self):
        # Set global parameters
        for sec in [self.soma, self.dend]:
            sec.Ra = 1
            sec.Ra = (
                1e-6 / (self.gc / self.pp * (h.area(0.5, sec=sec) * 1e-8) * 1e-3)
            ) / (2 * h.ri(0.5, sec=sec))
            sec.cm = self.global_cm

        # Soma-specific parameters
        self.soma.cm = self.global_cm * self.cm_ratio

        self.soma.insert("pas")
        self.soma.insert("constant")
        self.soma.insert("Na_conc")
        self.soma.insert("K_conc")
        self.soma.insert("Ca_conc")
        self.soma.insert("Kdr")
        self.soma.insert("Nas")
        self.soma.insert("CaN")
        self.soma.insert("KCa")
        self.soma.insert("extracellular")  # For stimulation

        self.soma.gmax_Nas = self.soma_gmax_Na
        self.soma.gmax_Kdr = self.soma_gmax_K
        self.soma.gmax_CaN = self.soma_gmax_CaN
        self.soma.gmax_KCa = self.soma_gmax_KCa

        self.soma.f_Ca_conc = self.soma_f_Caconc
        self.soma.alpha_Ca_conc = self.soma_alpha_Caconc
        self.soma.kCa_Ca_conc = self.soma_kCa_Caconc

        self.soma.g_pas = self.soma_g_pas
        self.soma.e_pas = self.global_e_pas

        # Dendrite-specific parameters
        self.dend.insert("pas")
        self.dend.insert("CaN")
        self.dend.insert("CaL")
        self.dend.insert("KCa")
        self.dend.insert("Ca_conc")
        self.dend.insert("K_conc")
        self.dend.insert("extracellular")  # For stimulation

        self.dend.f_Ca_conc = self.dend_f_Caconc
        self.dend.alpha_Ca_conc = self.dend_alpha_Caconc
        self.dend.kCa_Ca_conc = self.dend_kCa_Caconc

        self.dend.g_pas = self.dend_g_pas
        self.dend.e_pas = self.global_e_pas

        self.dend.gmax_CaN = self.dend_gmax_CaN
        self.dend.gmax_CaL = self.dend_gmax_CaL
        self.dend.gmax_KCa = self.dend_gmax_KCa

    def position(self, x, y, z):
        for sec in [self.soma, self.dend]:
            for i in range(sec.n3d()):
                h.pt3dchange(
                    i,
                    x - self.x + sec.x3d(i),
                    y - self.y + sec.y3d(i),
                    z - self.z + sec.z3d(i),
                    sec.diam3d(i),
                    sec=sec,
                )
        self.x, self.y, self.z = x, y, z

    def is_art(self):
        return False

    def is_reduced(self):
        return True

    def __repr__(self):
        return "BRK"


def ic_constant_f(
    x,
    template_class,
    param_dict,
    ic_constant,
    v_hold=-60,
    tstop=1000.0,
    dt=0.01,
    record_dt=0.01,
    celsius=36.0,
    use_cvode=False,
    use_coreneuron=True,
):
    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h.secondorder = 2
    h.dt = dt
    if record_dt < dt:
        record_dt = dt
    if use_cvode:
        h.cvode.active(1)
    if use_coreneuron:
        from neuron import coreneuron

        coreneuron.enable = True
    h.celsius = celsius

    cell = template_class({"BoothRinzelKiehn": param_dict})

    vec_t = h.Vector()
    vec_v = h.Vector()
    vec_t.record(h._ref_t, record_dt)
    vec_v.record(cell.soma(0.5)._ref_v, record_dt)

    h.tstop = tstop
    h.v_init = v_hold
    h.init()
    cell.soma.ic_constant = ic_constant + round(x, 6)
    h.finitialize(h.v_init)
    h.finitialize(h.v_init)
    try:
        h.run()
    except:
        pass

    t = vec_t.as_numpy()
    v = vec_v.as_numpy()
    mean_v = np.mean(v) if np.max(v) < 0.0 else 0.0
    return mean_v - v_hold


def run_iclamp(
    cell,
    amp,
    t0,
    t1,
    v_init=-65,
    tstop=1000.0,
    dt=0.01,
    record_dt=0.01,
    celsius=36.0,
    use_cvode=False,
    use_coreneuron=True,
):
    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h.secondorder = 2
    h.dt = dt
    if record_dt < dt:
        record_dt = dt
    if use_cvode:
        h.cvode.active(1)
    if use_coreneuron:
        from neuron import coreneuron

        coreneuron.enable = True
    h.celsius = celsius

    vec_t = h.Vector()
    vec_v = h.Vector()
    vec_t.record(h._ref_t, record_dt)
    vec_v.record(cell.soma(0.5)._ref_v, record_dt)

    stim = h.IClamp(0.5, sec=cell.soma)
    stim.delay = t0
    stim.dur = t1 - t0
    stim.amp = amp

    if tstop < t1 + 1.0:
        tstop = t1 + 1.0

    h.tstop = tstop
    h.v_init = v_init
    h.init()
    h.finitialize(h.v_init)
    h.run()

    return np.array(vec_t), np.array(vec_v)


def run_iclamp_steps(cell, Isteps, **kwargs):
    results = []
    for amp, t0, t1 in Isteps:
        try:
            t, v = run_iclamp(cell, amp, t0, t1, **kwargs)
        except:
            results.append(None)
        else:
            results.append({"t": t, "v": v})
    return results


fi_value_dtype = np.dtype([("frequency", float)])
isi_value_dtype = np.dtype(
    [
        ("first", float),
        ("last", float),
        ("ratio", float),
        ("mean", float),
        ("std", float),
        ("N", int),
    ]
)


def measure_deflection(t, v, t0, t1, stim_amp=None):
    start_index = int(np.argwhere(t >= t0 * 0.999)[0, 0])
    end_index = int(np.argwhere(t >= t1 * 0.999)[0, 0])
    deflect_fn = np.argmax if (stim_amp is not None and stim_amp > 0) else np.argmin
    v_window = v[start_index:end_index]
    peak_index = deflect_fn(v_window) + start_index
    return {
        "t_peak": t[peak_index],
        "v_peak": v[peak_index],
        "peak_index": peak_index,
        "t_baseline": t[start_index],
        "v_baseline": v[start_index],
        "baseline_index": start_index,
        "stim_amp": stim_amp,
    }


def fit_membrane_time_constant(t, v, t0, t1, rmse_max_tol=1.0):
    def exp_curve(x, a, inv_tau, y0):
        return y0 + a * np.exp(-inv_tau * x)

    start_index = int(np.argwhere(t >= t0 * 0.999)[0, 0])
    end_index = int(np.argwhere(t >= t1 * 0.999)[0, 0])
    p0 = (v[start_index] - v[end_index], 0.1, v[end_index])
    t_window = (t[start_index:end_index] - t[start_index]).astype(np.float64)
    v_window = v[start_index:end_index].astype(np.float64)
    try:
        popt, pcov = optimize.curve_fit(exp_curve, t_window, v_window, p0=p0)
    except (TypeError, RuntimeError):
        return np.nan, np.nan, np.nan

    pred = exp_curve(t_window, *popt)
    rmse = np.sqrt(np.mean((pred - v_window) ** 2))
    if rmse > rmse_max_tol:
        return np.nan, np.nan, np.nan
    return popt


def measure_time_constant(
    t, v, t0, t1, stim_amp, frac=0.1, baseline_interval=100.0, min_snr=20.0
):
    if np.max(t) < t0 or np.max(t) < t1:
        return np.nan

    deflection_results = measure_deflection(t, v, t0, t1, stim_amp)
    v_peak = deflection_results["v_peak"]
    peak_index = deflection_results["peak_index"]
    v_baseline = deflection_results["v_baseline"]
    start_index = deflection_results["baseline_index"]

    signal_val = np.abs(v_baseline - v_peak)
    noise_interval_start_index = int(
        np.argwhere(t >= (t0 - baseline_interval) * 0.999)[0, 0]
    )
    noise = np.std(v[noise_interval_start_index:start_index])

    snr = np.inf if noise == 0 else signal_val / noise
    if snr < min_snr:
        return np.nan

    search_result = np.flatnonzero(
        v[start_index:] <= frac * (v_peak - v_baseline) + v_baseline
    )
    if not search_result.size:
        return np.nan

    fit_start_index = search_result[0] + start_index
    fit_end_index = peak_index
    fit_start = t[fit_start_index]
    fit_end = t[fit_end_index]

    if not (fit_start < fit_end):
        return np.nan

    a, inv_tau, y0 = fit_membrane_time_constant(t, v, fit_start, fit_end)
    return 1.0 / inv_tau


def measure_passive(t, v, t0, t1, stim_amp):
    if np.max(t) < t0 or np.max(t) < t1:
        return {"Rinp": np.nan, "tau": np.nan}
    deflection_results = measure_deflection(t, v, t0, t1, stim_amp=stim_amp)
    v_peak = deflection_results["v_peak"]
    v_baseline = deflection_results["v_baseline"]
    Rinp = (v_peak - v_baseline) / stim_amp
    tau = measure_time_constant(t, v, t0, t1, stim_amp)
    return {"Rinp": Rinp, "tau": tau}


def detect_spikes(T, Y, t0, t1, before_peak=50.0):
    spk_info_dtype = np.dtype(
        [
            ("Vpeak", float),
            ("Tpeak", float),
            ("amplitude", float),
            ("T0", float),
            ("T1", float),
        ]
    )

    dt = np.mean(np.diff(T))
    pre_period_idxs = np.argwhere(T < t0 - before_peak).flat
    pre_peak_info = signal.find_peaks(
        Y[pre_period_idxs], height=-20.0, width=(None, int(before_peak / dt))
    )
    N_peaks_pre = len(pre_peak_info[0])

    spk_period_idxs = np.argwhere(np.logical_and(T >= t0 - before_peak, T <= t1)).flat
    T_spk = T[spk_period_idxs]
    Y_spk = Y[spk_period_idxs]
    peak_info = signal.find_peaks(
        Y_spk, height=-20.0, width=(None, int(before_peak / dt))
    )
    peak_idxs = peak_info[0]
    if len(peak_idxs) == 0:
        return N_peaks_pre, 0, None, None

    dydt = np.gradient(Y_spk, T_spk)
    mean_dydt = np.mean(dydt)
    peak_idx = peak_idxs[0]
    T_peak = T_spk[peak_idx]
    T_before_idxs = np.argwhere(
        np.isclose(T_spk, T_peak - before_peak, rtol=1e-4, atol=1e-4)
    )
    if len(T_before_idxs) == 0:
        return N_peaks_pre, 0, None, None

    T_before_idx = T_before_idxs[0][0]
    sd_dydt = np.std(dydt[T_before_idx:peak_idx])
    dydt_threshold = mean_dydt + 2 * sd_dydt
    try:
        threshold_idx = np.argwhere(dydt[T_before_idx:peak_idx] >= dydt_threshold)[0]
    except:
        return N_peaks_pre, 0, None, None

    threshold = Y_spk[T_before_idx:peak_idx][threshold_idx][0]

    period_idxs = np.argwhere(np.logical_and(T >= t0, T <= t1)).flat
    T = T[period_idxs]
    Y = Y[period_idxs]

    threshold_crossings = np.diff(Y > threshold, prepend=False)
    up_crossing_idx = np.argwhere(threshold_crossings)[::2, 0]
    crossing_idx = np.argwhere(threshold_crossings)[:, 0]
    N_peaks = len(up_crossing_idx)

    spk_info = None
    if N_peaks > 0:
        Y_intervals = np.split(Y, crossing_idx[1::2])[:-1]
        T_intervals = np.split(T, crossing_idx[1::2])[:-1]
        peak_idxs = [np.argmax(Yi) for Yi in Y_intervals]
        N_peaks = len(peak_idxs)
        Y_peaks, T_peaks, peak_amps = [], [], []
        for j, (T_interval, pidx) in enumerate(zip(T_intervals, peak_idxs)):
            if len(T_interval) < 2:
                N_peaks -= 1
                continue
            peak_amps.append(np.max(Y_intervals[j]) - threshold)
            Y_peaks.append(Y_intervals[j][pidx])
            T_peaks.append(
                T_interval[pidx] if pidx < len(T_interval) else T_interval[-1]
            )

        if N_peaks > 0:
            spk_info = np.zeros(shape=(N_peaks,), dtype=spk_info_dtype)
            for p in range(N_peaks):
                spk_info[p]["Vpeak"] = Y_peaks[p]
                spk_info[p]["Tpeak"] = T_peaks[p]
                spk_info[p]["T0"] = T_intervals[p][0]
                spk_info[p]["T1"] = T_intervals[p][-1]
                spk_info[p]["amplitude"] = peak_amps[p]

    return N_peaks_pre, N_peaks, threshold, spk_info


def measure_spike_features(iclamp_results, t0, t1):
    N_sweeps = len(iclamp_results)
    pre_spk_cnt = np.zeros(shape=(N_sweeps, 1), dtype=int)
    spk_cnt = np.zeros(shape=(N_sweeps, 1), dtype=int)
    spk_infos = []
    thresholds = np.zeros(shape=(N_sweeps,), dtype=np.float32)
    mean_spike_amplitudes = np.zeros(shape=(N_sweeps,), dtype=np.float32)
    for i in range(N_sweeps):
        if iclamp_results[i] is None:
            continue
        T = iclamp_results[i]["t"]
        Y = iclamp_results[i]["v"]
        N_peaks_pre, N_peaks, threshold, spk_info = detect_spikes(T, Y, t0, t1)
        spk_infos.append(spk_info)
        pre_spk_cnt[i, 0] = N_peaks_pre
        spk_cnt[i, 0] = N_peaks
        if threshold is not None:
            thresholds[i] = threshold
        if spk_info is not None:
            mean_spike_amplitudes[i] = np.mean(spk_info["amplitude"])
    return pre_spk_cnt, spk_cnt, spk_infos, thresholds, mean_spike_amplitudes


def measure_fI(spk_cnt, t0, t1, cur_steps):
    N_inj = len(cur_steps)
    fI_array = np.zeros(shape=(N_inj,), dtype=fi_value_dtype)
    for i in range(N_inj):
        if i >= len(spk_cnt):
            break
        fI_array[i]["frequency"] = spk_cnt[i, 0] * 1000 / (t1 - t0)
    return fI_array


def measure_ISI(cur_steps_amp, spk_infos):
    N = len(cur_steps_amp)
    ISI_array = np.zeros(shape=(N,), dtype=isi_value_dtype)
    for field in ["first", "last", "ratio", "mean", "std"]:
        ISI_array[field] = np.nan
    for i in range(N):
        if i >= len(spk_infos):
            break
        spk_info = spk_infos[i]
        N_peaks = spk_info.shape[0] if spk_info is not None else 0
        ISI_array[i]["N"] = N_peaks
        if N_peaks >= 2:
            ISI = np.diff(spk_info["Tpeak"])
            ISI_array[i]["first"] = ISI[0]
            if ISI.shape[0] > 1:
                ISI_array[i]["last"] = ISI[-1]
                ISI_array[i]["ratio"] = ISI[-1] / ISI[0]
                ISI_array[i]["mean"] = np.mean(ISI)
                ISI_array[i]["std"] = np.std(ISI)
    return ISI_array


class ExperimentalProtocol:
    def __init__(self, params_dict, target_namespace=None):
        self.params_dict = params_dict
        self.target_namespace = target_namespace
        self._init_params(params_dict)

    def _init_params(self, params):
        num_config = params["Numerics"]
        self.t0 = num_config["t0"]
        self.tstop = num_config["tstop"]
        self.v_init = num_config["v_init"]
        self.use_cvode = num_config.get("adaptive", False)
        self.use_coreneuron = num_config.get("use_coreneuron", False)
        self.dt = num_config.get("dt", 0.01)
        self.record_dt = num_config.get("record_dt", self.dt)

        target_config = params["Targets"]
        if self.target_namespace is not None:
            target_config = params["Target namespaces"][self.target_namespace]

        self.v_hold = target_config["V_hold"]["val"]
        self.v_rest = target_config["V_rest"]["val"]

        Rin_config = target_config["Rin"]
        self.target_rn = (Rin_config["lower"][0], Rin_config["upper"][0])
        self.rn_exp_type = "vclamp" if "V" in Rin_config else "iclamp"

        tau0_config = target_config["tau0"]
        self.target_tau = (tau0_config["lower"][0], tau0_config["upper"][0])

        f_I_config = target_config["f_I"]
        N_exp = len(f_I_config["I"])
        self.exp_i_inj_amp_f_I = np.asarray(f_I_config["I"]) * f_I_config.get(
            "I_factor", 1.0
        )
        self.exp_i_inj_t0_f_I = f_I_config["t"][0]
        self.exp_i_inj_t1_f_I = f_I_config["t"][1]
        self.exp_i_mean_rate_f_I = (
            np.asarray(f_I_config["mean"]) if "mean" in f_I_config else None
        )
        self.exp_i_ub_rate_f_I = np.asarray(
            f_I_config.get("upper", f_I_config.get("mean"))
        )
        self.exp_i_lb_rate_f_I = np.asarray(
            f_I_config.get("lower", f_I_config.get("mean"))
        )

        spike_amp_config = target_config["spike_amp"]
        self.exp_i_ub_spk_amp = (
            np.asarray(spike_amp_config["upper"])
            if "upper" in spike_amp_config
            else None
        )
        self.exp_i_lb_spk_amp = (
            np.asarray(spike_amp_config["lower"])
            if "lower" in spike_amp_config
            else None
        )

        spike_adaptation_config = target_config["spike_adaptation"]
        self.exp_i_mean_spk_adaptation = (
            np.asarray(spike_adaptation_config["mean"])
            if "mean" in spike_adaptation_config
            else None
        )
        self.exp_i_ub_spk_adaptation = np.asarray(
            spike_adaptation_config.get(
                "upper", spike_adaptation_config.get("mean", np.zeros(N_exp))
            )
        )
        self.exp_i_lb_spk_adaptation = np.asarray(
            spike_adaptation_config.get(
                "lower", spike_adaptation_config.get("mean", np.zeros(N_exp))
            )
        )

        self.target_threshold = target_config["threshold"]

        constraint_config = target_config.get("Constraints", {})
        first_ISI_constraints = constraint_config.get("first_ISI", {})
        self.exp_first_ISI_mean = np.asarray(
            first_ISI_constraints.get("mean", np.zeros((N_exp,)))
        )
        self.exp_first_ISI_lower = np.asarray(
            first_ISI_constraints.get("lower", 0.5 * self.exp_first_ISI_mean)
        )
        self.exp_first_ISI_upper = np.asarray(
            first_ISI_constraints.get("upper", 1.5 * self.exp_first_ISI_mean)
        )

    def run_iclamp(self, cell, target, tstop=10000.0):
        target_config = self.params_dict["Targets"]
        if self.target_namespace is not None:
            target_config = self.params_dict["Target namespaces"][self.target_namespace]
        this_target_config = target_config[target]

        this_target_amp = this_target_config["I"][0] * this_target_config.get(
            "I_factor", 1.0
        )
        tstim = this_target_config.get("t", None)
        if tstim is None:
            t0 = tstop / 2.0
            t1 = t0 + 1000.0
        else:
            t0, t1 = tstim

        t, v = run_iclamp(
            cell, t0=t0, t1=t1, amp=this_target_amp, tstop=tstop, v_init=self.v_hold
        )
        return {"t": t, "v": v, "t0": t0, "t1": t1, "stim_amp": this_target_amp}

    def run_iclamp_steps(self, cell):
        Isteps = np.asarray(
            [
                (amp, self.exp_i_inj_t0_f_I, self.exp_i_inj_t1_f_I)
                for amp in self.exp_i_inj_amp_f_I
            ]
        )
        return run_iclamp_steps(
            cell,
            Isteps=Isteps,
            v_init=self.v_hold,
            record_dt=self.record_dt,
            tstop=self.tstop,
            use_cvode=self.use_cvode,
            use_coreneuron=self.use_coreneuron,
        )


def range_distance(x, lb, ub):
    """Return 0 if x is within [lb, ub], otherwise the distance to the nearest bound."""
    return 0.0 if (x >= lb) and (x <= ub) else min(abs(x - lb), abs(x - ub))


def init_cell(pp, v_hold=-60, celsius=36.0, ic_constant_val=None):
    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h.secondorder = 2
    h.celsius = celsius

    cell = BRK({"BoothRinzelKiehn": pp})

    h.v_init = v_hold
    h.init()

    if ic_constant_val is None:
        cell.init_ic(h.v_init)
        ic_constant_0 = cell.soma.ic_constant

        x0 = 0.0
        try:
            x0, res = optimize.brentq(
                ic_constant_f,
                -0.5,
                0.5,
                args=(BRK, pp, ic_constant_0, h.v_init),
                xtol=1e-6,
                maxiter=200,
                disp=False,
                full_output=True,
            )
        except ValueError:
            x0 = 0.0
        else:
            if not res.converged:
                x0 = 0.0

        ic_constant_val = ic_constant_0 + x0

    cell.soma.ic_constant = ic_constant_val
    h.finitialize(h.v_init)
    h.finitialize(h.v_init)
    return cell


def make_obj_fun(protocol_config_dict, feature_dtypes, target_namespace, worker):
    exp_protocol = ExperimentalProtocol(
        protocol_config_dict, target_namespace=target_namespace
    )
    return partial(obj_fun, exp_protocol, feature_dtypes)


def obj_fun(exp_protocol, feature_dtypes, pp):
    cell = init_cell(pp, v_hold=exp_protocol.v_hold)
    ic_constant_hold = cell.soma.ic_constant

    initial_v_error_hold = float(
        ic_constant_f(0.0, BRK, pp, ic_constant_hold, v_hold=exp_protocol.v_hold)
    )
    initial_v_constr = 1 if abs(initial_v_error_hold) < 1.0 else -1

    # Measure passive properties (input resistance, time constant)
    cell = init_cell(pp, v_hold=exp_protocol.v_hold, ic_constant_val=ic_constant_hold)

    rn, tau = np.nan, np.nan
    if initial_v_constr > 0:
        try:
            iclamp_results = exp_protocol.run_iclamp(cell, target="Rin", tstop=3000.0)
        except:
            pass
        else:
            passive_results = measure_passive(**iclamp_results)
            rn = passive_results["Rinp"]
            tau = passive_results["tau"]

    target_rn = exp_protocol.target_rn
    target_tau = exp_protocol.target_tau
    rn_obj_value = range_distance(rn, target_rn[0], target_rn[1]) ** 2
    tau_obj_value = range_distance(tau, target_tau[0], target_tau[1]) ** 2

    tau_constr = 1 if (tau > 0.0 and tau < 1000.0) else -1
    rn_constr = 1 if (rn > 0.0 and rn < 1000.0) else -1

    # Run f-I current steps
    cell = init_cell(pp, v_hold=exp_protocol.v_hold, ic_constant_val=ic_constant_hold)
    iclamp_results = exp_protocol.run_iclamp_steps(cell)

    # Measure spike features
    pre_spk_cnt, spk_cnt, spk_infos, thresholds, mean_spike_amplitudes = (
        measure_spike_features(
            iclamp_results,
            exp_protocol.exp_i_inj_t0_f_I,
            exp_protocol.exp_i_inj_t1_f_I + 2.0,
        )
    )
    pre_spk_count_constr = -1 if np.sum(pre_spk_cnt) > 0 else 1

    ISI_values = measure_ISI(exp_protocol.exp_i_inj_amp_f_I, spk_infos)

    ISI_adaptation_dists = list(
        map(
            lambda ratio, target_range: range_distance(
                ratio * 100.0, target_range[0] * 100.0, target_range[1] * 100.0
            ),
            ISI_values["ratio"],
            zip(
                exp_protocol.exp_i_lb_spk_adaptation,
                exp_protocol.exp_i_ub_spk_adaptation,
            ),
        )
    )
    ISI_adaptation_obj_value = np.mean([dist**2 for dist in ISI_adaptation_dists])
    ISI_adaptation_constr = -1 if np.isnan(ISI_adaptation_obj_value) else 1

    first_ISI_constr = (
        1 if np.all(ISI_values["first"] > exp_protocol.exp_first_ISI_lower) else -1
    )

    fI_values = measure_fI(
        spk_cnt,
        exp_protocol.exp_i_inj_t0_f_I,
        exp_protocol.exp_i_inj_t1_f_I,
        exp_protocol.exp_i_inj_amp_f_I,
    )

    fI_mean_target_rate_diff = np.mean(
        [
            (target_rate - rate) ** 2
            for rate, target_rate in zip(
                fI_values["frequency"], exp_protocol.exp_i_mean_rate_f_I
            )
        ]
    )

    fI_range_dists = list(
        map(
            lambda rate, target_range: range_distance(
                rate, target_range[0], target_range[1]
            ),
            fI_values["frequency"],
            zip(exp_protocol.exp_i_lb_rate_f_I, exp_protocol.exp_i_ub_rate_f_I),
        )
    )
    fI_obj_value = np.mean([dist**2 for dist in fI_range_dists])

    mean_spike_amplitude_range_dists = list(
        map(
            lambda amp, target_amp: None
            if np.isnan(target_amp[0])
            else range_distance(amp, target_amp[0], target_amp[1]),
            mean_spike_amplitudes,
            zip(exp_protocol.exp_i_lb_spk_amp, exp_protocol.exp_i_ub_spk_amp),
        )
    )
    mean_spike_amplitude_obj_value = np.mean(
        [
            dist**2
            for dist in filter(
                lambda x: x is not None, mean_spike_amplitude_range_dists
            )
        ]
    )
    spike_amplitude_constr = -1 if np.isnan(mean_spike_amplitude_obj_value) else 1

    fI_rate_diff = np.diff(fI_values["frequency"][:-1])
    monotonic_fI_constr = 1 if np.all(fI_rate_diff > 0) else -1

    # Obtain ic_constant for v_rest target
    cell = init_cell(pp, v_hold=exp_protocol.v_rest)
    ic_constant_rest = cell.soma.ic_constant

    # Assemble results
    feature_values = np.asarray(
        [
            (
                ic_constant_hold,
                ic_constant_rest,
                initial_v_error_hold,
                rn,
                tau,
                fI_values,
                fI_mean_target_rate_diff,
                ISI_values,
                thresholds,
                mean_spike_amplitudes,
            )
        ],
        dtype=np.dtype(feature_dtypes),
    )

    obj_values = np.array(
        [
            rn_obj_value,
            tau_obj_value,
            fI_obj_value,
            mean_spike_amplitude_obj_value,
            # ISI_adaptation_obj_value,
        ],
        dtype=np.float32,
    )

    constr_values = np.array(
        [
            monotonic_fI_constr,
            rn_constr,
            tau_constr,
            spike_amplitude_constr,
            first_ISI_constr,
            ISI_adaptation_constr,
            pre_spk_count_constr,
            initial_v_constr,
        ],
        dtype=np.float32,
    )

    return obj_values, feature_values, constr_values


OBJECTIVE_NAMES = [
    "rn_error",
    "tau_error",
    "fI_error",
    "spike_amplitude_error",
    # "ISI_adaptation_error",
]
CONSTRAINT_NAMES = [
    "monotonic_fI",
    "rn_constr",
    "tau_constr",
    "spike_amplitude_constr",
    "first_ISI_constr",
    "ISI_adaptation_constr",
    "pre_spk_count",
    "initial_v_constr",
]

script_name = os.path.basename(__file__)


@click.command()
@click.option(
    "--config-path",
    default="config/motoneuron.yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--results-path",
    "-p",
    default="results",
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option("--num-epochs", default=5, type=int)
@click.option("--num-initial", default=10, type=int)
@click.option("--population-size", default=100, type=int)
@click.option("--num-generations", default=10, type=int)
@click.option("--seed", default=None, type=int)
@click.option("--optimizer", default="nsga2", type=str)
@click.option("--verbose", "-v", is_flag=True)
def main(
    config_path,
    results_path,
    num_epochs,
    num_initial,
    population_size,
    num_generations,
    seed,
    optimizer,
    verbose,
):
    comm = MPI.COMM_WORLD
    protocol_config_dict = None
    if comm.rank == 0:
        with open(config_path) as f:
            protocol_config_dict = yaml.load(f, Loader=yaml.FullLoader)

    local_random = default_rng(seed=seed) if seed is not None else None
    protocol_config_dict = comm.bcast(protocol_config_dict)

    celltype = protocol_config_dict["Celltype"]

    N_exp = len(protocol_config_dict["Targets"]["f_I"]["I"])
    feature_dtypes = [
        ("ic_constant_hold", np.float32),
        ("ic_constant_rest", np.float32),
        ("initial_v_error_hold", np.float32),
        ("rn", np.float32),
        ("tau", np.float32),
        ("fI", fi_value_dtype, N_exp),
        ("mean_fI_diff", np.float32),
        ("ISI", isi_value_dtype, N_exp),
        ("threshold", np.float32, N_exp),
        ("spike_amplitude", np.float32, N_exp),
    ]

    problem_parameters = protocol_config_dict["Parameters"]
    space = protocol_config_dict["Space"]

    dmosopt_params = {
        "opt_id": f"dmosopt_{celltype}_neuron",
        "obj_fun_init_name": "make_obj_fun",
        "obj_fun_init_module": "example_dmosopt_motoneuron",
        "obj_fun_init_args": {
            "protocol_config_dict": protocol_config_dict,
            "feature_dtypes": feature_dtypes,
            "target_namespace": None,
        },
        "problem_parameters": problem_parameters,
        "space": space,
        "objective_names": OBJECTIVE_NAMES,
        "constraint_names": CONSTRAINT_NAMES,
        "feature_dtypes": feature_dtypes,
        # Optimizer
        "optimizer": optimizer,
        "population_size": population_size,
        "num_generations": num_generations,
        # Surrogate: joint model via custom training
        "surrogate_custom_training": "dmosopt.custom_training.joint",
        "surrogate_custom_training_kwargs": {
            "mode": "c+o",
            "epochs": "auto",
        },
        # Sampling
        "n_initial": num_initial,
        "n_epochs": num_epochs,
        "initial_maxiter": 10,
        "initial_method": "slh",
        "termination_conditions": True,
        # Output
        "file_path": f"{results_path}/dmosopt_{celltype}.h5",
        "save": True,
        "local_random": local_random,
    }

    best = dmosopt.run(dmosopt_params, verbose=True)

    if best is not None:
        import matplotlib.pyplot as plt

        bestx, besty = best
        besty_dict = dict(besty)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].scatter(
            besty_dict["rn_error"],
            besty_dict["fI_error"],
            c="tab:red",
            s=10,
            label="Pareto front",
        )
        axes[0].set_xlabel("Input resistance error")
        axes[0].set_ylabel("f-I error")
        axes[0].set_title("Objective trade-off")
        axes[0].legend()

        obj_values = np.column_stack([besty_dict[n] for n in OBJECTIVE_NAMES])
        axes[1].boxplot(obj_values, tick_labels=OBJECTIVE_NAMES)
        axes[1].set_ylabel("Error")
        axes[1].set_title("Objective distributions")
        plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        plt.savefig(f"{celltype}_results.svg")
        print(f"Results saved to {celltype}_results.svg")


if __name__ == "__main__":
    main(
        args=sys.argv[
            (
                next(
                    (
                        i
                        for i, x in enumerate(sys.argv)
                        if os.path.basename(x) == script_name
                    ),
                    0,
                )
                + 1
            ) :
        ]
    )
