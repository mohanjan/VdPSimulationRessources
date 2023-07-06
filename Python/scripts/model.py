# ------------------------------------------------------------------------------
# Global imports
# ------------------------------------------------------------------------------
from typing import Any
from typing import Dict
import inspect
import math
from functools import partial

# ======================================
# Numerical modules
# ======================================
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.fft import rfft, rfftfreq
import torch as pt

# ======================================
# Plotting
# ======================================
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display


class Cochlea:
    def __init__(
        self,
        N: int = None,
        masses: pt.Tensor = None,
        T_sim: float = None,
        sampling_frequency: float = None,
        spring_const_base: pt.Tensor = None,
        spring_const_coupling: pt.Tensor = None,
        damping_coupling: pt.Tensor = None,
        wall: bool = None,
        nonlinear_damping: str = "van_der_pol",
        nonlinear_damping_function_params: Dict[str, Any] = None,
        driving: str = "pulse",
        driving_function_params: Dict[str, Any] = None,
    ):
        # Number of oscillators and their masses:
        self.N = 100 if N is None else N
        self.masses = pt.ones(self.N) if masses is None else masses

        # Length of simulation and sampling frequency:
        self.T_sim = 100 if T_sim is None else T_sim
        self.sampling_frequency = (
            1000 if sampling_frequency is None else sampling_frequency
        )
        self.time = pt.linspace(
            0, self.T_sim, int(self.T_sim * self.sampling_frequency)
        )

        # Spring coupling strengths to base (dim N x 1):
        self.spring_const_base = (
            pt.zeros(self.N) if spring_const_base is None else spring_const_base
        )

        # Coupling strengths in network - springs and dampings (both dim N+1 x N):
        self.spring_const_coupling = (
            pt.zeros(self.N, self.N + 2)
            if spring_const_coupling is None
            else spring_const_coupling
        )
        self.damping_coupling = (
            pt.zeros(self.N, self.N + 2)
            if damping_coupling is None
            else damping_coupling
        )
        self.wall = True if wall is None else wall
        if self.wall is not True:
            self.spring_const_coupling[:, 0] = 0
            self.spring_const_coupling[:, -1] = 0
            self.damping_coupling[:, 0] = 0
            self.damping_coupling[:, -1] = 0

        # Activate chosen driving and nonlinear damping functions:
        (self.nonlinear_damping_function, self.driving_function,) = self._get_function(
            nonlinear_damping,
            driving,
            {}
            if nonlinear_damping_function_params is None
            else nonlinear_damping_function_params,
            {} if driving_function_params is None else driving_function_params,
        )

    def _get_function(
        self,
        chosen_nonlinear_damping: str,
        chosen_driving: str,
        nonlinear_damping_params: Dict[str, Any],
        driving_params: Dict[str, Any],
    ):

        nonlinear_damping_function = None
        driving_function = None

        for obj_name, obj in inspect.getmembers(self):
            if inspect.ismethod(obj):

                if obj_name == f"_nonlinear_damping_{chosen_nonlinear_damping}":
                    nonlinear_damping_function = partial(
                        obj, **nonlinear_damping_params
                    )
                    print(f"==[ Using nonlinear damping function '{obj_name}'")

                elif obj_name == f"_driving_function_{chosen_driving}":
                    driving_function = partial(obj, **driving_params)
                    print(f"==[ Using driving function '{obj_name}'")

        if nonlinear_damping_function is None:
            raise ValueError(
                f"Invalid nonlinear damping function {chosen_nonlinear_damping}"
            )

        if driving_function is None:
            raise ValueError(f"Invalid driving function {chosen_driving}")

        return (nonlinear_damping_function, driving_function)

    # ============================================================================
    # Different functions for nonlinear damping terms
    # ============================================================================

    def _system_equations(
        self,
        t: float,
        state: np.array,
    ):
        state = pt.from_numpy(state)
        diff_state = pt.zeros_like(state)

        # Add ghost nodes at boundary
        pos_buffer = pt.zeros(self.N + 2)
        pos_buffer[1 : self.N + 1] = state[: self.N]  # Add the walls at the end
        pos_diff = pos_buffer.unsqueeze(1) - pos_buffer  # Create distance matrix

        vel_buffer = pt.zeros(self.N + 2)
        vel_buffer[1 : self.N + 1] = state[self.N :]  # Add the walls at the end
        vel_diff = vel_buffer.unsqueeze(1) - vel_buffer  # Create distance matrix

        # Force from coupled oscillators
        F_couplings = (
            -self.spring_const_coupling * pos_diff[1:-1, :]
            - self.damping_coupling * vel_diff[1:-1, :]
        ).sum(dim=1)

        # For scaling:
        w_eigen = pt.sqrt(self.spring_const_base / self.masses)
        scale = w_eigen

        # Velocity:
        diff_state[0 : self.N] = state[self.N :] * scale

        # Acceleration:
        diff_state[self.N :] = (
            scale
            * (
                -state[: self.N]
                + self.nonlinear_damping_function(state[: self.N], state[self.N :])
                * state[self.N :]
            )
            + F_couplings
            + self.driving_function(t)
        )

        return diff_state.numpy()

    def solve(
        self,
        init_state: pt.Tensor = None,
    ):
        if init_state is None:
            init_state = pt.zeros(2 * self.N)

        sol = solve_ivp(
            self._system_equations,
            (0.0, self.T_sim),
            init_state.numpy(),
            t_eval=self.time,
            method="RK45",
        )

        sol.y = pt.from_numpy(np.array(sol.y, dtype=np.float32))

        self.times = sol.t
        self.sol = sol.y

        return (self.times, self.sol)

    # ============================================================================
    # Different functions for nonlinear damping terms
    # ============================================================================

    def _nonlinear_damping_van_der_pol(
        self,
        pos,
        vel,
        nonlinearity_factor: pt.Tensor,
        **kwargs,
    ):
        return nonlinearity_factor * (1 - pos**2)

    # def _nonlinear_damping_2nd_order(
    #     self,
    #     pos,
    #     vel,
    #     **kwargs,
    # ):
    #     return 0.3 * vel**2 - 5

    # def _nonlinear_damping_vilfan_duke(
    #     self,
    #     pos,
    #     vel,
    #     **kwargs,
    # ):
    #     return pos**3 + ((self.masses * vel**2) / self.spring_const_base) ** (3 / 2)

    # ============================================================================
    # Different functions for driving force terms
    # ============================================================================

    def _driving_function_pulse(
        self,
        t,
        driven_osc: pt.Tensor,
        driving_amplitudes: pt.Tensor,
        pulse_time: float,
        pulse_width: float,
        **kwargs,
    ):
        pulse_force = (
            driving_amplitudes
            if t - pulse_time < pulse_width
            else pt.zeros_like(driving_amplitudes)
        )
        return driven_osc * pulse_force

    def _driving_function_cosine(
        self,
        t,
        driven_osc: pt.Tensor,
        driving_amplitudes: pt.Tensor,
        driving_frequencies: pt.Tensor,
        **kwargs,
    ):
        cosine_force = driving_amplitudes * pt.cos(
            2 * math.pi * driving_frequencies * t
        )
        return driven_osc * cosine_force

    def _driving_function_external_input(
        self,
        t,
        driven_osc: pt.Tensor,
        input_signal: pt.Tensor,
        input_time: pt.Tensor,
        input_start: float,
        **kwargs,
    ):
        if t <= input_start:
            input_force = 0
        else:
            input_force = input_signal[np.where(input_time > (t - input_start))[0][0]]
        return driven_osc * input_force

    def _driving_function_sweep(
        self,
        t,
        driven_osc: pt.Tensor,
        driving_amplitudes: pt.Tensor,
        low_freq: float,
        high_freq: float,
        **kwargs,
    ):
        sweep_force = driving_amplitudes * math.sin(
            2 * math.pi * (low_freq + t / self.T_sim * (high_freq - low_freq)) * t
        )
        return driven_osc * sweep_force

    # ============================================================================
    # Functions for analysis of system
    # ============================================================================

    def find_fourier_freqs(
        self,
        t_analysis: float = None,
    ):
        # fourier analysis:
        t_analysis = 10 if t_analysis is None else t_analysis
        spec_start = int((self.T_sim - t_analysis) * self.sampling_frequency)
        spec = rfft(self.sol[: self.N, -spec_start:].numpy(), axis=1)
        spec = np.abs(spec[:, 1:])
        freq = rfftfreq(len(self.times[-spec_start:]), 1 / self.sampling_frequency)
        freq = freq[1:]
        spec_threshold = np.where(spec > 1, spec, 0)
        spec_peaks = np.argmax(spec_threshold, axis=1)
        self.fourier_max_freq = np.where(spec_peaks > 0, freq[spec_peaks], 0)

    def find_zc_freqs(
        self,
        t_analysis: float = None,
    ):
        # zero-crossing analysis:
        t_analysis = 10 if t_analysis is None else t_analysis
        self.zc_pos_freq = np.zeros(self.N)
        self.zc_vel_freq = np.zeros(self.N)
        for ii in range(self.N):
            cross_pos = np.where(pt.abs(pt.diff(pt.sign(self.sol[ii, :]))) > 1)[0][::2]
            cross_vel = np.where(
                pt.abs(pt.diff(pt.sign(self.sol[ii + self.N, :]))) > 1
            )[0][::2]
            self.zc_pos_freq[ii] = 1 / np.mean(
                np.diff(
                    self.times[
                        cross_pos[cross_pos > t_analysis * self.sampling_frequency]
                    ]
                )
            )
            self.zc_vel_freq[ii] = 1 / np.mean(
                np.diff(
                    self.times[
                        cross_vel[cross_vel > t_analysis * self.sampling_frequency]
                    ]
                )
            )
        return self.zc_pos_freq # TODO: shitty code I added 
        

    # ============================================================================
    # Functions for plotting
    # ============================================================================

    def plot_temporal_dynamics(
        self,
        fig: plt.figure = None,
        ax: np.ndarray = None,
        t_min: float = None,
        t_max: float = None,
    ):
        # Are fig and ax given or should it be set up?
        setup = True if fig is None else False

        fig = plt.figure() if setup else fig
        ax = fig.add_subplot() if setup else ax

        c_dyn = ax.pcolormesh(
            np.arange(self.N),
            self.times,
            self.sol[: self.N, :].transpose(1, 0),
            shading="auto",
        )
        t_min = self.T_sim - 5 if t_min is None else t_min
        t_max = self.T_sim if t_max is None else t_max
        ax.set(
            ylabel="Time",
            xlabel="Oscillator",
            title="Temporal dynamics",
            ylim=(t_min, t_max),
        )
        fig.colorbar(c_dyn, ax=ax, label="Level")

        if setup:
            fig.tight_layout()
            fig.show()

    def plot_spectrum(
        self,
        fig: plt.figure = None,
        ax: np.ndarray = None,
        t_fft: float = None,
        f_min: float = None,
        f_max: float = None,
    ):
        t_fft = 10 if t_fft is None else t_fft
        f_min = 0 if f_min is None else f_min
        f_max = 7 if f_max is None else f_max

        spec_start = int((self.T_sim - t_fft) * self.sampling_frequency)
        spec = rfft(self.sol[: self.N, -spec_start:].numpy(), axis=1)
        spec = np.abs(spec[:, 1:])
        freq = rfftfreq(len(self.times[-spec_start:]), 1 / self.sampling_frequency)
        freq = freq[1:]
        spec_threshold = np.where(spec > 1, spec, 0)
        spec_peaks = np.argmax(spec_threshold, axis=1)
        self.fourier_max_freq = np.where(spec_peaks > 0, freq[spec_peaks], 0)

        spec_scaled = spec.transpose(1, 0) / spec[np.arange(self.N), spec_peaks]

        # Are fig and ax given or should it be set up?
        setup = True if fig is None else False

        fig = plt.figure() if setup else fig
        ax = fig.add_subplot() if setup else ax

        c_spec = ax.pcolormesh(np.arange(self.N), freq, spec_scaled, shading="auto")
        ax.set(
            ylabel="Frequency",
            xlabel="Oscillator",
            title="Spectrum analysis",
            ylim=(f_min, f_max),
        )
        fig.colorbar(c_spec, ax=ax, label="Level")

        if setup:
            fig.tight_layout()
            fig.show()

    def plot_single_track(
        self,
        osc: np.ndarray = None,
        fig: plt.figure = None,
        ax: np.ndarray = None,
    ):
        # Are fig and ax given or should it be set up?
        setup = True if fig is None else False

        fig = plt.figure() if setup else fig
        ax = fig.add_subplot() if setup else ax

        osc = np.array([0, self.N - 1]) if osc is None else osc

        for osc_plot in osc:
            ax.plot(
                self.times,
                self.sol[osc_plot, :],
                label=("Position oscillator: " + str(osc_plot)),
            )
            ax.plot(
                self.times,
                self.sol[self.N + osc_plot, :],
                label=("Velocity oscillator: " + str(osc_plot)),
            )
            ax.set(xlabel="Time", ylabel="Level", xlim=(self.T_sim - 10, self.T_sim))

        ax.legend(loc="lower left")
        if setup:
            fig.tight_layout()
            fig.show()

    def plot_phase_space_trajectory(
        self,
        fig: plt.figure = None,
        ax: np.ndarray = None,
    ):
        # Are fig and ax given or should it be set up?
        setup = True if fig is None else False

        fig = plt.figure() if setup else fig
        ax = fig.add_subplot() if setup else ax

        plot_start = int(2 * self.sampling_frequency)
        for ii in np.arange(self.N)[::10]:
            ax.plot(
                self.sol[ii, -plot_start:],
                self.sol[self.N + ii, -plot_start:],
                label=str(ii),
                alpha=0.5,
            )
        ax.set(xlabel="Position", ylabel="Velocity")

        ax.legend(loc="lower left")
        if setup:
            fig.tight_layout()
            fig.show()

    def plot_animated_phase_space_trajectory(
        self,
        frames: int = 100,
        fig: plt.figure = None,
        ax: np.ndarray = None,
    ):
        from matplotlib.animation import FuncAnimation
        from IPython import display

        fig = plt.figure()
        ax = fig.add_subplot()

        plot_start = int(2 * self.sampling_frequency)

        def animate(i):
            ax.clear()
            t = i * 10
            for ii in np.arange(self.N)[::10]:
                ax.plot(
                    self.sol[ii, -plot_start:],
                    self.sol[self.N + ii, -plot_start:],
                    alpha=0.5,
                )
                ax.plot(
                    self.sol[ii, -plot_start + t],
                    self.sol[self.N + ii, -plot_start + t],
                    "o",
                    label=str(ii),
                )
            ax.legend(loc="lower left")
            ax.set(xlabel="Position", ylabel="Velocity")

        # run the animation
        ani = FuncAnimation(
            fig,
            animate,
            frames=frames,
            interval=100,
            repeat=False,
        )

        video = ani.to_jshtml()
        html_code = display.HTML(video)
        display.display(html_code)
