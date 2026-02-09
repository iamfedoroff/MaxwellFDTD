# ******************************************************************************************
# Difference between a sin² pulse and a Gaussian pulse
# ******************************************************************************************

import FFTW
import PyPlot as plt

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

lam0 = 1e-6
w0 = 2*pi * C0 / lam0

tau0 = 20e-15   # FWHM pulse duration

tmin, tmax, Nt = -5*tau0, 5*tau0, 1024
t = range(tmin, tmax, Nt)
dt = t[2] - t[1]
w = 2*pi * FFTW.rfftfreq(Nt, 1/dt)


# sin2 pulse -------------------------------------------------------------------------------
E_sin2 = @. sin(pi/2 * (t - tau0) / tau0)^2 * cos(w0 * t)
E_sin2[@. abs(t) > tau0] .= 0.0


S_sin2 = FFTW.rfft(FFTW.fftshift(E_sin2))
@. S_sin2 = 2 * S_sin2 * dt

# Gaussian pulse ---------------------------------------------------------------------------
tau_einv = tau0 / (2 * sqrt(log(2)))
E_gauss = @. exp(-t^2 / tau_einv^2) * cos(w0 * t)

S_gauss = FFTW.rfft(FFTW.fftshift(E_gauss))
@. S_gauss = 2 * S_gauss * dt


# Plotting ---------------------------------------------------------------------------------
fig, paxes = plt.subplots(2,1; constrained_layout=true)
ax = paxes[1]
ax.plot(t/1e-15, E_gauss, label="Gaussian")
ax.plot(t/1e-15, E_sin2, label="sin²")
ax.grid()
ax.legend()
ax.set_xlabel("Time (fs)")
ax.set_ylabel("Electric field E(t)")

ax = paxes[2]
ax.plot(w/w0, abs.(S_gauss), label="Gaussian")
ax.plot(w/w0, abs.(S_sin2), label="sin²")
ax.grid()
ax.legend()
ax.set_xlabel("Frequency (ω/ω₀)")
ax.set_ylabel("Spectral amplitude |S(ω)|")
ax.set_xlim(0,2)
