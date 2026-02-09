import FFTW

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val
const EPS0 = VacuumElectricPermittivity.val
const QE = ElementaryCharge.val
const ME = ElectronMass.val
const HBAR = ReducedPlanckConstant.val


function waveform(t, p)
    w0, tau0, dt0 = p

    # monochromatic:
    # t < dt0 ? A0 = exp(-(t - dt0)^2 / tau0^2) : A0 = 1
    # T = A0 * cos(w0 * (t - dt0))


    # gaussian pulse:
    # T = exp(-(t - dt0)^2 / tau0^2) * cos(w0 * (t - dt0))

    # sin2 pulse:
    # lm = 1   # FWHM = tau0/2
    # lm = 2   # FWHM = tau0
    lm = 4   # FWHM = 2*tau0
    T = t > 2*tau0 ? 0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * cos(w0 * (t - 2*tau0))

    return T
end


lam0 = 2e-6   # (m) wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 5*tau0   # (s) delay time for source injection
w0 = 2*pi * C0 / lam0   # frequency
p = (w0, tau0, dt0)

t = range(0, 500e-15, 5451)
Nt = length(t)
dt = t[2] - t[1]

I0 = 5e13*1e4
E0 = sqrt(I0 / (1 * EPS0 * C0 / 2))

Et = @. E0 * waveform(t, (p,))


# ******************************************************************************************
# Plasma ...................................................................................
rho0 = 2.5e25
ionrate(I) = 8.85e-105 * I^6.5

rho = zeros(Nt)
for it=1:Nt-1
    II = 1 * EPS0 * C0 / 2 * abs2(Et[it])
    RI = ionrate(II)
    rho[it+1] = rho0 - (rho0 - rho[it]) * exp(-RI * dt)
end

drho = zeros(Nt)
for it=1:Nt
    II = 1 * EPS0 * C0 / 2 * abs2(Et[it])
    RI = ionrate(II)
    drho[it] = RI * (rho0 - rho[it])
end


# Plasma current ...........................................................................
nuc = 5e12
w = 2*pi * FFTW.fftfreq(Nt, 1/dt)

F = @. rho * Et
F = FFTW.ifft(F)
Jp1 = @. QE^2/ME * (nuc + 1im*w) / (nuc^2 + w^2) * F
Jp1 = FFTW.fft(Jp1)
Jp1 = @. real(Jp1)

Jp2 = zeros(Nt)
for it=1:Nt-1
    Jp2[it+1] = (1 + nuc*dt) * Jp2[it] + QE^2/ME*dt * rho[it] * Et[it]
end

Ap = 2 / (nuc*dt/2 + 1)
Bp = (nuc*dt/2 - 1) / (nuc*dt/2 + 1)
Cp = QE^2/ME*dt^2 / (nuc*dt/2 + 1)
Pp = zeros(Nt)
for it=2:Nt-1
    Pp[it+1] = Ap * Pp[it] + Bp * Pp[it-1] + Cp * rho[it] * Et[it]
end
Jp3 = zeros(Nt)
for it=2:Nt-1
    Jp3[it] = (Pp[it+1] - Pp[it-1]) / (2*dt)
end


# Multiphoton losses current ...............................................................
Ui = 12.063 * QE
Wph = HBAR * w0
K = ceil(Ui / Wph)

Ja1 = zeros(Nt)
for it=1:Nt
    It = abs2(Et[it])
    if It >= 1e-30
        invI = 1 / It
    else
        invI = 0
    end
    Ja1[it] = K*Wph * drho[it] * Et[it] * invI
end

Pa = zeros(Nt)
for it=1:Nt-1
    It = abs2(Et[it])
    if It >= 1e-30
        invI = 1 / It
    else
        invI = 0
    end
    Pa[it+1] = Pa[it] + K*Wph * dt * drho[it] * Et[it] * invI
end
Ja2 = zeros(Nt)
for it=2:Nt-1
    Ja2[it] = (Pa[it+1] - Pa[it-1]) / (2*dt)
end




# ******************************************************************************************
import PyPlot as plt

plt.figure(constrained_layout=true)

# plt.plot(t/1e-15, Et ./ maximum(Et) .* maximum(rho))
plt.plot(t/1e-15, abs2.(Et) ./ maximum(abs2, Et) .* maximum(rho))
plt.plot(t/1e-15, rho)
plt.plot(t/1e-15, drho ./ maximum(drho) * maximum(rho))
plt.xlim(0, 100)

# plt.plot(t/1e-15, Et ./ maximum(Et) .* maximum(Jp1), "k-")
# plt.plot(t/1e-15, Jp1)
# plt.plot(t/1e-15, Jp2)
# plt.plot(t/1e-15, Jp3)
# # plt.plot(t/1e-15, Pp)

# plt.plot(t/1e-15, Et ./ maximum(Et) .* maximum(Ja1))
# plt.plot(t/1e-15, Ja1)
# plt.plot(t/1e-15, Ja2)
# # plt.plot(t/1e-15, Pa)

plt.xlim(0, 100)
