import GLMakie as mak
import HDF5
import Printf: @sprintf
import PyPlot as plt


# ******************************************************************************
# 1D
# ******************************************************************************
function plot_1d(fname, svar; vmin=-1, vmax=1, cmap=:seismic)
    fp = HDF5.h5open(fname, "r")
    z = HDF5.read(fp, "z")
    t = HDF5.read(fp, "t")
    F = HDF5.read(fp, svar)
    HDF5.close(fp)

    plt.figure(constrained_layout=true)
    plt.pcolormesh(t, z, F; vmin, vmax, cmap)
    plt.xlabel("t (s)")
    plt.ylabel("z (m)")
    plt.colorbar()
    plt.show()
    return nothing
end


function inspect1D(fname, svar; vmin=-1, vmax=1)
    fp = HDF5.h5open(fname, "r")
    z = HDF5.read(fp, "z")
    t = HDF5.read(fp, "t")
    F = HDF5.read(fp, svar)
    HDF5.close(fp)

    fig = mak.Figure(resolution=(950,992), fontsize=14)
    ax = mak.Axis(fig[1,1]; xlabel="z", ylabel=svar)
    mak.display(fig)

    it = 1
    line = mak.lines!(ax, z, F[:,it])
    mak.ylims!(ax, (vmin,vmax))
    ax.title[] = "$(it)"

    sg = mak.SliderGrid(fig[2,1], (label="Time", range=1:length(t), startvalue=1))
    mak.on(sg.sliders[1].value) do it
        line[2] = F[:,it]
        ax.title[] = "$(it)"
    end
    return nothing
end



# ******************************************************************************
# 2D
# ******************************************************************************
function plot_2d(fname, svar, t0; vmin=-1, vmax=1, cmap=:seismic)
    fp = HDF5.h5open(fname, "r")
    x = HDF5.read(fp, "x")
    y = HDF5.read(fp, "y")
    t = HDF5.read(fp, "t")
    F = HDF5.read(fp, svar)
    HDF5.close(fp)

    it = argmin(abs.(t .- t0))

    plt.figure(constrained_layout=true)
    plt.pcolormesh(x, y, transpose(F[:,:,it]); vmin, vmax, cmap)
    plt.xlabel("t (s)")
    plt.ylabel("x (m)")
    plt.colorbar()
    plt.show()
    return nothing
end


function inspect2D(
    fname, svar; xu=1, yu=1, tu=1, vmin=-1, vmax=1, cmap=:seismic, aspect=1,
)
    fp = HDF5.h5open(fname, "r")
    x = HDF5.read(fp, "x")
    y = HDF5.read(fp, "y")
    t = HDF5.read(fp, "t")
    F = HDF5.read(fp, svar)
    HDF5.close(fp)

    @. x = x / xu
    @. y = y / yu
    @. t = t / tu
    sxu = space_units_string(xu)
    syu = space_units_string(yu)
    stu = time_units_string(tu)

    fig = mak.Figure(resolution=(950,992), fontsize=14)
    ax = mak.Axis(fig[1,1]; xlabel="x ($sxu)", ylabel="y ($syu)", aspect)
    mak.display(fig)

    it = 1
    hm = mak.heatmap!(
        ax, x, y, F[:,:,it]; colormap=cmap, colorrange=(vmin,vmax),
    )
    mak.Colorbar(fig[2,1], hm; vertical=false, label=svar)
    ax.title[] = @sprintf("%d:     %.3f (%s)", it, t[it], stu)

    sg = mak.SliderGrid(fig[3,1], (label="Time", range=1:length(t), startvalue=1))
    mak.on(sg.sliders[1].value) do it
        hm[3] = F[:,:,it]
        ax.title[] = @sprintf("%d:     %.3f (%s)", it, t[it], stu)
    end
    return nothing
end


function inspect2D_xsec(fname, svar, x0, y0; vmin=-1, vmax=1)
    fp = HDF5.h5open(fname, "r")
    x = HDF5.read(fp, "x")
    y = HDF5.read(fp, "y")
    t = HDF5.read(fp, "t")
    F = HDF5.read(fp, svar)
    HDF5.close(fp)

    ix0 = argmin(abs.(x .- x0))
    iy0 = argmin(abs.(y .- y0))

    fig = mak.Figure(resolution=(950,992), fontsize=14)
    ax1 = mak.Axis(fig[1,1]; xlabel="x", ylabel=svar)
    ax2 = mak.Axis(fig[2,1]; xlabel="y", ylabel=svar)
    mak.display(fig)

    it = 1
    line1 = mak.lines!(ax1, x, F[:,iy0,it])
    line2 = mak.lines!(ax2, y, F[ix0,:,it])
    mak.ylims!(ax1, (vmin,vmax))
    mak.ylims!(ax2, (vmin,vmax))
    ax1.title[] = "$(it)"

    sg = mak.SliderGrid(fig[3,1], (label="Time", range=1:length(t), startvalue=1))
    mak.on(sg.sliders[1].value) do it
        line1[2] = F[:,iy0,it]
        line2[2] = F[ix0,:,it]
        ax1.title[] = "$(it)"
    end
    return nothing
end


# ******************************************************************************
# 3D
# ******************************************************************************
function plot_3d(fname, svar, t0; vmin=-1, vmax=1, cmap=:seismic)
    fp = HDF5.h5open(fname, "r")
    x = HDF5.read(fp, "x")
    y = HDF5.read(fp, "y")
    z = HDF5.read(fp, "z")
    t = HDF5.read(fp, "t")
    F = HDF5.read(fp, svar)
    HDF5.close(fp)

    ix = div(length(x), 2)
    iy = div(length(y), 2)
    iz = div(length(z), 2)
    it = argmin(abs.(t .- t0))

    fig, paxes = plt.subplots(1,3; constrained_layout=true, figsize=(15,5))

    ax = paxes[1]
    ax.pcolormesh(x, y, transpose(F[:,:,iz,it]); vmin, vmax, cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = paxes[2]
    ax.pcolormesh(x, z, transpose(F[:,iy,:,it]); vmin, vmax, cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("$(it)")

    ax = paxes[3]
    ax.pcolormesh(y, z, transpose(F[ix,:,:,it]); vmin, vmax, cmap)
    ax.set_xlabel("y")
    ax.set_ylabel("z")

    plt.show()

    return nothing
end


function inspect3D(
    fname, svar; xu=1, yu=1, zu=1, tu=1, vmin=-1, vmax=1, cmap=:seismic,
)
    fp = HDF5.h5open(fname, "r")
    x = HDF5.read(fp, "x")
    y = HDF5.read(fp, "y")
    z = HDF5.read(fp, "z")
    t = HDF5.read(fp, "t")
    F = HDF5.read(fp, svar)
    HDF5.close(fp)

    @. x = x / xu
    @. y = y / yu
    @. z = z / zu
    @. t = t / tu
    sxu = space_units_string(xu)
    syu = space_units_string(yu)
    szu = space_units_string(zu)
    stu = time_units_string(tu)

    fig = mak.Figure(resolution=(950,992), fontsize=14)
    ax = mak.Axis3(
        fig[1,1];
        aspect=:data, perspectiveness=0,
        xlabel="x ($sxu)", ylabel="y ($syu)", zlabel="z ($szu)",
    )
    mak.display(fig)

    cmap = mak.to_colormap(cmap)
    Nmid = div(length(cmap),2)
    # cmap[Nmid+1] = mak.RGBAf(0,0,0,0)
    @. cmap[Nmid:Nmid+2] = mak.RGBAf(0,0,0,0)

    it = 1
    img = mak.volume!(
        ax, x, y, z, F[:,:,:,it];
        colormap=cmap, colorrange=(vmin,vmax),
        algorithm=:absorption, absorption=4f0,
    )
    mak.Colorbar(fig[1,2], img; label=svar)
    ax.title[] = @sprintf("%d:     %.3f (%s)", it, t[it], stu)

    sg = mak.SliderGrid(fig[2,1], (label="Time", range=1:length(t), startvalue=1))
    mak.on(sg.sliders[1].value) do it
        img[4] = F[:,:,:,it]
        ax.title[] = @sprintf("%d:     %.3f (%s)", it, t[it], stu)
    end
    return nothing
end


function inspect3D_xsec(
    fname, svar; xu=1, yu=1, zu=1, tu=1, vmin=-1, vmax=1, cmap=:seismic,
)
    fp = HDF5.h5open(fname, "r")
    x = HDF5.read(fp, "x")
    y = HDF5.read(fp, "y")
    z = HDF5.read(fp, "z")
    t = HDF5.read(fp, "t")
    F = HDF5.read(fp, svar)
    HDF5.close(fp)

    @. x = x / xu
    @. y = y / yu
    @. z = z / zu
    @. t = t / tu
    sxu = space_units_string(xu)
    syu = space_units_string(yu)
    szu = space_units_string(zu)
    stu = time_units_string(tu)

    ix, iy, iz = (div(length(p),2) for p in (x,y,z))

    fig = mak.Figure(resolution=(1600,600), fontsize=14)
    ax1 = mak.Axis(fig[1,1]; xlabel="x ($sxu)", ylabel="y ($syu)")
    ax2 = mak.Axis(fig[1,2]; xlabel="x ($sxu)", ylabel="z ($szu)")
    ax3 = mak.Axis(fig[1,3]; xlabel="y ($syu)", ylabel="z ($szu)")
    mak.display(fig)

    colormap = cmap
    colorrange = (vmin,vmax)

    it = 1
    hm1 = mak.heatmap!(ax1, x, y, F[:,:,iz,it]; colormap, colorrange)
    hm2 = mak.heatmap!(ax2, x, z, F[:,iy,:,it]; colormap, colorrange)
    hm3 = mak.heatmap!(ax3, y, z, F[ix,:,:,it]; colormap, colorrange)

    ax2.title[] = @sprintf("%d:     %.3f (%s)", it, t[it], stu)
    mak.Colorbar(fig[2,3], hm1; label=svar, vertical=false)

    sg = mak.SliderGrid(fig[2,1:2], (label="Time", range=1:length(t), startvalue=1))
    mak.on(sg.sliders[1].value) do it
        hm1[3] = F[:,:,iz,it]
        hm2[3] = F[:,iy,:,it]
        hm3[3] = F[ix,:,:,it]
        ax2.title[] = @sprintf("%d:     %.3f (%s)", it, t[it], stu)
    end
    return nothing
end



function movie3D(fname, svar; vmin=-1, vmax=1, cmap=:seismic)
    fp = HDF5.h5open(fname, "r")
    x = HDF5.read(fp, "x")
    y = HDF5.read(fp, "y")
    z = HDF5.read(fp, "z")
    t = HDF5.read(fp, "t")
    F = HDF5.read(fp, svar)
    HDF5.close(fp)

    fig = mak.Figure(resolution=(950,992), fontsize=14)
    ax = mak.Axis3(
        fig[1,1];
        aspect=:data, perspectiveness=0,
        xlabel="x", ylabel="y", zlabel="z",
    )
    mak.display(fig)

    cmap = mak.to_colormap(cmap)
    Nmid = div(length(cmap),2)
    # cmap[Nmid+1] = mak.RGBAf(0,0,0,0)
    @. cmap[Nmid:Nmid+2] = mak.RGBAf(0,0,0,0)

    it = 1
    img = mak.volume!(
        ax, x, y, z, F[:,:,:,it];
        colormap=cmap, colorrange=(vmin,vmax),
        algorithm=:absorption, absorption=4f0,
    )
    mak.Colorbar(fig[1,2], img; label=svar)
    ax.title[] = "$(it)"


    ext = splitext(fname)[end]
    fname_movie = replace(fname, ext => ".mp4")
    mak.record(fig, fname_movie, 1:length(t)) do it
        img[4] = F[:,:,:,it]
        ax.title[] = "$(it)"
        # ax.azimuth[] = 1.7pi + 0.3 * sin(2pi * it / 120)
        # ax.elevation[] = 1.7pi + 0.3 * sin(2pi * it / 120)
    end
    return nothing
end


# ******************************************************************************
# Util
# ******************************************************************************
function space_units_string(xu)
    sxu = "arb. u."
    if xu == 1
        sxu = "m"
    elseif xu == 1e-2
        sxu = "cm"
    elseif  xu == 1e-3
        sxu = "mm"
    elseif xu == 1e-6
        sxu = "um"
    elseif sxu == 1e-9
        sxu = "nm"
    end
    return sxu
end


function time_units_string(tu)
    stu = "arb. u."
    if tu == 1
        stu = "s"
    elseif  tu == 1e-3
        stu = "ms"
    elseif tu == 1e-6
        stu = "us"
    elseif tu == 1e-9
        stu = "ns"
    elseif tu == 1e-12
        stu = "ps"
    elseif tu == 1e-15
        stu = "fs"
    elseif tu == 1e-18
        stu = "as"
    end
    return stu
end
