using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using StaticArrays
import CUDA
@init_parallel_stencil(CUDA, Float64, 2)
include("StokesSolver_multixpu.jl")

using Plots, Plots.Measures, ImplicitGlobalGrid, MAT
using Test
import Random, MPI

default(size=(1200, 1000), framestyle=:box, label=false, grid=false, margin=10mm)

"""
    StokesFlow2D(; Nt=20, Nx=35, Ny=45, RAND_MARKER_POS::Bool=true, do_plot::Bool=true, print_info::Bool=true)

Input args:
Nt              : number of timesteps
Nx, Ny          : number of grid points
RAND_MARKER_POS : whether to add random perturbation to initial marker coords
do_plot         : whether to create Plots
print_info      : whether any info is printed to console

Output: Currently just Vy, an array of size (Nx, Ny+1)
"""
@views function StokesFlow2D(; Nt=20, Nx=35, Ny=45, RAND_MARKER_POS::Bool=true, do_plot::Bool=true, print_info::Bool=true)
    rank, dims, nprocs, coords, comm_cart = init_global_grid(Nx, Ny, 1)

    ENV["GKSwstype"] = "nul"
    (rank == 0) && (!isdir("viz_out")) && mkdir("viz_out")

    # --- PARAMETERS ---
    # time
    maxdisp = 0.5                               # dt is determined s.t. no marker moves further than maxdisp cells
    # physical parameters
    g_y = 9.81                                  # earth gravity
    lx, ly = 10, 10                             # global domain size
    lxl, lyl = lx / dims[1], ly / dims[2]       # local domain size
    μ_air, μ_matrix, μ_plume = 1e-2, 1e0, 1e-1  # Viscosity
    ρ_air, ρ_matrix, ρ_plume = 1e-3, 3.3, 3.2   # Density, kg/m^3
    plume_x, plume_y = lx / 2, ly / 2           # plume midpoint
    plume_r = ly / 5                            # plume radius
    air_height = 0.2 * ly                       # height of the 'sticky air' layer on top
    # discretization parameters
    marker_density = 5                          # use this amount of markers per grid step per dimension
    # random numbers for initial marker postions
    Random.seed!(rank)                          # seed default RNG for random marker positions

    # derived quantities
    dx = lx / (dims[1] * (Nx - 1))              # global grid resolution
    dy = ly / (dims[2] * (Ny - 1))              # note: cannot use nx_g(), ny_g() here because Nx, Ny means grid points, not cells
    Nmx = (Nx - 1) * marker_density             # local number of markers
    Nmy = (Ny - 1) * marker_density
    Nm = Nmx * Nmy

    # --- ARRAYS ---
    # marker CPU arrays for easy initializing with IC
    x_m = zeros(Nm)                             # marker x coords
    y_m = zeros(Nm)                             # marker y coords
    ρ_m = zeros(Nm)                             # marker property: density
    μ_m = zeros(Nm)                             # marker property: viscosity

    # grid array allocations
    # maybe Vx & Vy sizes need adjustments for marker interpolation (multi-GPU case, simplify single GPU). TODO
    P = @zeros(Nx - 1, Ny - 1)
    Vx = @zeros(Nx, Ny + 1)                     # Velocity in x-direction
    Vy = @zeros(Nx + 1, Ny)                     # Velocity in y-direction
    ρ_vy = @zeros(Nx + 1, Ny)                   # Density on vy-nodes
    μ_b = @zeros(Nx, Ny)                        # Viscosity μ on basic nodes
    μ_p = @zeros(Nx - 1, Ny - 1)                # Viscosity μ on pressure nodes

    # additional arrays for Stokes Solver
    τxx = @zeros(Nx - 1, Ny - 1)
    τyy = @zeros(Nx - 1, Ny - 1)
    τxy = @zeros(Nx, Ny)
    ∇V = @zeros(Nx - 1, Ny - 1)
    dτPt = @zeros(Nx - 1, Ny - 1)
    Rx = @zeros(Nx - 2, Ny - 1)
    Ry = @zeros(Nx - 1, Ny - 2)
    dVxdτ = @zeros(Nx - 2, Ny - 1)
    dVydτ = @zeros(Nx - 1, Ny - 2)
    dτVx = @zeros(Nx - 2, Ny - 1)
    dτVy = @zeros(Nx - 1, Ny - 2)

    # additional arrays for marker -> grid interpolation
    interp_size = max.(size(ρ_vy), size(μ_b), size(μ_p)) # these represent the arrays that are filled by interpolation
    val_wt_sum = @zeros(interp_size)
    wt_sum = @zeros(interp_size)

    # coordinates for all grid points
    x = [(ix - 1) * dx for ix = 1:Nx]              # basic nodes
    y = [(iy - 1) * dy for iy = 1:Ny]
    x_p = [(ix - 1) * dx + 0.5dx for ix = 1:Nx-1]  # pressure nodes
    y_p = [(iy - 1) * dy + 0.5dy for iy = 1:Ny-1]
    x_vx = x                                       # Vx nodes
    y_vx = [(iy - 1) * dy - 0.5dy for iy = 1:Ny+1]
    x_vy = [(ix - 1) * dx - 0.5dx for ix = 1:Nx+1] # Vy nodes
    y_vy = y
    # consistency checks
    @assert size(x_p, 1) == size(P, 1) && size(y_p, 1) == size(P, 2)
    @assert size(x_vx, 1) == size(Vx, 1) && size(y_vx, 1) == size(Vx, 2)
    @assert size(x_vy, 1) == size(Vy, 1) && size(y_vy, 1) == size(Vy, 2)
    @assert size(ρ_vy) == size(Vy)

    # --- INITIAL CONDITIONS ---
    setInitialMarkerCoords!(x_m, y_m, Nmx, Nmy, x, y, RAND_MARKER_POS::Bool)
    setInitialMarkerProperties!(coords, lxl, lyl, x_m, y_m, ρ_m, μ_m, Nm, μ_air, μ_matrix, μ_plume, ρ_air, ρ_matrix, ρ_plume, plume_x, plume_y, plume_r, air_height)
    if do_plot
        saveStats!(dims, Nt)
        saveMarkers!(0, rank, coords, [lxl, lyl], x_m, y_m, ρ_m)
        saveGrid!(0, rank, x, y, μ_b, x_p, y_p, P, x_vx, y_vx, Vx, x_vy, y_vy, Vy)
    end
    # transform marker arrays to xPU arrays
    x_m = Data.Array(x_m)
    y_m = Data.Array(y_m)
    ρ_m = Data.Array(ρ_m)
    μ_m = Data.Array(μ_m)

    # --- TIMESTEPPING ---
    times = zeros(4)

    dt = 0.0
    t_tot = 0.0
    for t = 1:Nt
        if print_info
            (rank == 0) && @show t, t_tot
        end

        # interpolate material properties to grid
        t1 = @elapsed begin
            # TODO: multi-xpu interpolation
            bilinearMarkerToGrid!(x_vy[1], y_vy[1], ρ_vy, x_m, y_m, ρ_m, dx, dy, val_wt_sum, wt_sum)
            bilinearMarkerToGrid!(x[1], y[1], μ_b, x_m, y_m, μ_m, dx, dy, val_wt_sum, wt_sum)
            bilinearMarkerToGrid!(x_p[1], y_p[1], μ_p, x_m, y_m, μ_m, dx, dy, val_wt_sum, wt_sum)
        end

        # calculate velocities on grid
        t2 = @elapsed begin
            dt = solveStokes!(P, Vx, Vy, ρ_vy, μ_b, μ_p,
                τxx, τyy, τxy, ∇V, dτPt, Rx, Ry, dVxdτ, dVydτ, dτVx, dτVy,
                g_y, dx, dy, Nx, Ny,
                dt, maxdisp, comm_cart; use_free_surface_stabilization=true,
                ϵ=1e-5,
                print_info=print_info && rank == 0)
        end

        # move markers
        t3 = @elapsed begin
            @parallel (1:Nm) moveMarkersRK4!(x_m, y_m, Vx, Vy, x_vx[1], y_vx[1], x_vy[1], y_vy[1], dt, lx, ly, dx, dy)
        end

        # exchange markers
        t4 = @elapsed begin
            x_m, y_m, ρ_m, μ_m = exchangeMarkers!(comm_cart, dims, [lxl, lyl], Array(x_m), Array(y_m), Array(ρ_m), Array(μ_m))
            # transform marker arrays to xPU arrays
            x_m = Data.Array(x_m)
            y_m = Data.Array(y_m)
            ρ_m = Data.Array(ρ_m)
            μ_m = Data.Array(μ_m)
        end

        # plot current state
        if do_plot
            saveMarkers!(t, rank, coords, [lxl, lyl], x_m, y_m, ρ_m)
            saveGrid!(t, rank, x, y, μ_b, x_p, y_p, P, x_vx, y_vx, Vx, x_vy, y_vy, Vy)
        end

        if t > 2
            times[1] += t1
            times[2] += t2
            times[3] += t3
            times[4] += t4
        end

        t_tot += dt
    end

    timesum = sum(times)
    times .= times ./ timesum .* 100
    if print_info
        (rank == 0) && println("\nTime Summary: Total ", timesum, " seconds\n",
            "  MarkerToGrid   : ", times[1], " %\n",
            "  StokesSolver   : ", times[2], " %\n",
            "  MoveMarkers    : ", times[3], " %\n",
            "  exchangeMarkers: ", times[4], " %\n")
    end

    return Array(Vx)
end

"""
    bilinearInterp(v1, v2, v3, v4, dxij, dyij)

Performs bilinear interpolation using repeated linear interpolation
"""
function bilinearInterp(v1, v2, v3, v4, dxij, dyij)
    s1 = (1 - dxij) * (1 - dyij) * v1
    s2 = dxij * (1 - dyij) * v2
    s3 = (1 - dxij) * dyij * v3
    s4 = dxij * dyij * v4
    return s1 + s2 + s3 + s4
end

"""
    interpolateV(x, y, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dx, dy)

Interpolates grid velocities to 2D positions
"""
function interpolateV(x, y, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dx, dy)
    # Interpolate Vx
    ix, iy, dxij, dyij = topleftIndexRelDist(x_vx_min, y_vx_min, x, y, dx, dy)
    #index range failsafe, in case advection moves the particles out of domain
    if ix < 1
        ix = 1
        dxij = 0.0
    end
    if iy < 1
        iy = 1
        dyij = 0.0
    end
    if ix >= size(Vx, 1)
        ix = size(Vx, 1) - 1
        dxij = 1.0
    end
    if iy >= size(Vx, 2)
        iy = size(Vx, 2) - 1
        dyij = 1.0
    end
    v1 = Vx[ix, iy]
    v2 = Vx[ix+1, iy]
    v3 = Vx[ix, iy+1]
    v4 = Vx[ix+1, iy+1]
    vx = bilinearInterp(v1, v2, v3, v4, dxij, dyij)
    # Continuity-based velocity correction for Vx: the interpolated field will have zero divergence!
    # TODO: no correction if too close to boundary: Vx and Vy arrays must have more ghost cells for multi-GPU!
    correction = 0.0
    # right half of cell => extend stencil to the right
    if dxij > 0.5 && ix + 2 <= size(Vx, 1)
        v5 = Vx[ix+2, iy]
        v6 = Vx[ix+2, iy+1]
        correction = 0.5 * (dxij - 0.5)^2 * (
                         (1 - dyij) * (v1 - 2v2 + v5) +
                         dyij * (v3 - 2v4 + v6))
        # left  half of cell => extend stencil to the left
    elseif dxij < 0.5 && ix - 1 >= 1
        v5 = Vx[ix-1, iy]
        v6 = Vx[ix-1, iy+1]
        correction = 0.5 * (dxij - 0.5)^2 * (
                         (1 - dyij) * (v5 - 2v1 + v2) +
                         dyij * (v6 - 2v3 + v4))
    end
    vx += correction

    # Interpolate Vy
    ix, iy, dxij, dyij = topleftIndexRelDist(x_vy_min, y_vy_min, x, y, dx, dy)
    #index range failsafe, in case advection moves the particles out of domain
    if ix < 1
        ix = 1
        dxij = 0.0
    end
    if iy < 1
        iy = 1
        dyij = 0.0
    end
    if ix >= size(Vy, 1)
        ix = size(Vy, 1) - 1
        dxij = 1.0
    end
    if iy >= size(Vy, 2)
        iy = size(Vy, 2) - 1
        dyij = 1.0
    end
    v1 = Vy[ix, iy]
    v2 = Vy[ix+1, iy]
    v3 = Vy[ix, iy+1]
    v4 = Vy[ix+1, iy+1]
    vy = bilinearInterp(v1, v2, v3, v4, dxij, dyij)
    # Continuity-based velocity correction for Vy: the interpolated field will have zero divergence!
    # TODO: no correction if too close to boundary: Vx and Vy arrays must have more ghost cells for multi-GPU!
    correction = 0.0
    # lower half of cell => extend stencil to the bottom
    if dyij > 0.5 && iy + 2 <= size(Vy, 2)
        v5 = Vy[ix, iy+2]
        v6 = Vy[ix+1, iy+2]
        correction = 0.5 * (dyij - 0.5)^2 * (
                         (1 - dxij) * (v1 - 2v3 + v5) +
                         dxij * (v2 - 2v4 + v6))
        # upper half of cell => extend stencil to the top
    elseif dyij < 0.5 && iy - 1 >= 1
        v5 = Vy[ix, iy-1]
        v6 = Vy[ix+1, iy-1]
        correction = 0.5 * (dyij - 0.5)^2 * (
                         (1 - dxij) * (v5 - 2v1 + v3) +
                         dxij * (v6 - 2v2 + v4))
    end
    vx += correction

    return vx, vy
end

"""
    moveMarkersRK4!(x_m, y_m, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dt, lx, ly, dx, dy)

Moves markers according to a fourth order Runge-Kutta method
"""
@parallel_indices (m) function moveMarkersRK4!(x_m, y_m, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dt, lx, ly, dx, dy)

    # Runge-Kutta 4th order
    rk4_dt = @SVector [0.0, 0.5dt, 0.5dt, dt]
    rk4_wt = @SVector [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]

    x_old = x_m[m] # old position
    y_old = y_m[m]
    vx_eff, vy_eff = 0.0, 0.0 # 'effective' velocity for explicit update: x_new = x_old + v_eff*dt
    vx_rk, vy_rk = 0.0, 0.0 # velocity at previous/current point

    for it = 1:4 # loop over points A-D
        # position of current point based on previous point velocities
        x_rk = x_old + rk4_dt[it] * vx_rk
        y_rk = y_old + rk4_dt[it] * vy_rk

        # interpolate velocity to current point
        vx_rk, vy_rk = interpolateV(x_rk, y_rk, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dx, dy)

        # apply RK4 scheme: add up weighted velocities
        vx_eff += rk4_wt[it] * vx_rk
        vy_eff += rk4_wt[it] * vy_rk
    end

    # move particle
    x_new = x_old + vx_eff * dt
    y_new = y_old + vy_eff * dt

    # explicitly restrict particles to stay on domain
    # (optional, does not really change anything if BC correctly implemented and dt small enough)
    # !! TODO: CHANGE, if global domain is not 0-lx and 0-ly !!
    x_new = min(max(x_new, 0), lx)
    y_new = min(max(y_new, 0), ly)

    # write back updated positions
    x_m[m] = x_new
    y_m[m] = y_new

    return nothing
end

"""
    topleftIndexRelDist(x_grid_min, y_grid_min, x, y, dx, dy)

Compute indices `ix`, `iy` of the top left node with respect to the given 2D position as well as the relative distances
"""
@views function topleftIndexRelDist(x_grid_min, y_grid_min, x, y, dx, dy)
    # TODO: ix,iy may be out of bounds if the grid does not cover (x,y), this is not checked here
    # indices: may be out of bounds if the grid does not cover (x,y)
    ix = floor(Int, (x - x_grid_min) / dx) + 1
    iy = floor(Int, (y - y_grid_min) / dy) + 1
    # position of top left node with index (ix,iy)
    x_ix = x_grid_min + (ix - 1) * dx
    y_iy = y_grid_min + (iy - 1) * dy
    # compute relative distances
    dxij = (x - x_ix) / dx
    dyij = (y - y_iy) / dy
    return ix, iy, dxij, dyij
end

"""
    bilinearMarkerToGrid!(x_grid_min, y_grid_min, val_grid, x_m, y_m, val_m, dx, dy, val_wt_sum, wt_sum)

Interpolates markers to grid points
"""
# TODO: multi-xpu interpolation
@views function bilinearMarkerToGrid!(x_grid_min, y_grid_min, val_grid, x_m, y_m, val_m, dx, dy, val_wt_sum, wt_sum)

    Nx, Ny = size(val_grid)
    Nm = size(x_m, 1)
    @assert (size(y_m, 1) == Nm) && (size(val_m, 1) == Nm)

    val_wt_sum .= 0.0
    wt_sum .= 0.0

    # interpolate by summing up the marker contributions to global arrays
    @parallel (1:Nm) atomicAddInterpolation(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy)

    # finally compute actual value from the sums
    @parallel (1:Nx, 1:Ny) safeDivision(val_grid, Data.Array(val_wt_sum), Data.Array(wt_sum))

    return nothing
end

"""
    atomicAddInterpolation(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy)

Sums up all interpolation weights in an atomic fashion
"""
@parallel_indices (m) function atomicAddInterpolation(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy)
    xm = x_m[m]
    ym = y_m[m]
    val = val_m[m]

    # get indices and relative distance to top left node w.r.t marker m.
    # may be 0, when the marker is further left or up than the first grid node
    ix, iy, dxmij, dymij = topleftIndexRelDist(x_grid_min, y_grid_min, xm, ym, dx, dy)
    if !(ix >= 0 && ix <= Nx && iy >= 0 && iy <= Ny)
        @ps_println("ATTENTION: erroneous marker position")
    else
        # sum up weights, if the respective node exists
        if iy > 0
            if ix > 0
                # 1) top left
                w = (1 - dxmij) * (1 - dymij)
                CUDA.@atomic wt_sum[ix, iy] += w
                CUDA.@atomic val_wt_sum[ix, iy] += w * val
            end
            if ix < Nx
                # 2) top right
                w = dxmij * (1 - dymij)
                CUDA.@atomic wt_sum[ix+1, iy] += w
                CUDA.@atomic val_wt_sum[ix+1, iy] += w * val
            end
        end
        if iy < Ny
            if ix > 0
                # 3) bottom left
                w = (1 - dxmij) * dymij
                CUDA.@atomic wt_sum[ix, iy+1] += w
                CUDA.@atomic val_wt_sum[ix, iy+1] += w * val
            end
            if ix < Nx
                # 4) bottom right
                w = dxmij * dymij
                CUDA.@atomic wt_sum[ix+1, iy+1] += w
                CUDA.@atomic val_wt_sum[ix+1, iy+1] += w * val
            end
        end
    end

    return nothing
end

"""
    safeDivision(result, numerator, denominator)

Performs division and warns about zero denominator
"""
@parallel_indices (ix, iy) function safeDivision(result, numerator, denominator)
    denom = denominator[ix, iy]
    if denom != 0.0
        result[ix, iy] = numerator[ix, iy] / denom
    else
        @ps_println("PROBLEM: no markers close enough for value interpolation at node: ix = ", ix, ", iy = ", iy)
    end
    return nothing
end

"""
    setInitialMarkerCoords!(x_m, y_m, Nmx, Nmy, xlims, ylims, RAND_MARKER_POS::Bool; rng=Random.GLOBAL_RNG)

Sets initial coordinates and properties of the markers

`x` and `y` are coordinates of the basic grid nodes

`xlims` and `ylims` contain domain lower and upper limits at start and end indices respectively
"""
@views function setInitialMarkerCoords!(x_m, y_m, Nmx, Nmy, xlims, ylims, RAND_MARKER_POS::Bool; rng=Random.GLOBAL_RNG)
    Nm = Nmx * Nmy
    @assert size(x_m, 1) == (Nm)
    @assert size(y_m, 1) == (Nm)
    dxm = (xlims[end] - xlims[1]) / Nmx
    dym = (ylims[end] - ylims[1]) / Nmy
    xcoords = LinRange(xlims[1] + 0.5dxm, xlims[end] - 0.5dxm, Nmx)
    ycoords = LinRange(ylims[1] + 0.5dym, ylims[end] - 0.5dym, Nmy)
    m = 1
    for ix = 1:Nmx
        for iy = 1:Nmy
            x_m[m] = xcoords[ix]
            y_m[m] = ycoords[iy]
            m += 1
        end
    end
    if RAND_MARKER_POS
        x_m .+= (rand(rng, Nm) .- 0.5) .* dxm
        y_m .+= (rand(rng, Nm) .- 0.5) .* dym
    end
    return nothing
end

"""
    setInitialMarkerProperties!(coords, lx, ly, x_m, y_m, ρ_m, μ_m, Nm, μ_air, μ_matrix, μ_plume, ρ_air, ρ_matrix, ρ_plume, plume_x, plume_y, plume_r, air_height)

Sets initial marker properties `ρ_m` and `μ_m` according to whether their coordinates correspond to
1. sticky air
2. plume / inclusion
3. the surrounding matrix
"""
@views function setInitialMarkerProperties!(coords, lx, ly, x_m, y_m, ρ_m, μ_m, Nm, μ_air, μ_matrix, μ_plume, ρ_air, ρ_matrix, ρ_plume, plume_x, plume_y, plume_r, air_height)
    @assert size(x_m, 1) == Nm
    @assert size(y_m, 1) == Nm
    @assert size(ρ_m) == size(μ_m) && size(μ_m, 1) == Nm
    for m = 1:Nm
        x = x_m[m] + lx * coords[1]
        y = y_m[m] + ly * coords[2]
        if y < air_height
            ρ_m[m] = ρ_air
            μ_m[m] = μ_air
        elseif (x - plume_x)^2 + (y - plume_y)^2 < plume_r^2
            ρ_m[m] = ρ_plume
            μ_m[m] = μ_plume
        else
            ρ_m[m] = ρ_matrix
            μ_m[m] = μ_matrix
        end
    end
    return nothing
end

"""
    saveStats!(dims, nt)

Saves visualization relevant stats to disk in .mat format
"""
@views function saveStats!(dims, nt)
    file = matopen("viz_out/stats.mat", "w")

    write(file, "dims", dims)
    write(file, "nt", nt)

    close(file)

    return nothing
end

"""
    saveMarkers!(it, rank, coords, localDomain, x_m, y_m, ρ_m)

Saves marker positions and densities to disk in .mat format
"""
@views function saveMarkers!(it, rank, coords, localDomain, x_m, y_m, ρ_m)
    file = matopen(string(@sprintf("viz_out/markers_%04d_%04d", it, rank), ".mat"), "w")

    write(file, "x_m", convert.(Float32, Array(x_m) .+ (localDomain[1] * coords[1])))
    write(file, "y_m", convert.(Float32, Array(y_m) .+ (localDomain[2] * coords[2])))
    write(file, "rho_m", convert.(Float32, Array(ρ_m)))

    close(file)

    return nothing
end

"""
    saveGrid(x, y, x_p, y_p, x_vx, y_vx, x_vy, y_vy, P, Vx, Vy)

Saves relevant arrays on various grid points to disk in .mat format
"""
@views function saveGrid!(it, rank, x, y, μ_b, x_p, y_p, P, x_vx, y_vx, Vx, x_vy, y_vy, Vy)
    file = matopen(string(@sprintf("viz_out/grid_%04d_%04d", it, rank), ".mat"), "w")

    write(file, "x", convert.(Float32, Array(x)))
    write(file, "y", convert.(Float32, Array(y)))
    write(file, "mu_b", convert.(Float32, Array(μ_b)))

    write(file, "x_p", convert.(Float32, Array(x_p)))
    write(file, "y_p", convert.(Float32, Array(y_p)))
    write(file, "P", convert.(Float32, Array(P)))

    write(file, "x_vx", convert.(Float32, Array(x_vx)))
    write(file, "y_vx", convert.(Float32, Array(y_vx)))
    write(file, "Vx", convert.(Float32, Array(Vx)))

    write(file, "x_vy", convert.(Float32, Array(x_vy)))
    write(file, "y_vy", convert.(Float32, Array(y_vy)))
    write(file, "Vy", convert.(Float32, Array(Vy)))

    close(file)

    return nothing
end

"""
    dstToSrcIdx!(i)

Transforms destination to source rank index
"""
@views function dstToSrcIdx!(i)
    @assert i >= 1 && i <= 8
    (i >= 5) && return (i + 4) % 9 + 1
    return i + 4
end

"""
    neihgbourIdxToShift!(i)

Calculates three-dimensional shift for a given neighbour index
"""
@views function neihgbourIdxToShift!(i)
    @assert i >= 1 && i <= 8
    (i == 1) && return [-1, -1, 0]
    (i == 2) && return [0, -1, 0]
    (i == 3) && return [1, -1, 0]
    (i == 4) && return [1, 0, 0]
    (i == 5) && return [1, 1, 0]
    (i == 6) && return [0, 1, 0]
    (i == 7) && return [-1, 1, 0]
    (i == 8) && return [-1, 0, 0]
end

"""
    coordsToNeihgbourCoords!(coords, i)

Transforms coordinates of the current rank to the neighbour coordinates for a given neighbour index
"""
@views function coordsToNeihgbourCoords!(coords, i)
    @assert i >= 1 && i <= 8
    return coords .+ neihgbourIdxToShift!(i)
end

"""
    posToNeighbourIdx!(localDomain, x, y)

Retrieves a potential neighbour index given a local position
"""
@views function posToNeighbourIdx!(localDomain, x, y)
    if y < 0.0
        (x < 0.0) && return 1
        (x > localDomain[1]) && return 3
        return 2
    end
    if y > localDomain[2]
        (x < 0.0) && return 7
        (x > localDomain[1]) && return 5
        return 6
    end
    (x < 0.0) && return 8
    (x > localDomain[1]) && return 4
    return 0 # inside local domain, no neighbour
end

"""
    exchangeMarkers!(comm, dims, localDomain, x_m, y_m, ρ_m, μ_m)

Exchanges markers outside of local boundaries to respective neighbouring ranks
"""
@views function exchangeMarkers!(comm, dims, localDomain, x_m, y_m, ρ_m, μ_m)
    numNeighbors = 8
    coords = MPI.Cart_coords(comm)

    # calculate send / receive ranks
    dstRanks = zeros(Int32, numNeighbors)
    srcRanks = zeros(Int32, numNeighbors)
    for i = 1:numNeighbors
        ncoords = coordsToNeihgbourCoords!(coords, i)
        nrank = MPI.MPI_PROC_NULL
        if (ncoords[1] >= 0) && (ncoords[1] <= dims[1] - 1) && (ncoords[2] >= 0) && (ncoords[2] <= dims[2] - 1)
            nrank = MPI.Cart_rank(comm, ncoords)
        end
        dstRanks[i] = nrank
    end
    for i = 1:numNeighbors
        srcRanks[i] = dstRanks[dstToSrcIdx!(i)]
    end

    # initialize send / receive buffers
    sendBuffersx = Array{Float64}[]
    sendBuffersy = Array{Float64}[]
    sendBuffersρ = Array{Float64}[]
    sendBuffersμ = Array{Float64}[]
    recvBuffersx = Array{Float64}[]
    recvBuffersy = Array{Float64}[]
    recvBuffersρ = Array{Float64}[]
    recvBuffersμ = Array{Float64}[]
    for _ = 1:numNeighbors
        push!(sendBuffersx, Array{Float64}(undef, 0))
        push!(sendBuffersy, Array{Float64}(undef, 0))
        push!(sendBuffersρ, Array{Float64}(undef, 0))
        push!(sendBuffersμ, Array{Float64}(undef, 0))
        push!(recvBuffersx, Array{Float64}(undef, 0))
        push!(recvBuffersy, Array{Float64}(undef, 0))
        push!(recvBuffersρ, Array{Float64}(undef, 0))
        push!(recvBuffersμ, Array{Float64}(undef, 0))
    end

    # initialize send / receive requests
    sendSizeReqs = Array{MPI.Request}(undef, 0)
    sendDataReqs = Array{MPI.Request}(undef, 0)
    recvSizeReqs = Array{MPI.Request}(undef, 0)
    recvDataReqs = Array{MPI.Request}(undef, 0)

    # initialize temporary arrays for new local markers
    tmpx = Array{Float64}(undef, 0)
    tmpy = Array{Float64}(undef, 0)
    tmpρ = Array{Float64}(undef, 0)
    tmpμ = Array{Float64}(undef, 0)

    # separate markers within the local boundary from the ones outside
    sz = size(x_m, 1)
    @assert size(y_m, 1) == sz && size(ρ_m, 1) == sz && size(μ_m, 1) == sz
    for i = 1:sz
        dstId = posToNeighbourIdx!(localDomain, x_m[i], y_m[i])
        if dstId == 0
            # inside, push to tmp
            push!(tmpx, x_m[i])
            push!(tmpy, y_m[i])
            push!(tmpρ, ρ_m[i])
            push!(tmpμ, μ_m[i])
        else
            # outside, prepare to send
            shift = neihgbourIdxToShift!(dstId)
            x_m[i] -= localDomain[1] * shift[1]
            y_m[i] -= localDomain[2] * shift[2]
            push!(sendBuffersx[dstId], x_m[i])
            push!(sendBuffersy[dstId], y_m[i])
            push!(sendBuffersρ[dstId], ρ_m[i])
            push!(sendBuffersμ[dstId], μ_m[i])
        end
    end

    # send size and buffers to destination ranks
    for i = 1:numNeighbors
        dst = dstRanks[i]
        (dst == MPI.MPI_PROC_NULL) && continue
        push!(sendSizeReqs, MPI.Isend(size(sendBuffersx[i], 1), dst, i, comm))
        push!(sendDataReqs, MPI.Isend(sendBuffersx[i], dst, i, comm))
        push!(sendSizeReqs, MPI.Isend(size(sendBuffersy[i], 1), dst, i + 1 * numNeighbors, comm))
        push!(sendDataReqs, MPI.Isend(sendBuffersy[i], dst, i + 1 * numNeighbors, comm))
        push!(sendSizeReqs, MPI.Isend(size(sendBuffersρ[i], 1), dst, i + 2 * numNeighbors, comm))
        push!(sendDataReqs, MPI.Isend(sendBuffersρ[i], dst, i + 2 * numNeighbors, comm))
        push!(sendSizeReqs, MPI.Isend(size(sendBuffersμ[i], 1), dst, i + 3 * numNeighbors, comm))
        push!(sendDataReqs, MPI.Isend(sendBuffersμ[i], dst, i + 3 * numNeighbors, comm))
    end

    # initialize receive sizes buffers
    recvSizesx = Array{Int64}[]
    recvSizesy = Array{Int64}[]
    recvSizesρ = Array{Int64}[]
    recvSizesμ = Array{Int64}[]
    for _ = 1:numNeighbors
        push!(recvSizesx, zeros(Int64, 1))
        push!(recvSizesy, zeros(Int64, 1))
        push!(recvSizesρ, zeros(Int64, 1))
        push!(recvSizesμ, zeros(Int64, 1))
    end

    # receive sizes 
    for i = 1:numNeighbors
        src = srcRanks[i]
        (src == MPI.MPI_PROC_NULL) && continue
        push!(recvSizeReqs, MPI.Irecv!(recvSizesx[i], src, i, comm))
        push!(recvSizeReqs, MPI.Irecv!(recvSizesy[i], src, i + 1 * numNeighbors, comm))
        push!(recvSizeReqs, MPI.Irecv!(recvSizesρ[i], src, i + 2 * numNeighbors, comm))
        push!(recvSizeReqs, MPI.Irecv!(recvSizesμ[i], src, i + 3 * numNeighbors, comm))
    end

    # wait for all sizes to arrive
    MPI.Waitall!(recvSizeReqs)

    # receive data
    for i = 1:numNeighbors
        src = srcRanks[i]
        (src == MPI.MPI_PROC_NULL) && continue

        recvBuffersx[i] = zeros(recvSizesx[i][1])
        push!(recvDataReqs, MPI.Irecv!(recvBuffersx[i], src, i, comm))

        recvBuffersy[i] = zeros(recvSizesy[i][1])
        push!(recvDataReqs, MPI.Irecv!(recvBuffersy[i], src, i + 1 * numNeighbors, comm))

        recvBuffersρ[i] = zeros(recvSizesρ[i][1])
        push!(recvDataReqs, MPI.Irecv!(recvBuffersρ[i], src, i + 2 * numNeighbors, comm))

        recvBuffersμ[i] = zeros(recvSizesμ[i][1])
        push!(recvDataReqs, MPI.Irecv!(recvBuffersμ[i], src, i + 3 * numNeighbors, comm))
    end

    # wait for the rest to complete
    MPI.Waitall!(recvDataReqs)
    MPI.Waitall!(sendSizeReqs)
    MPI.Waitall!(sendDataReqs)

    # add received markers to tmp arrays
    for i = 1:numNeighbors
        tmpx = append!(tmpx, recvBuffersx[i])
        tmpy = append!(tmpy, recvBuffersy[i])
        tmpρ = append!(tmpρ, recvBuffersρ[i])
        tmpμ = append!(tmpμ, recvBuffersμ[i])
    end

    # return tmp arrays
    return tmpx, tmpy, tmpρ, tmpμ
end

StokesFlow2D()

#@testset "StokesFlow2D_gpu" begin
#    nt = 10
#    nx, ny = 35, 45
#    # tests should not depend on a rng seed, see the Warning at https://docs.julialang.org/en/v1/stdlib/Random/
#    result = StokesFlow2D(; Nt=nt, Nx=nx, Ny=ny, RAND_MARKER_POS=false, do_plot=false, print_info=false)
#    inds = [181, 219, 388, 444, 637, 920, 1049, 1074, 1223, 1367]
#    refs = [0.04974145217766237, -0.03512273865731893, -0.14844495700026145, 0.4927706169377982, -0.1486118118461089, 0.21083507100494975, -0.03009881585240687, -0.20891915972705769, -0.05836431607719553, 0.025274386167270724]
#    @test all(isapprox.(refs, result[inds]; atol=1e-4))
#end
