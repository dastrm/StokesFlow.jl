using ParallelStencil
using ParallelStencil.FiniteDifferences2D
const USE_GPU = true
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using Plots, Plots.Measures, ImplicitGlobalGrid, MAT
using Test
import Random

include("StokesSolver_multixpu.jl")
include("MarkerToGrid.jl")
include("MoveMarkers.jl")
include("MarkerExchange.jl")

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
    grid = ImplicitGlobalGrid.get_global_grid()

    ENV["GKSwstype"] = "nul"
    if do_plot
        (rank == 0) && (!isdir("viz_out")) && mkdir("viz_out")
        MPI.Barrier(comm_cart) # ensures that dir viz_out exists from now on
    end

    # --- PARAMETERS ---
    # time
    maxdisp = 0.5                               # dt is determined s.t. no marker moves further than maxdisp cells
    # physical parameters
    g_y = 9.81                                  # earth gravity
    lx, ly = 10, 10                             # global domain size
    lxl, lyl = lx / dims[1], ly / dims[2]       # local domain size
    μ_air, μ_matrix, μ_plume = 1e-2, 1e0, 1e-1  # Viscosity
    ρ_air, ρ_matrix, ρ_plume = 1e-3, 3.3, 3.2   # Density
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
    P = @zeros(Nx - 1, Ny - 1)
    Vx = @zeros(Nx + 2, Ny + 1)                   # Velocity in x-direction
    Vy = @zeros(Nx + 1, Ny + 2)                   # Velocity in y-direction
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
    Vx_small = @zeros(Nx, Ny + 1)
    Vy_small = @zeros(Nx + 1, Ny)

    # additional arrays for marker -> grid interpolation
    interp_size = max.(size(ρ_vy), size(μ_b), size(μ_p)) # these represent the arrays that are filled by interpolation
    val_wt_sum = @zeros(interp_size)
    wt_sum = @zeros(interp_size)

    # coordinates for all grid points
    x = [(ix - 1) * dx for ix = 1:Nx]              # basic nodes
    y = [(iy - 1) * dy for iy = 1:Ny]
    x_p = [(ix - 1) * dx + 0.5dx for ix = 1:Nx-1]  # pressure nodes
    y_p = [(iy - 1) * dy + 0.5dy for iy = 1:Ny-1]
    x_vx = [(ix - 2) * dx for ix = 1:Nx+2] # Vx nodes
    y_vx = [(iy - 1) * dy - 0.5dy for iy = 1:Ny+1]
    x_vy = [(ix - 1) * dx - 0.5dx for ix = 1:Nx+1] # Vy nodes
    y_vy = [(iy - 2) * dy for iy = 1:Ny+2]
    x_ρ = x_vy                            # nodes for ρ: same as Vy, but smaller in y
    y_ρ = y_vy[2:end-1]
    # consistency checks
    @assert size(x_p, 1) == size(P, 1) && size(y_p, 1) == size(P, 2)
    @assert size(x_vx, 1) == size(Vx, 1) && size(y_vx, 1) == size(Vx, 2)
    @assert size(x_vy, 1) == size(Vy, 1) && size(y_vy, 1) == size(Vy, 2)
    @assert size(x_ρ, 1) == size(ρ_vy, 1) && size(y_ρ, 1) == size(ρ_vy, 2)

    # --- INITIAL CONDITIONS ---
    setInitialMarkerCoords!(coords, dims, dx, dy, x_m, y_m, Nmx, Nmy, x, y, RAND_MARKER_POS::Bool)
    setInitialMarkerProperties!(coords, lxl, lyl, x_m, y_m, ρ_m, μ_m, Nm, μ_air, μ_matrix, μ_plume, ρ_air, ρ_matrix, ρ_plume, plume_x, plume_y, plume_r, air_height)
    if do_plot
        saveStats!(0, rank, dims)
        saveMarkers!(0, rank, coords, [lxl, lyl], dx, dy, x_m, y_m, ρ_m)
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
            bilinearMarkerToGrid!(x_ρ[1], y_ρ[1], ρ_vy, x_m, y_m, ρ_m, dx, dy, val_wt_sum, wt_sum, grid)
            bilinearMarkerToGrid!(x[1], y[1], μ_b, x_m, y_m, μ_m, dx, dy, val_wt_sum, wt_sum, grid)
            bilinearMarkerToGrid!(x_p[1], y_p[1], μ_p, x_m, y_m, μ_m, dx, dy, val_wt_sum, wt_sum, grid)
        end

        # calculate velocities on grid
        t2 = @elapsed begin
            dt, _ = solveStokes!(P, Vx, Vy, ρ_vy, μ_b, μ_p,
                τxx, τyy, τxy, ∇V, dτPt, Rx, Ry, dVxdτ, dVydτ, dτVx, dτVy, Vx_small, Vy_small,
                g_y, dx, dy, Nx, Ny,
                dt, maxdisp, comm_cart; use_free_surface_stabilization=true,
                ϵ=1e-5,
                print_info=print_info && rank == 0)
        end

        # move markers
        t3 = @elapsed begin
            moveMarkersRK4!(x_m, y_m, Vx, Vy, x_vx[1], y_vx[1], x_vy[1], y_vy[1], dt, lx, ly, dx, dy)
        end

        # exchange markers
        t4 = @elapsed begin
            x_m, y_m, ρ_m, μ_m = exchangeMarkers!(comm_cart, dims, [lxl, lyl], dx, dy, Array(x_m), Array(y_m), Array(ρ_m), Array(μ_m))
            # transform marker arrays to xPU arrays
            x_m = Data.Array(x_m)
            y_m = Data.Array(y_m)
            ρ_m = Data.Array(ρ_m)
            μ_m = Data.Array(μ_m)
        end

        # plot current state
        if do_plot
            saveStats!(t, rank, dims)
            saveMarkers!(t, rank, coords, [lxl, lyl], dx, dy, x_m, y_m, ρ_m)
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
    setInitialMarkerCoords!(coords, dims, dx, dy, x_m, y_m, Nmx, Nmy, xlims, ylims, RAND_MARKER_POS::Bool; rng=Random.GLOBAL_RNG)

Sets initial coordinates and properties of the markers

`x` and `y` are coordinates of the basic grid nodes

`xlims` and `ylims` contain domain lower and upper limits at start and end indices respectively
"""
@views function setInitialMarkerCoords!(coords, dims, dx, dy, x_m, y_m, Nmx, Nmy, xlims, ylims, RAND_MARKER_POS::Bool; rng=Random.GLOBAL_RNG)
    Nm = Nmx * Nmy
    @assert size(x_m, 1) == (Nm)
    @assert size(y_m, 1) == (Nm)
    xlimslower = coords[1] == 0 ? xlims[1] : xlims[1] + dx / 2
    xlimsupper = coords[1] == dims[1] - 1 ? xlims[end] : xlims[end] - dx / 2
    ylimslower = coords[2] == 0 ? ylims[1] : ylims[1] + dy / 2
    ylimsupper = coords[2] == dims[2] - 1 ? ylims[end] : ylims[end] - dy / 2
    dxm = (xlimsupper - xlimslower) / Nmx
    dym = (ylimsupper - ylimslower) / Nmy
    xcoords = LinRange(xlimslower + 0.5dxm, xlimsupper - 0.5dxm, Nmx)
    ycoords = LinRange(ylimslower + 0.5dym, ylimsupper - 0.5dym, Nmy)
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
    saveStats!(nt, rank, dims)

Saves visualization relevant stats to disk in .mat format
"""
@views function saveStats!(nt, rank, dims)
    (rank != 0) && return nothing

    file = matopen("viz_out/stats.mat", "w")

    write(file, "dims", dims)
    write(file, "nt", nt)

    close(file)

    return nothing
end

"""
    saveMarkers!(nt, rank, coords, localDomain, dx, dy, x_m, y_m, ρ_m)

Saves marker positions and densities to disk in .mat format
"""
@views function saveMarkers!(nt, rank, coords, localDomain, dx, dy, x_m, y_m, ρ_m)
    file = matopen(string(@sprintf("viz_out/markers_%04d_%04d", nt, rank), ".mat"), "w")

    write(file, "x_m", convert.(Float32, Array(x_m) .+ ((localDomain[1] - dx) * coords[1])))
    write(file, "y_m", convert.(Float32, Array(y_m) .+ ((localDomain[2] - dy) * coords[2])))
    write(file, "rho_m", convert.(Float32, Array(ρ_m)))

    close(file)

    return nothing
end

"""
    saveGrid!(nt, rank, x, y, μ_b, x_p, y_p, P, x_vx, y_vx, Vx, x_vy, y_vy, Vy)

Saves relevant arrays on various grid points to disk in .mat format
"""
@views function saveGrid!(nt, rank, x, y, μ_b, x_p, y_p, P, x_vx, y_vx, Vx, x_vy, y_vy, Vy)
    file = matopen(string(@sprintf("viz_out/grid_%04d_%04d", nt, rank), ".mat"), "w")

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


StokesFlow2D()
