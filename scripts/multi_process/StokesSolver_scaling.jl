using ParallelStencil
using ParallelStencil.FiniteDifferences2D

const USE_GPU = if haskey(ENV, "USE_GPU")
    ENV["USE_GPU"] == "true" ? true : false
else
    false
end

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using Plots, Plots.Measures, ImplicitGlobalGrid, MAT
using Test
import Random

include("../multi_process/StokesSolver_multixpu.jl")
include("../multi_process/MarkerExchange.jl")
include("../multi_process/MarkerToGrid.jl")

ENV["GKSwstype"] = "nul"

default(size=(600, 500), framestyle=:box, label=false, grid=true, margin=10mm, lw=1.5, markershape=:circle, markersize=5)

"""
    timeStokesSolver(n)

Calculates the effective memory throughput of the Stokes solver
"""
@views function timeStokesSolver(n)
    Nx, Ny = n, n
    Lx_glob, Ly_glob = 10, 10
    RAND_MARKER_POS = true

    function density(x, y)
        if y < air_height
            ρ_air
        elseif (x - plume_x)^2 + (y - plume_y)^2 < plume_r^2
            ρ_plume
        else
            ρ_matrix
        end
    end
    function viscosity(x, y)
        if y < air_height
            μ_air
        elseif (x - plume_x)^2 + (y - plume_y)^2 < plume_r^2
            μ_plume
        else
            μ_matrix
        end
    end

    rank, dims, nprocs, coords, comm_cart = init_global_grid(Nx, Ny, 1; quiet=true, init_MPI=false)
    grid = ImplicitGlobalGrid.get_global_grid()

    # --- PARAMETERS ---
    maxdisp = 0.5                 # dt is determined s.t. no marker moves further than maxdisp cells. must be <= 0.5
    g_y = 9.81                    # earth gravity
    marker_density = 5            # use this amount of markers per grid step per dimension

    # derived quantities
    dx = Lx_glob / (nx_g() - 1)       # grid step
    dy = Ly_glob / (ny_g() - 1)
    lx = (Nx - 1) * dx                # local domain size
    ly = (Ny - 1) * dy

    # random numbers for initial marker postions
    Random.seed!(rank)            # seed default RNG for random marker positions


    # --- GRID ARRAYS ---
    # grid array allocations
    P = @zeros(Nx - 1, Ny - 1)
    Vx = @zeros(Nx + 2, Ny + 1)                 # Velocity in x-direction
    Vy = @zeros(Nx + 1, Ny + 2)                 # Velocity in y-direction
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

    # local coordinates for all grid points
    x = [(ix - 1) * dx for ix = 1:Nx]              # basic nodes
    y = [(iy - 1) * dy for iy = 1:Ny]
    x_p = [(ix - 1) * dx + 0.5dx for ix = 1:Nx-1]  # pressure nodes
    y_p = [(iy - 1) * dy + 0.5dy for iy = 1:Ny-1]
    x_vx = [(ix - 2) * dx for ix = 1:Nx+2]         # Vx nodes
    y_vx = [(iy - 1) * dy - 0.5dy for iy = 1:Ny+1]
    x_vy = [(ix - 1) * dx - 0.5dx for ix = 1:Nx+1] # Vy nodes
    y_vy = [(iy - 2) * dy for iy = 1:Ny+2]
    x_ρ = x_vy                                     # nodes for ρ: same as Vy, but smaller in y
    y_ρ = y_vy[2:end-1]
    # consistency checks
    @assert size(x_p, 1) == size(P, 1) && size(y_p, 1) == size(P, 2)
    @assert size(x_vx, 1) == size(Vx, 1) && size(y_vx, 1) == size(Vx, 2)
    @assert size(x_vy, 1) == size(Vy, 1) && size(y_vy, 1) == size(Vy, 2)
    @assert size(x_ρ, 1) == size(ρ_vy, 1) && size(y_ρ, 1) == size(ρ_vy, 2)


    # --- MARKER ARRAYS & INITIAL CONDITIONS (MARKER PROPERTIES) ---
    x_m, y_m, ρ_m, μ_m = initializeMarkersCPU(comm_cart, dims, coords, marker_density::Integer, lx, ly, dx, dy, Nx, Ny, RAND_MARKER_POS)
    #setInitialMarkerProperties!(x_m, y_m, ρ_m, μ_m, x0, y0, density, viscosity)

    # transform marker arrays to xPU arrays
    x_m = Data.Array(x_m)
    y_m = Data.Array(y_m)
    ρ_m = Data.Array(ρ_m)
    μ_m = Data.Array(μ_m)

    dt = 0.0

    # interpolate material properties to grid
    bilinearMarkerToGrid!(x_ρ[1], y_ρ[1], ρ_vy, x_m, y_m, ρ_m, dx, dy, val_wt_sum, wt_sum, grid)
    bilinearMarkerToGrid!(x[1], y[1], μ_b, x_m, y_m, μ_m, dx, dy, val_wt_sum, wt_sum, grid)
    bilinearMarkerToGrid!(x_p[1], y_p[1], μ_p, x_m, y_m, μ_m, dx, dy, val_wt_sum, wt_sum, grid)

    # calculate velocities on grid
    GC.gc()
    GC.enable(false)
    _, T_eff = solveStokes!(P, Vx, Vy, ρ_vy, μ_b, μ_p, τxx, τyy, τxy, ∇V, dτPt, Rx, Ry, dVxdτ, dVydτ, dτVx, dτVy,
        Vx_small, Vy_small, g_y, dx, dy, Nx, Ny, dt, maxdisp, comm_cart; use_free_surface_stabilization=true, ϵ=1e-5, print_info=false)
    GC.enable(true)

    finalize_global_grid(; finalize_MPI=false)

    return T_eff
end

"""
    strongScaling()

Performs strong scaling with one process and saves the results in a PNG image
"""
@views function strongScaling()
    MPI.Init()
    @assert MPI.Comm_size(MPI.COMM_WORLD) == 1 "Strong scaling needs 1 process"

    all_n = [2^(5 + i) for i = 0:5]
    T_effs = []
    for n = all_n
        T_eff = timeStokesSolver(n)
        push!(T_effs, T_eff)
        @show n, T_eff
    end

    MPI.Finalize()

    p = plot(all_n, T_effs, xlabel="Nx=Ny", ylabel="T_eff [GB/s]", title="Strong Scaling")
    png(p, "strongScaling.png")

    return nothing
end

"""
    weakScaling()

Performs one weak scaling run with a given amount of processes
"""
@views function weakScaling()
    MPI.Init()
    # run sequentally with 1,4,16,25,64 processes

    n = 2^10 # TODO: best n according to strong scaling
    T_eff = timeStokesSolver(n)

    T_effs_min = MPI.Reduce(T_eff, MPI.MIN, 0, MPI.COMM_WORLD)
    T_effs_sum = MPI.Reduce(T_eff, MPI.SUM, 0, MPI.COMM_WORLD)
    T_effs_max = MPI.Reduce(T_eff, MPI.MAX, 0, MPI.COMM_WORLD)

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    if (rank == 0)
        np = MPI.Comm_size(MPI.COMM_WORLD)
        T_effs_avg = T_effs_sum / np
        @show np, T_effs_min
        @show np, T_effs_avg
        @show np, T_effs_max
    end

    MPI.Finalize()

    return nothing
end

"""
    weakScalingPlot()

Saves the previously obtained weak scaling results in a PNG image
"""
@views function weakScalingPlot()
    np = [1, 4, 16, 25, 64]

    T_effs_min = [1.0, 2.0, 3.0, 4.0, 5.0] # TODO: insert values from weakScaling()
    T_effs_avg = [1.0, 2.0, 3.0, 4.0, 5.0] # TODO: insert values from weakScaling()
    T_effs_max = [1.0, 2.0, 3.0, 4.0, 5.0] # TODO: insert values from weakScaling()
    T_effs_min_rel = T_effs_min ./ T_effs_min[1]
    T_effs_avg_rel = T_effs_avg ./ T_effs_avg[1]
    T_effs_max_rel = T_effs_max ./ T_effs_max[1]

    plot(np, T_effs_min_rel; label="min")
    plot!(np, T_effs_avg_rel; label="avg")
    p = plot!(np, T_effs_max_rel; xlabel="np", ylabel="T_eff_rel", label="max", title="Weak Scaling")
    png(p, "weakScaling.png")

    return nothing
end

#strongScaling()  # TODO: run this
#weakScaling() # TODO: run this
#weakScalingPlot() # TODO: run this
