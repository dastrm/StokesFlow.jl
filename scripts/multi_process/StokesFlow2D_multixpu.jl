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

include("StokesSolver_multixpu.jl")
include("MarkerToGrid.jl")
include("MoveMarkers.jl")
include("MarkerExchange.jl")
include("../GlobalGather.jl")

# uncomment for Piz Daint
#ENV["GKSwstype"] = "nul" 

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
@views function StokesFlow2D(Nt, Nx, Ny, Lx_glob, Ly_glob, density, viscosity;
                             dimx::Integer=0, dimy::Integer=0,
                             RAND_MARKER_POS::Bool=true,
                             plot_fields_live::Bool=false, plot_markers_live::Bool=true, save_to_file::Bool=true, print_info::Bool=true,
                             init_MPI::Bool=init_MPI)

    rank, dims, nprocs, coords, comm_cart = init_global_grid(Nx, Ny, 1; dimx=dimx, dimy=dimy, dimz=1, quiet=!print_info, init_MPI=init_MPI)
    grid = ImplicitGlobalGrid.get_global_grid()

    if plot_fields_live || plot_markers_live || save_to_file
        (rank == 0) && (!isdir("viz_out")) && mkdir("viz_out")
        MPI.Barrier(comm_cart) # ensures that dir viz_out exists from now on
    end


    # --- PARAMETERS ---
    maxdisp = 0.5                 # dt is determined s.t. no marker moves further than maxdisp cells. must be <= 0.5
    g_y = 9.81                    # earth gravity
    marker_density = 5            # use this amount of markers per grid step per dimension

    # derived quantities
    dx = Lx_glob/(nx_g()-1)       # grid step
    dy = Ly_glob/(ny_g()-1)
    lx = (Nx-1)*dx                # local domain size
    ly = (Ny-1)*dy
    x0 = coords[1]*(Nx-2)*dx      # offset of local coordinates
    y0 = coords[2]*(Ny-2)*dy

    # random numbers for initial marker postions
    Random.seed!(rank)            # seed default RNG for random marker positions


    # --- GRID ARRAYS ---
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

    # local coordinates for all grid points
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


    # --- MARKER ARRAYS & INITIAL CONDITIONS (MARKER PROPERTIES) ---
    x_m, y_m, ρ_m, μ_m = initializeMarkersCPU(comm_cart,dims,coords,marker_density::Integer,lx,ly,dx,dy,Nx,Ny,RAND_MARKER_POS)
    setInitialMarkerProperties!(x_m, y_m, ρ_m, μ_m, x0, y0, density, viscosity)
    if save_to_file
        saveStats!(0, rank, dims)
        saveMarkers!(0, rank, coords, [lx, ly], dx, dy, x_m, y_m, ρ_m)
        saveGrid!(0, rank, x, y, μ_b, x_p, y_p, P, x_vx, y_vx, Vx, x_vy, y_vy, Vy)
    end
    if plot_markers_live
        plot_markers(x_m, y_m, ρ_m, μ_m, dims, dx, dy, lx, ly, rank, comm_cart, 0)
    end
    # transform marker arrays to xPU arrays
    x_m = Data.Array(x_m)
    y_m = Data.Array(y_m)
    ρ_m = Data.Array(ρ_m)
    μ_m = Data.Array(μ_m)

    
    # --- TIMESTEPPING ---
    times = zeros(5)

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
            x_m, y_m, ρ_m, μ_m = exchangeMarkers!(comm_cart, dims, [lx, ly], dx, dy, Array(x_m), Array(y_m), Array(ρ_m), Array(μ_m))
            # transform marker arrays to xPU arrays
            x_m = Data.Array(x_m)
            y_m = Data.Array(y_m)
            ρ_m = Data.Array(ρ_m)
            μ_m = Data.Array(μ_m)
        end

        # plot current state
        t5 = @elapsed begin
            if save_to_file
                saveStats!(t, rank, dims)
                saveMarkers!(t, rank, coords, [lx, ly], dx, dy, x_m, y_m, ρ_m)
                saveGrid!(t, rank, x, y, μ_b, x_p, y_p, P, x_vx, y_vx, Vx, x_vy, y_vy, Vy)
            end
            if plot_markers_live
                plot_markers(Array(x_m), Array(y_m), Array(ρ_m), Array(μ_m), dims, dx, dy, lx, ly, rank, comm_cart, t)
            end
            if plot_fields_live
                plot_fields(Array(Vx),Array(Vy),Array(μ_b),Array(P),rank,dims,Nx,Ny,dx,dy,t)
            end
        end

        if t > 2
            times[1] += t1
            times[2] += t2
            times[3] += t3
            times[4] += t4
            times[5] += t5
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
            "  exchangeMarkers: ", times[4], " %\n",
            "  saving/plotting: ", times[5], " %\n")
    end

    return Array(Vx), Array(Vy), t_tot
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



function plot_markers(x_m, y_m, ρ_m, μ_m, dims, dx, dy, lx, ly, me, comm, step)
    x_m_glob = zeros(0)
    y_m_glob = zeros(0)
    ρ_m_glob = zeros(0)
    μ_m_glob = zeros(0)
    gather_markers!(x_m,y_m,ρ_m,μ_m,x_m_glob,y_m_glob,ρ_m_glob,μ_m_glob,dims,dx,dy,lx,ly,me,comm)
    if me == 0
        # ghost boundaries are visible in the plot due to the markers' plotting order.
        # optionally sorting the global markers improves the plot visually
        #=
        perm = sortperm(x_m_glob.*lx.*dims[1].+y_m_glob)
        x_m_glob = x_m_glob[perm]
        y_m_glob = y_m_glob[perm]
        ρ_m_glob = ρ_m_glob[perm]
        μ_m_glob = μ_m_glob[perm]
        =#
        opts = (color=Int.(round.(exp.(ρ_m_glob))), size=(1200, 1150), margin=1mm, legend=false, yflip=true, markersize=3, markerstrokewidth=0)
        p1 = plot(scatter(x_m_glob, y_m_glob; opts...))
        
        if haskey(ENV,"GKSwstype") && ENV["GKSwstype"] == "nul"
            savefig(plot(p1),"viz_out/markers_$(step).png")
        else
            display(plot(p1))
        end
    end
    return
end

function plot_fields(Vx,Vy,μ_b,P,me,dims,Nx,Ny,dx,dy,step)

    Vx_glob, Vy_glob = gather_V_grid(Vx, Vy, me, dims, Nx, Ny)
    μ_glob , P_glob  = gather_V_grid(μ_b, P, me, dims, Nx, Ny)

    if me == 0
        opts = (size=(1200, 1000), margin=10mm, c=:inferno, yflip=true)
        
        x = [(ix - 1) * dx for ix = 1:size(μ_glob,1)]
        y = [(iy - 1) * dy for iy = 1:size(μ_glob,2)]
        x_p = [(ix - 1) * dx + 0.5dx for ix = 1:size(P_glob,1)]
        y_p = [(iy - 1) * dy + 0.5dy for iy = 1:size(P_glob,2)]
        x_vx = [(ix - 2) * dx for ix = 1:size(Vx_glob,1)]
        y_vx = [(iy - 1) * dy - 0.5dy for iy = 1:size(Vx_glob,2)]
        x_vy = [(ix - 1) * dx - 0.5dx for ix = 1:size(Vy_glob,1)]
        y_vy = [(iy - 2) * dy for iy = 1:size(Vy_glob,2)]

        p1 = heatmap(x, y, μ_glob'; title="μ_b", opts...)
        p2 = heatmap(x_p, y_p, P_glob'; title="Pressure", opts...)
        p3 = heatmap(x_vx, y_vx, Vx_glob'; title="Vx", opts...)
        p4 = heatmap(x_vy, y_vy, Vy_glob'; title="Vy", opts...)

        if haskey(ENV,"GKSwstype") && ENV["GKSwstype"] == "nul"
            savefig(plot(p1, p2, p3, p4),"viz_out/fields_$(step).png")
        else
            display(plot(p1, p2, p3, p4))
        end
    end
    return
end


function initializeMarkersCPU(comm,dims,coords,marker_density::Integer,lx,ly,dx,dy,Nx,Ny,RAND_MARKER_POS; rng=Random.GLOBAL_RNG)

    dxm = dx/marker_density
    dym = dy/marker_density

    xloBNDRY = coords[1] == 0
    xhiBNDRY = coords[1] == dims[1]-1
    yloBNDRY = coords[2] == 0
    yhiBNDRY = coords[2] == dims[2]-1

    x_less_hi = marker_density%2 == 0 ? dxm/2 : dxm
    y_less_hi = marker_density%2 == 0 ? dym/2 : dym

    x_less_lo = marker_density%2 == 0 ? dxm/2 : 0.0
    y_less_lo = marker_density%2 == 0 ? dym/2 : 0.0

    xlimlo = xloBNDRY ? dxm/2    : dx/2    + x_less_lo
    xlimhi = xhiBNDRY ? lx-dxm/2 : lx-dx/2 - x_less_hi
    ylimlo = yloBNDRY ? dym/2    : dy/2    + y_less_lo
    ylimhi = yhiBNDRY ? ly-dym/2 : ly-dy/2 - y_less_hi

    lo_less = Int(ceil(marker_density/2))
    hi_less = Int(floor(marker_density/2))

    Nmx = marker_density*(Nx-1) - (xloBNDRY ? 0 : lo_less) - (xhiBNDRY ? 0 : hi_less)
    Nmy = marker_density*(Ny-1) - (yloBNDRY ? 0 : lo_less) - (yhiBNDRY ? 0 : hi_less)
    Nm = Nmx*Nmy

    xcoords = LinRange(xlimlo, xlimhi, Nmx)
    ycoords = LinRange(ylimlo, ylimhi, Nmy)

    x_m = zeros(Nm)
    y_m = zeros(Nm)
    ρ_m = zeros(Nm)
    μ_m = zeros(Nm)

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
        if marker_density%2 == 1
            x_m, y_m, ρ_m, μ_m = exchangeMarkers!(comm,dims,[lx,ly],dx,dy,x_m,y_m,ρ_m,μ_m)
        end
    end
    
    return x_m, y_m, ρ_m, μ_m
end


"""
    setInitialMarkerProperties!(x_m, y_m, ρ_m, μ_m, x0, y0, density, viscosity)

Sets initial marker properties `ρ_m` and `μ_m` according to what
1. density(x_glob,y_glob)
2. viscosity(x_glob,y_glob)
evaluates to, where x_glob and y_glob describe global coordinates
"""
function setInitialMarkerProperties!(x_m, y_m, ρ_m, μ_m, x0, y0, density, viscosity)
    Nm = size(x_m,1)
    @assert (size(y_m,1) == Nm) && (size(ρ_m,1) == Nm) && (size(μ_m,1) == Nm)
    ρ_m .= density.(x0.+x_m, y0.+y_m)
    μ_m .= viscosity.(x0.+x_m, y0.+y_m)
    return
end

function example_call()
    Nt = 20
    Nx = 42
    Ny = 42
    Lx_glob = 10
    Ly_glob = 10

    μ_air, μ_matrix, μ_plume = 1e-2, 1e0, 1e-1  # Viscosity
    ρ_air, ρ_matrix, ρ_plume = 1e-3, 3.3, 3.2   # Density
    plume_x, plume_y = Lx_glob / 2, Ly_glob / 2 # plume midpoint
    plume_r = min(Lx_glob, Ly_glob) / 5         # plume radius
    air_height = 0.2 * Ly_glob                  # height of the 'sticky air' layer on top
    function density(x,y)
        if y < air_height
            ρ_air
        elseif (x - plume_x)^2 + (y - plume_y)^2 < plume_r^2
            ρ_plume
        else
            ρ_matrix
        end
    end
    function viscosity(x,y)
        if y < air_height
        μ_air
        elseif (x - plume_x)^2 + (y - plume_y)^2 < plume_r^2
            μ_plume
        else
            μ_matrix
        end
    end

    init_MPI = !MPI.Initialized()
    
    return StokesFlow2D(Nt, Nx, Ny, Lx_glob, Ly_glob, density, viscosity;
        RAND_MARKER_POS=true, plot_fields_live=false, plot_markers_live=true, save_to_file=true, print_info=true, init_MPI=init_MPI)
end

example_call()
