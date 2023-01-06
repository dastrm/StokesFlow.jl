using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using StaticArrays
import CUDA
@init_parallel_stencil(CUDA, Float64, 2)
include("../single_process/StokesSolver.jl")

using Plots,Plots.Measures
using Test
import Random

default(size=(1200,1000),framestyle=:box,label=false,grid=true,margin=8mm)

function plot_scaling()
    all_n = [64,128,192,256,320,384,448,512]
    T_effs = []
    for n=all_n
        T_eff = time_stokes_single_process(n)
        push!(T_effs,T_eff)
        @show n, T_eff
    end
    display(scatter(all_n,T_effs,xlabel="Nx=Ny",ylabel="T_eff",title="StokesSolver: Scaling on Single GPU",markersize=8))
end

function time_stokes_single_process(n)

    Nx=n
    Ny=n

    # --- PARAMETERS ---
    # time
    maxdisp = 0.5                               # dt is determined s.t. no marker moves further than maxdisp cells
    # physical parameters
    g_y = 9.81                                  # earth gravity
    lx, ly = 10, 10                             # domain size
    μ_air, μ_matrix, μ_plume = 1e-2, 1e0, 1e-1  # Viscosity
    ρ_air, ρ_matrix, ρ_plume = 1e-3, 3.3, 3.2   # Density, kg/m^3
    plume_x, plume_y = lx/2, ly/2               # plume midpoint
    plume_r = ly/5                              # plume radius
    air_height = 0.2*ly                         # height of the 'sticky air' layer on top
    # discretization parameters
    marker_density = 5                          # use this amount of markers per grid step per dimension

    # derived quantities
    dx, dy = lx/(Nx-1), ly/(Ny-1)               # grid resolution
    Nmx = (Nx-1)*marker_density                 # number of markers
    Nmy = (Ny-1)*marker_density
    Nm  = Nmx*Nmy

    # --- ARRAYS ---
    # marker CPU arrays for easy initializing with IC
    xy_m = zeros(2,Nm)                          # marker coords
    ρ_m  = zeros(Nm)                            # marker property: density
    μ_m  = zeros(Nm)                            # marker property: viscosity

    # grid array allocations
    # maybe Vx & Vy sizes need adjustments for marker interpolation (multi-GPU case, simplify single GPU). TODO
    P    = @zeros(Nx-1,Ny-1)
    Vx   = @zeros(Nx+2,Ny+1)                    # Velocity in x-direction, extended in x-direction in preparation for multi-processing
    Vy   = @zeros(Nx+1,Ny+2)                    # Velocity in y-direction, extended in y-direction in preparation for multi-processing
    ρ_vy = @zeros(Nx+1,Ny  )                    # Density on vy-nodes
    μ_b  = @zeros(Nx  ,Ny  )                    # Viscosity μ on basic nodes
    μ_p  = @zeros(Nx-1,Ny-1)                    # Viscosity μ on pressure nodes

    # additional arrays for Stokes Solver
    τxx   = @zeros(Nx-1,Ny-1)
    τyy   = @zeros(Nx-1,Ny-1)
    τxy   = @zeros(Nx  ,Ny  )
    ∇V    = @zeros(Nx-1,Ny-1)
    dτPt  = @zeros(Nx-1,Ny-1)
    Rx    = @zeros(Nx-2,Ny-1)
    Ry    = @zeros(Nx-1,Ny-2)
    dVxdτ = @zeros(Nx-2,Ny-1)
    dVydτ = @zeros(Nx-1,Ny-2)
    dτVx  = @zeros(Nx-2,Ny-1)
    dτVy  = @zeros(Nx-1,Ny-2)

    # additional arrays for marker -> grid interpolation
    interp_size = max.(size(ρ_vy),size(μ_b),size(μ_p)) # these represent the arrays that are filled by interpolation
    val_wt_sum = @zeros(interp_size)
    wt_sum = @zeros(interp_size)

    # coordinates for all grid points
    x    = [(ix-1)*dx       for ix=1:Nx  ] # basic nodes
    y    = [(iy-1)*dy       for iy=1:Ny  ]
    x_p  = [(ix-1)*dx+0.5dx for ix=1:Nx-1] # pressure nodes
    y_p  = [(iy-1)*dy+0.5dy for iy=1:Ny-1]
    x_vx = [(ix-2)*dx       for ix=1:Nx+2] # Vx nodes
    y_vx = [(iy-1)*dy-0.5dy for iy=1:Ny+1]
    x_vy = [(ix-1)*dx-0.5dx for ix=1:Nx+1] # Vy nodes
    y_vy = [(iy-2)*dy       for iy=1:Ny+2]
    x_ρ  = x_vy                            # nodes for ρ: same as Vy, but smaller in y
    y_ρ  = y_vy[2:end-1]
    # consistency checks
    @assert size(x_p ,1) == size(P   ,1) && size(y_p ,1) == size(P   ,2)
    @assert size(x_vx,1) == size(Vx  ,1) && size(y_vx,1) == size(Vx  ,2)
    @assert size(x_vy,1) == size(Vy  ,1) && size(y_vy,1) == size(Vy  ,2)
    @assert size(x_ρ,1)  == size(ρ_vy,1) && size(y_ρ ,1) == size(ρ_vy,2)

    # --- INITIAL CONDITIONS ---
    setInitialMarkerCoords!(xy_m, Nmx, Nmy, x, y, false)
    setInitialMarkerProperties!(xy_m, ρ_m, μ_m, Nm, μ_air, μ_matrix, μ_plume, ρ_air, ρ_matrix, ρ_plume, plume_x, plume_y, plume_r, air_height)

    # transform marker arrays to xPU arrays
    xy_m = Data.Array(xy_m)
    ρ_m  = Data.Array(ρ_m)
    μ_m  = Data.Array(μ_m)

    dt = 0.0

    # interpolate material properties to grid
    bilinearMarkerToGrid!(x_ρ[1],y_ρ[1],ρ_vy,xy_m,ρ_m,dx,dy, val_wt_sum,wt_sum)
    bilinearMarkerToGrid!(x[1]  ,y[1]  ,μ_b ,xy_m,μ_m,dx,dy, val_wt_sum,wt_sum)
    bilinearMarkerToGrid!(x_p[1],y_p[1],μ_p ,xy_m,μ_m,dx,dy, val_wt_sum,wt_sum)

    dt, T_eff = solveStokes!(P,Vx,Vy,ρ_vy,μ_b,μ_p,
                τxx, τyy, τxy, ∇V, dτPt, Rx, Ry, dVxdτ, dVydτ, dτVx, dτVy,
                g_y, dx, dy, Nx, Ny,
                dt, maxdisp; use_free_surface_stabilization=true,
                ϵ=1e-5,
                print_info=true)

    return T_eff

end

@views function setInitialMarkerCoords!(xy_m, Nmx, Nmy, xlims, ylims, RAND_MARKER_POS::Bool; rng=Random.GLOBAL_RNG)
    Nm = Nmx*Nmy
    @assert size(xy_m) == (2,Nm)
    dxm = (xlims[end]-xlims[1]) / Nmx
    dym = (ylims[end]-ylims[1]) / Nmy
    xcoords = LinRange(xlims[1]+0.5dxm,xlims[end]-0.5dxm,Nmx)
    ycoords = LinRange(ylims[1]+0.5dym,ylims[end]-0.5dym,Nmy)
    m = 1
    for ix=1:Nmx
        for iy=1:Nmy
            xy_m[1,m] = xcoords[ix]
            xy_m[2,m] = ycoords[iy]
            m += 1
        end
    end
    if RAND_MARKER_POS
        xy_m[1,:] .+= (rand(rng,Nm).-0.5).*dxm
        xy_m[2,:] .+= (rand(rng,Nm).-0.5).*dym
    end
    return nothing
end

@views function setInitialMarkerProperties!(xy_m, ρ_m, μ_m, Nm, μ_air, μ_matrix, μ_plume, ρ_air, ρ_matrix, ρ_plume, plume_x, plume_y, plume_r, air_height)
    @assert size(xy_m) == (2,Nm)
    @assert size(ρ_m) == size(μ_m) && size(μ_m,1) == Nm
    for m=1:Nm
        x,y = xy_m[:,m]
        if y < air_height
            ρ_m[m] = ρ_air
            μ_m[m] = μ_air
        elseif (x-plume_x)^2 + (y-plume_y)^2 < plume_r^2
            ρ_m[m] = ρ_plume
            μ_m[m] = μ_plume
        else
            ρ_m[m] = ρ_matrix
            μ_m[m] = μ_matrix
        end
    end
    return nothing
end

@views function bilinearMarkerToGrid!(x_grid_min,y_grid_min,val_grid,xy_m,val_m,dx,dy, val_wt_sum,wt_sum)

    Nx,Ny = size(val_grid)
    Nm = size(xy_m,2)

    val_wt_sum .= 0.0
    wt_sum     .= 0.0

    # interpolate by summing up the marker contributions to global arrays
    @parallel (1:Nm) atomicAddInterpolation(xy_m,val_m,wt_sum,val_wt_sum,Nx,Ny,x_grid_min,y_grid_min,dx,dy)

    # finally compute actual value from the sums
    @parallel (1:Nx,1:Ny) safeDivision(val_grid, Data.Array(val_wt_sum), Data.Array(wt_sum))

    return nothing
end

@parallel_indices (m) function atomicAddInterpolation(xy_m,val_m,wt_sum,val_wt_sum,Nx,Ny,x_grid_min,y_grid_min,dx,dy)
    xm = xy_m[1,m]
    ym = xy_m[2,m]
    val = val_m[m]

    # get indices and relative distance to top left node w.r.t marker m.
    # may be 0, when the marker is further left or up than the first grid node
    ix,iy,dxmij,dymij = topleftIndexRelDist(x_grid_min,y_grid_min,xm,ym,dx,dy)
    if !(ix>=0 && ix<=Nx && iy>=0 && iy<=Ny)
        @ps_println("ATTENTION: erroneous marker position")
    else

        # sum up weights, if the respective node exists
        if iy>0
            if ix>0
                # 1) top left
                w = (1-dxmij)*(1-dymij)
                CUDA.@atomic wt_sum[ix,iy] += w
                CUDA.@atomic val_wt_sum[ix,iy] += w*val
            end
            if ix<Nx
                # 2) top right
                w = dxmij*(1-dymij)
                CUDA.@atomic wt_sum[ix+1,iy] += w
                CUDA.@atomic val_wt_sum[ix+1,iy] += w*val
            end
        end
        if iy<Ny
            if ix>0
                # 3) bottom left
                w = (1-dxmij)*dymij
                CUDA.@atomic wt_sum[ix,iy+1] += w
                CUDA.@atomic val_wt_sum[ix,iy+1] += w*val
            end
            if ix<Nx
                # 4) bottom right
                w = dxmij*dymij
                CUDA.@atomic wt_sum[ix+1,iy+1] += w
                CUDA.@atomic val_wt_sum[ix+1,iy+1] += w*val
            end
        end
    end

    return nothing
end

@parallel_indices (ix,iy) function safeDivision(result,numerator,denominator)
    denom = denominator[ix,iy]
    if denom != 0.0
        result[ix,iy] = numerator[ix,iy] / denom
    else
        @ps_println("PROBLEM: no markers close enough for value interpolation at node: ix = ", ix, ", iy = ", iy)
    end
    return nothing
end

function topleftIndexRelDist(x_grid_min, y_grid_min, x, y, dx, dy)
    # indices: may be out of bounds if the grid does not cover (x,y)
    ix = floor(Int,(x-x_grid_min)/dx) + 1
    iy = floor(Int,(y-y_grid_min)/dy) + 1
    # position of top left node with index (ix,iy)
    x_ix = x_grid_min + (ix-1)*dx
    y_iy = y_grid_min + (iy-1)*dy
    # compute relative distances
    dxij = (x-x_ix)/dx
    dyij = (y-y_iy)/dy
    return ix,iy,dxij,dyij
end

plot_scaling()
