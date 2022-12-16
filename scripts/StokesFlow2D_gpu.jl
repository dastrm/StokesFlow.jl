using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using StaticArrays
import CUDA
@init_parallel_stencil(CUDA, Float64, 2)
include("StokesSolver.jl")

using Plots,Plots.Measures
using Test
import Random

default(size=(1200,1000),framestyle=:box,label=false,grid=false,margin=10mm)

"""
Input args:
Nt              : number of timesteps
Nx, Ny          : number of grid points
RAND_MARKER_POS : whether to add random perturbation to initial marker coords
do_plot         : whether to create Plots
print_info      : whether any info is printed to console

Output: Currently just Vy, an array of size (Nx, Ny+1)
"""
@views function StokesFlow2D(;Nt=20, Nx=35, Ny=45, RAND_MARKER_POS::Bool=true, do_plot::Bool=true, print_info::Bool=true)

    # --- PARAMETERS ---
    # time
    maxdisp = 0.5                               # dt is determined s.t. no marker moves further than maxdisp cells
    # physical parameters
    g_y = 9.81                                  # earth gravity, m/s^2
    lx, ly = 100000, 100000                     # domain size, m
    μ_air, μ_matrix, μ_plume = 1e17, 1e19, 1e18 # Viscosity, Pa*s
    ρ_air, ρ_matrix, ρ_plume = 1   , 3300, 3200 # Density, kg/m^3
    plume_x, plume_y = lx/2, ly/2               # plume midpoint
    plume_r = 20000                             # plume radius
    air_height = 0.2*ly                         # height of the 'sticky air' layer on top
    # discretization parameters
    marker_density = 5                          # use this amount of markers per grid step per dimension
    # random numbers for initial marker postions
    Random.seed!(42)                            # seed default RNG for random marker positions. TODO multi-xPU: should be dependent on rank, or do not seed?

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
    Vx   = @zeros(Nx  ,Ny+1)                    # Velocity in x-direction
    Vy   = @zeros(Nx+1,Ny  )                    # Velocity in y-direction
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
    x_vx = x                               # Vx nodes
    y_vx = [(iy-1)*dy-0.5dy for iy=1:Ny+1]
    x_vy = [(ix-1)*dx-0.5dx for ix=1:Nx+1] # Vy nodes
    y_vy = y
    # consistency checks
    @assert size(x_p ,1) == size(P ,1) && size(y_p ,1) == size(P ,2)
    @assert size(x_vx,1) == size(Vx,1) && size(y_vx,1) == size(Vx,2)
    @assert size(x_vy,1) == size(Vy,1) && size(y_vy,1) == size(Vy,2)
    @assert size(ρ_vy)   == size(Vy)

    # --- INITIAL CONDITIONS ---
    setInitialMarkerCoords!(xy_m, Nmx, Nmy, x, y, RAND_MARKER_POS::Bool)
    setInitialMarkerProperties!(xy_m, ρ_m, μ_m, Nm, μ_air, μ_matrix, μ_plume, ρ_air, ρ_matrix, ρ_plume, plume_x, plume_y, plume_r, air_height)
    if do_plot
        vline(x)
        hline!(y)
        display(scatter!(xy_m[1,:],xy_m[2,:],color=Int.(round.(ρ_m)),xlims=(x[1],x[end]),ylims=(y[1],y[end]),aspect_ratio=1,yflip=true,legend=false,markersize=3,markerstrokewidth=0))
    end
    # transform marker arrays to xPU arrays
    xy_m = Data.Array(xy_m)
    ρ_m  = Data.Array(ρ_m)
    μ_m  = Data.Array(μ_m)

    # --- TIMESTEPPING ---
    dt = 0.0
    t_tot = 0.0
    for t=1:Nt
        if print_info
            @show t, t_tot
        end

        # interpolate material properties to grid
        # TODO implement fast GPU version
        bilinearMarkerToGrid!(x_vy[1],y_vy[1],ρ_vy,xy_m,ρ_m,dx,dy, val_wt_sum,wt_sum)
        bilinearMarkerToGrid!(x[1]   ,y[1]   ,μ_b ,xy_m,μ_m,dx,dy, val_wt_sum,wt_sum)
        bilinearMarkerToGrid!(x_p[1] ,y_p[1] ,μ_p ,xy_m,μ_m,dx,dy, val_wt_sum,wt_sum)

        # calculate velocities on grid
        dt = solveStokes!(P,Vx,Vy,ρ_vy,μ_b,μ_p,
                τxx, τyy, τxy, ∇V, dτPt, Rx, Ry, dVxdτ, dVydτ, dτVx, dτVy,
                g_y, dx, dy, Nx, Ny,
                dt, maxdisp; use_free_surface_stabilization=true,
                print_info=print_info)

        # plot current state
        if do_plot
            showPlot(x,y,x_p,y_p,x_vx,y_vx,x_vy,y_vy, P,Vx,Vy,ρ_vy,μ_b,μ_p, xy_m,ρ_m, lx,ly)
            #display(scatter(Array(xy_m)[1,:],Array(xy_m)[2,:],color=Int.(round.(Array(ρ_m))),xlims=(x[1],x[end]),ylims=(y[1],y[end]),aspect_ratio=1,yflip=true,legend=false,markersize=3,markerstrokewidth=0))
        end

        # move markers
        @assert dt ≈ maxdisp*min(dx/maximum(Vx),dy/maximum(Vy))
        @parallel (1:Nm) moveMarkersRK4!(xy_m,Vx,Vy,x_vx[1],y_vx[1],x_vy[1],y_vy[1],dt,lx,ly,dx,dy)

        t_tot += dt
    end


    return Array(Vx)
end


function bilinearInterp(v1,v2,v3,v4,dxij,dyij)
    s1 = (1-dxij)*(1-dyij)*v1
    s2 =    dxij *(1-dyij)*v2
    s3 = (1-dxij)*   dyij*v3
    s4 =    dxij *   dyij*v4
    return s1+s2+s3+s4
end

#interpolates grid velocities to postion x,y
function interpolateV(x,y,Vx,Vy,x_vx_min,y_vx_min,x_vy_min,y_vy_min,dx,dy)
    # Interpolate Vx
    ix,iy,dxij,dyij = topleftIndexRelDist(x_vx_min,y_vx_min,x,y,dx,dy)
    #index range failsafe, in case advection moves the particles out of domain
    if ix < 1 ix=1; dxij=0.0 end
    if iy < 1 iy=1; dyij=0.0 end
    if ix >= size(Vx,1) ix=size(Vx,1)-1; dxij=1.0 end
    if iy >= size(Vx,2) iy=size(Vx,2)-1; dyij=1.0 end
    v1 = Vx[ix  ,iy  ]
    v2 = Vx[ix+1,iy  ]
    v3 = Vx[ix  ,iy+1]
    v4 = Vx[ix+1,iy+1]
    vx = bilinearInterp(v1,v2,v3,v4,dxij,dyij)
    # Continuity-based velocity correction for Vx: the interpolated field will have zero divergence!
    # TODO: no correction if too close to boundary: Vx and Vy arrays must have more ghost cells for multi-GPU!
    correction = 0.0
    # right half of cell => extend stencil to the right
    if dxij > 0.5 && ix+2 <= size(Vx,1)
        v5 = Vx[ix+2,iy  ]
        v6 = Vx[ix+2,iy+1]
        correction = 0.5*(dxij-0.5)^2*(
                        (1-dyij)*(v1-2v2+v5) +
                           dyij *(v3-2v4+v6))
    # left  half of cell => extend stencil to the left
    elseif dxij < 0.5 && ix-1 >= 1
        v5 = Vx[ix-1,iy]
        v6 = Vx[ix-1,iy+1]
        correction = 0.5*(dxij-0.5)^2*(
                        (1-dyij)*(v5-2v1+v2) +
                           dyij *(v6-2v3+v4))
    end
    vx += correction

    # Interpolate Vy
    ix,iy,dxij,dyij = topleftIndexRelDist(x_vy_min,y_vy_min,x,y,dx,dy)
    #index range failsafe, in case advection moves the particles out of domain
    if ix < 1 ix=1; dxij=0.0 end
    if iy < 1 iy=1; dyij=0.0 end
    if ix >= size(Vy,1) ix=size(Vy,1)-1; dxij=1.0 end
    if iy >= size(Vy,2) iy=size(Vy,2)-1; dyij=1.0 end
    v1 = Vy[ix,iy]
    v2 = Vy[ix+1,iy]
    v3 = Vy[ix,iy+1]
    v4 = Vy[ix+1,iy+1]
    vy = bilinearInterp(v1,v2,v3,v4,dxij,dyij)
    # Continuity-based velocity correction for Vy: the interpolated field will have zero divergence!
    # TODO: no correction if too close to boundary: Vx and Vy arrays must have more ghost cells for multi-GPU!
    correction = 0.0
    # lower half of cell => extend stencil to the bottom
    if dyij > 0.5 && iy+2 <= size(Vy,2)
        v5 = Vy[ix  ,iy+2]
        v6 = Vy[ix+1,iy+2]
        correction = 0.5*(dyij-0.5)^2*(
                        (1-dxij)*(v1-2v3+v5) +
                           dxij *(v2-2v4+v6))
    # upper half of cell => extend stencil to the top
    elseif dyij < 0.5 && iy-1 >=1
        v5 = Vy[ix  ,iy-1]
        v6 = Vy[ix+1,iy-1]
        correction = 0.5*(dyij-0.5)^2*(
                        (1-dxij)*(v5-2v1+v3) +
                           dxij *(v6-2v2+v4))
    end
    vx += correction

    return vx, vy
end

@parallel_indices (m) function moveMarkersRK4!(xy_m,Vx,Vy,x_vx_min,y_vx_min,x_vy_min,y_vy_min,dt,lx,ly,dx,dy)

    # Runge-Kutta 4th order
    rk4_dt = @SVector [0.0, 0.5dt, 0.5dt, dt]
    rk4_wt = @SVector [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0];

    x_old = xy_m[1,m] # old position
    y_old = xy_m[2,m]
    vx_eff, vy_eff = 0.0, 0.0 # 'effective' velocity for explicit update: x_new = x_old + v_eff*dt
    vx_rk , vy_rk  = 0.0, 0.0 # velocity at previous/current point

    for it=1:4 # loop over points A-D
        # position of current point based on previous point velocities
        x_rk = x_old + rk4_dt[it]*vx_rk
        y_rk = y_old + rk4_dt[it]*vy_rk

        # interpolate velocity to current point
        vx_rk, vy_rk = interpolateV(x_rk,y_rk,Vx,Vy,x_vx_min,y_vx_min,x_vy_min,y_vy_min,dx,dy)

        # apply RK4 scheme: add up weighted velocities
        vx_eff += rk4_wt[it]*vx_rk
        vy_eff += rk4_wt[it]*vy_rk
    end

    # move particle
    x_new = x_old + vx_eff*dt
    y_new = y_old + vy_eff*dt

    # explicitly restrict particles to stay on domain
    # (optional, does not really change anything if BC correctly implemented and dt small enough)
    # !! TODO: CHANGE, if global domain is not 0-lx and 0-ly !!
    x_new = min(max(x_new,0),lx)
    y_new = min(max(y_new,0),ly)

    # write back updated positions
    xy_m[1,m] = x_new
    xy_m[2,m] = y_new

    return nothing
end


function showPlot(x,y,x_p,y_p,x_vx,y_vx,x_vy,y_vy, P,Vx,Vy,ρ_vy,μ_b,μ_p,xy_m,ρ_m, lx,ly)

    #p1 = heatmap(x_vy ,  y_vy, Array(ρ_vy)' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="ρ_vy")
    p2 = heatmap(x     ,  y   , Array(μ_b )' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="μ_b" )
    #p3 = heatmap(x_p  ,  y_p , Array(μ_p )' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="μ_p" )
    #p4 = scatter(xy_m[1,:],xy_m[2,:],color=Int.(round.(ρ_m)),xlims=(x[1],x[end]),ylims=(y[1],y[end]),aspect_ratio=1,yflip=true,legend=false,markersize=3,markerstrokewidth=0)

    p5 = heatmap(x_p ,  y_p, Array(P)' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Pressure")
    p6 = heatmap(x_vx, y_vx, Array(Vx)', yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Vx")
    p7 = heatmap(x_vy, y_vy, Array(Vy)', yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Vy")

    display(plot(p2,p5,p6,p7))
    #display(plot(p4))
    return nothing
end

# compute indices ix, iy of top left node w.r.t position (x,y)
# as well as relative distances (x_grid[ix]-x)/dx, (y_grid[iy]-y)/dy
# !!! ix,iy may be out of bounds if the grid does not cover (x,y), this is not checked here
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

@views function bilinearMarkerToGrid!(x_grid_min,y_grid_min,val_grid,xy_m,val_m,dx,dy, val_wt_sum,wt_sum)
    #=
        # Currently, a naive version without cell-lists is implemented:
        1.) each thread gets a particle, computes its weights and contribution to each of the four nodes
        2.) perform atomic add on two global nodal arrays: sum(weights) and sum(weights*particleValue)
        3.) do division: val_grid = sum(weights*particleValue) / sum(weights) 

        TODO:
        1.) build cell list according to x_grid_min, y_grid_min, dx and dy
        2.) set global arrays for sum(weights) and sum(weights*particleValue) to zero
        3.) assign a warp to each cell. each thread does:
            - loop over "his" particles in a warp. should be mostly 1, sometimes 0 or >1 (with ~25 particles/cell)
            - inside this loop:
                1. load particle position
                2. compute weights dxmij, dymij
                3. add up weights and weights*particleValue for the particles
            - after the loop: do warp-level reduction (sum) of 1. weights and 2. weights*particleValue for each of the four nodes
        4.) then, use shared memory for reduction over warps in each block:
                - calculate sum(weights) and sum(weights*particleValue) for each node,
                - then add this atomically to global arrays (to sync with other blocks on nodes located on the block boundaries)
        5.) launch a separate kernel for the division sum(weights*particleValue)/sum(weights) for each nodes

        Possible Variations:
        1.) have a global array that has 8 entries per node; for sum(weights) and sum(weights*particleValue) for each of the 4 cells next to the node
            => instead of using shared memory and using an atomic add, each warp just writes its 8 values to the global array
            This should be easier and might even be faster (avoid atomic add) ???
        2.) use one thread per cell: loop over all particles in a cell
    =#

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

"""
Sets initial coordinates and properties of the markers
x, y are coordinates of the basic grid nodes
xlims, ylims contain domain lower and upper limits at indices [1] and [end], respectively
"""
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

"""
Sets initial marker properties ρ_m and μ_m according to whether their coordinates correspond to
1. sticky air
2. plume / inclusion
3. the surrounding matrix
"""
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


StokesFlow2D()


@testset "StokesFlow2D_gpu" begin
    nt    = 10
    nx,ny = 35,45
    # tests should not depend on a rng seed, see the Warning at https://docs.julialang.org/en/v1/stdlib/Random/
    result = StokesFlow2D(;Nt=nt,Nx=nx,Ny=ny,RAND_MARKER_POS=false,do_plot=false,print_info=false)
    inds   = [181, 219, 388, 444, 637, 920, 1049, 1074, 1223, 1367]
    refs   = [5.16167425967578e-10, -4.448226006752059e-10, -1.5042959025661068e-9, 4.969181007693054e-9, -1.466557924043276e-9, 2.1178605832979315e-9, -3.027897978585612e-10, -2.0912258038281033e-9, -5.849238321416495e-10, 2.5293112415907947e-10]
    @test all(refs .≈ result[inds])
end
