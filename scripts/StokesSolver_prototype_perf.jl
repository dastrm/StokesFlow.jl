#=
This script solves the 2D Stokes & Continuity equations for given arrays of μ, ρ.

∂τxx/∂x + ∂τxy/∂y - ∂P/∂x = 0
∂τyx/∂x + ∂τyy/∂y - ∂P/∂y = -ρ*g*y
∂vx/∂x + ∂vy/∂y = 0
τxx = 2μ*∂vx/∂x
τyy = 2μ*∂vy/∂y
τxy = τyx = μ*(∂vx/∂y + ∂vy/∂x)

Additionally, dt is calculated as dt = maxdisp*min(dx/maximum(Vx),dy/maximum(Vy)),
and for the rhs of the y-Stokes equation, the density field ρ is implicitly advected
to the next timestep using Vx, Vy. ("Free surface stabilization")

Otherwise, PT iterations are implemented similarly to the 2D minapp of ParallelStencil.jl
https://github.com/omlins/ParallelStencil.jl/blob/main/miniapps/Stokes2D.jl
=#

#=
What's still missing:
- Performance (Kernel fusing, tuning relaxation parameters, ...)
  see e.g. https://github.com/PTsolvers/PseudoTransientStokes.jl/blob/main/scripts/Stokes2D_ve_perf.jl
=#

const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, Test

# ADDITIONAL PARALLEL STENCIL MACROS, needed for free surface stabilization
import ..ParallelStencil: INDICES
ix, iy = INDICES[1], INDICES[2]
ixi, iyi = :($ix+1), :($iy+1)
# average in both dimension, and inner elements in y. corresponds to @av(@inn_y(..))
macro av_inn_y(A::Symbol)  esc(:(($A[$ix  ,$iyi ] + $A[$ix+1,$iyi ] + $A[$ix,$iyi+1] + $A[$ix+1,$iyi+1])*0.25 )) end
# central finite differences in x, inner elements in y. corresponds to @d_xi(@av_x(..))
macro   d_xi_2(A::Symbol)  esc(:( $A[$ix+2,$iyi ] - $A[$ix  ,$iyi] )) end
# central finite differences in y, inner elements in x. corresponds to @d_yi(@av_y(..))
macro   d_yi_2(A::Symbol)  esc(:( $A[$ixi ,$iy+2] - $A[$ixi ,$iy ] )) end

@views function Stokes2D()
    # Physics
    lx, ly = [100000, 100000]./100000 # domain size
    μ_air, μ_matrix, μ_plume = [1e17, 1e19, 1e18]./1e19 # Viscosity
    ρ_air, ρ_matrix, ρ_plume = [1   , 3300, 3200]./1000 # Density
    g_y = 9.81
    plume_x, plume_y = lx/2, ly/2 # plume midpoint
    plume_r = 20000 / 100000 # plume radius
    air_height = 0.2*ly

    use_free_surface_stabilization = true
    maxdisp = 0.5

    # Numerics
    nx, ny = 127,127#35, 45

    # Derived Numerics
    dx, dy = lx/(nx-1), ly/(ny-1)

    # Array allocations
    P    = @zeros(nx-1,ny-1)
    τxx  = @zeros(nx-1,ny-1) #p-nodes
    τyy  = @zeros(nx-1,ny-1) #p-nodes
    Vx   = @zeros(nx  ,ny+1)
    Vy   = @zeros(nx+1,ny  )
    τxy  = @zeros(nx  ,ny  )

    # coordinates for points
    x    = [(ix-1)*dx       for ix=1:nx  ] # basic nodes = τ_xy-nodes
    y    = [(iy-1)*dy       for iy=1:ny  ]
    x_p  = [(ix-1)*dx+0.5dx for ix=1:nx-1] # pressure nodes
    y_p  = [(iy-1)*dy+0.5dy for iy=1:ny-1]
    x_vx = x                               # Vx nodes
    y_vx = [(iy-1)*dy-0.5dy for iy=1:ny+1]
    x_vy = [(ix-1)*dx-0.5dx for ix=1:nx+1] # Vy nodes
    y_vy = y
    x_τxy= x
    y_τxy= y

    # set density & viscosity
    ρ_vy  = zeros(nx+1,ny  ) # on Vy-nodes
    μ_b = zeros(nx  ,ny  ) # on τxy-nodes
    μ_p   = zeros(nx-1,ny-1) # on P-nodes (also τxx-, τyy-nodes)
    set_properties!(x_p  ,y_p  ,μ_p  ,plume_x,plume_y,plume_r,μ_matrix,μ_plume,μ_air,air_height) # viscosity: P-nodes
    set_properties!(x_τxy,y_τxy,μ_b,plume_x,plume_y,plume_r,μ_matrix,μ_plume,μ_air,air_height) # viscosity: τxy-nodes
    set_properties!(x_vy ,y_vy ,ρ_vy ,plume_x,plume_y,plume_r,ρ_matrix,ρ_plume,ρ_air,air_height) # density: Vy-nodes
    ρ_vy  = Data.Array(ρ_vy)
    μ_b = Data.Array(μ_b)
    μ_p   = Data.Array(μ_p)

    #=
    # plot density & viscosity
    p1 = heatmap(x_p  ,y_p  ,Array(μ_p)'  ,yflip=true,title="Viscosity μ_p"  ,xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    p2 = heatmap(x_τxy,y_τxy,Array(μ_b)',yflip=true,title="Viscosity μ_b",xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    p3 = heatmap(x_vy ,y_vy ,Array(ρ_vy)' ,yflip=true,title="Density ρ_vy"   ,xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    display(plot(p1,p2,p3))
    #display(plot(p1)); display(plot(p2)); display(plot(p3))
    =#

    # more arrays
    ∇V        = @zeros(nx-1,ny-1)
    dτP      = @zeros(nx-1,ny-1)
    Rx        = @zeros(nx-2,ny-1)
    Ry        = @zeros(nx-1,ny-2)
    dVxdτ     = @zeros(nx-2,ny-1)
    dVydτ     = @zeros(nx-1,ny-2)
    dτVx      = @zeros(nx-2,ny-1)
    dτVy      = @zeros(nx-1,ny-2)
    Vdmp      = 4.0
    Vsc       = 0.25#1.0         # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc      = 1.0/4.0 
    min_dxy2  = min(dx,dy)^2
    max_nxy   = max(nx,ny)
    dampX     = 1.0-Vdmp/nx # damping term for the x-momentum equation
    dampY     = 1.0-Vdmp/ny # damping term for the y-momentum equation
    
    dt = 0.0

    # PT setup and iterations
    @parallel compute_timesteps!(dτVx, dτVy, dτP, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    ncheck = 500
    t1 = 0; itert1 = 11
    ϵ = 1e-9 # tol
    err_evo1=[]; err_evo2=[]
    err = 2ϵ; iter=1; niter=0; iterMax=100000
    while err > ϵ && iter <= iterMax
        if (iter==itert1) t1 = Base.time() end

        # TODO improve kernels
        @parallel compute_P!(∇V, P, Vx, Vy, dτP, dx, dy)
        @parallel compute_τ!(∇V, τxx, τyy, τxy, Vx, Vy, μ_p, μ_b, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVxdτ, dVydτ, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, dampX, dampY, dx, dy, dt)
        @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)

        # Free slip BC
        @parallel (1:size(Vx,1)) bc_y_noflux!(Vx)
        @parallel (1:size(Vx,2)) bc_x_zero!(Vx)
        @parallel (1:size(Vy,1)) bc_y_zero!(Vy)
        @parallel (1:size(Vy,2)) bc_x_noflux!(Vy)

        if use_free_surface_stabilization
            dt = maxdisp*min(dx/maximum(Vx),dy/maximum(Vy))
        end

        if iter%ncheck == 0
            # TODO compute error better
            mean_Rx = mean(abs.(Rx)); mean_Ry = mean(abs.(Ry)); mean_∇V = mean(abs.(∇V))
            err = maximum([mean_Rx, mean_Ry, mean_∇V])
            push!(err_evo1, err); push!(err_evo2,iter)
            #@printf("Total steps = %d, err = %1.3e [mean_Rx=%1.3e, mean_Ry=%1.3e, mean_∇V=%1.3e] \n", iter, err, mean_Rx, mean_Ry, mean_∇V)
        end

        iter+=1; niter+=1
    end
    t2 = Base.time()
    t_it = (t2-t1)/(niter-itert1+1)
    A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number) 
    T_eff    = A_eff/t_it
    @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", niter, err, t2-t1, round(T_eff, sigdigits=2))

    #=
    # Visualization
    p1 = heatmap(x_p ,  y_p, Array(P)' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Pressure")
    p2 = heatmap(x_vy, y_vy, Array(Vy)', yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Vy")
    #p4 = heatmap(X[2:end-1], Yv[2:end-1], log10.(abs.(Array(Ry)')), aspect_ratio=1, xlims=(X[2],X[end-1]), ylims=(Yv[2],Yv[end-1]), c=:inferno, title="log10(Ry)")
    p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
    p6 = heatmap(x_vx, y_vx, Array(Vx)', yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Vx")
    #display(plot(p1, p2, p4, p5))
    display(plot(p1,p2,p5,p6))
    =#

    return Array(Vx), Array(Vy)
end


@parallel function compute_timesteps!(dτVx, dτVy, dτP, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    @all(dτVx) = Vsc*min_dxy2/@av_xa(μ_p)/4.1
    @all(dτVy) = Vsc*min_dxy2/@av_ya(μ_p)/4.1
    @all(dτP)  = Ptsc*4.1*@all(μ_p)/max_nxy
    return
end

@parallel function compute_P!(∇V, P, Vx, Vy, dτP, dx, dy)
    @all(∇V)  = @d_xi(Vx)/dx + @d_yi(Vy)/dy
    @all(P)   = @all(P) - @all(dτP)*@all(∇V)
    return
end

@parallel function compute_τ!(∇V, τxx, τyy, τxy, Vx, Vy, μ_p, μ_b, dx, dy)
    @all(τxx) = 2.0*@all(μ_p)*(@d_xi(Vx)/dx - 1.0/3.0*@all(∇V))
    @all(τyy) = 2.0*@all(μ_p)*(@d_yi(Vy)/dy - 1.0/3.0*@all(∇V))
    @all(τxy) = 2.0*@all(μ_b)*(0.5*(@d_ya(Vx)/dy + @d_xa(Vy)/dx))
    return
end

@parallel function compute_dV!(Rx, Ry, dVxdτ, dVydτ, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, dampX, dampY, dx, dy, dt)
    @all(Rx)    = @d_xa(τxx)/dx + @d_yi(τxy)/dy - @d_xa(P)/dx
    @all(Ry)    = @d_ya(τyy)/dy + @d_xi(τxy)/dx - @d_ya(P)/dy + 
                    g_y*(@inn(ρ_vy) - dt*(@av_inn_y(Vx)*@d_xi_2(ρ_vy)/(2*dx) + @inn(Vy)*@d_yi_2(ρ_vy)/(2*dy)))
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    return
end

@parallel function compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
    @inn(Vx) = @inn(Vx) + @all(dτVx)*@all(dVxdτ)
    @inn(Vy) = @inn(Vy) + @all(dτVy)*@all(dVydτ)
    return
end

# side boundaries
@parallel_indices (iy) function bc_x_zero!(A::Data.Array)
    A[1  , iy] = 0.0
    A[end, iy] = 0.0
    return
end
@parallel_indices (iy) function bc_x_noflux!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

# horizontal boundaries
@parallel_indices (ix) function bc_y_zero!(A::Data.Array)
    A[ix, 1  ] = 0.0
    A[ix, end] = 0.0
    return
end
@parallel_indices (ix) function bc_y_noflux!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end


# This function sets material properties in arr depending on where the node is
function set_properties!(x,y,arr,plume_x,plume_y,plume_r,matrix,plume,air,air_height)
    nx, ny = size(arr)
    @assert nx <= size(x,1) && ny <= size(y,1)
    for ix=1:nx
        for iy=1:ny
            val = 0.0
            if y[iy] < air_height
                val = air
            elseif (x[ix]-plume_x)^2 + (y[iy]-plume_y)^2 <= plume_r^2
                val = plume
            else
                val = matrix
            end
            arr[ix,iy] = val
        end
    end
    return nothing
end

#Stokes2D()

@testset "StokesPrototype" begin
    Vx, Vy = Stokes2D() # with Nx = Ny = 127
    indsx = [3, 56, 60, 90, 99]
    indsy = [28, 68, 95, 96, 127]
    refsX = [-0.00037740176256595914 6.714046038354702e-5 0.00025308827285352 0.00025441536351831996 0.00023890573893336602; -0.002592175411900232 0.0005020525259329576 0.0019509822422144605 0.0018969715666351265 0.0010950028140514069; -0.0013559074661861751 0.0002549196102522001 0.0010425458839549392 0.001009294888462192 0.0005616178216118598; 0.005118644439457764 -0.0012976498989358668 -0.003580674973359176 -0.0035410459393814256 -0.0025608227116031244; 0.004590773065094249 -0.000972901497461574 -0.003130724328322405 -0.0031198745457242235 -0.0025232813527596046]
    refsY = [0.00316821054109413 0.006524368487621358 0.003978255591168619 0.0038510995001933995 0.0; -0.004026809431748747 -0.009951698997411856 -0.004845877202480422 -0.004647055882448395 0.0; -0.004545379157896374 -0.010981561240649675 -0.0054485849911919335 -0.005213888559230767 0.0; -0.000398421923889966 0.0010221245300420553 -0.0005914086732068255 -0.000600005663732289 0.0; 0.001232186058094373 0.00359245103372998 0.001402707747374657 0.001330987051017147 0.0]
    @test all(isapprox.(refsX, Vx[indsx,indsy]; atol=1e-7))
    @test all(isapprox.(refsY, Vy[indsx,indsy]; atol=1e-7))
end
