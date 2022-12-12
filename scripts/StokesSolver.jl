if !@isdefined(USE_GPU)
    const USE_GPU = true
end
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics

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

# STOKES SOLVER
# Currently this is just:
# - copy-paste from StokesSolver_prototype.jl
# - additional free surface stabilization (using the density implicitly advected to the next timestep)
#   This is done to avoid oscillations of the free surface.
@views function solveStokes!(P,Vx,Vy,ρ_vy,μ_b,μ_p,
                            τxx, τyy, τxy, ∇V, dτPt, Rx, Ry, dVxdτ, dVydτ, dτVx, dτVy,
                            g_y, dx, dy, Nx, Ny,
                            dt, maxdisp; use_free_surface_stabilization::Bool=true)

    Vdmp      = 4.0
    Vsc       = 0.25        # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc      = 1.0/4.0 
    min_dxy2  = min(dx,dy)^2
    max_nxy   = max(Nx,Ny)
    dampX     = 1.0-Vdmp/Nx # damping term for the x-momentum equation
    dampY     = 1.0-Vdmp/Ny # damping term for the y-momentum equation

    if !use_free_surface_stabilization
        dt=0.0
    end

    # PT setup and iterations
    @parallel compute_timesteps!(dτVx, dτVy, dτPt, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    ncheck = 500
    t1 = 0; itert1 = 11
    ϵ = 0.01 # tol
    err_evo1=[]; err_evo2=[]
    err = 2ϵ; iter=1; niter=0; iterMax=100000
    while err > ϵ && iter <= iterMax
        if (iter==itert1) t1 = Base.time() end

        # TODO improve kernels
        # TODO make var names consistent
        @parallel compute_P!(∇V, P, Vx, Vy, dτPt, dx, dy)
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
    A_eff    = (3*2)/1e9*Nx*Ny*sizeof(Data.Number) 
    T_eff    = A_eff/t_it
    @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", niter, err, t2-t1, round(T_eff, sigdigits=2))

    if !use_free_surface_stabilization
        dt = maxdisp*min(dx/maximum(Vx),dy/maximum(Vy))
    end

    return dt
end

@parallel function compute_timesteps!(dτVx, dτVy, dτPt, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    @all(dτVx) = Vsc*min_dxy2/@av_xa(μ_p)/4.1
    @all(dτVy) = Vsc*min_dxy2/@av_ya(μ_p)/4.1
    @all(dτPt) = Ptsc*4.1*@all(μ_p)/max_nxy
    return
end

@parallel function compute_P!(∇V, Pt, Vx, Vy, dτPt, dx, dy)
    @all(∇V)  = @d_xi(Vx)/dx + @d_yi(Vy)/dy
    @all(Pt)  = @all(Pt) - @all(dτPt)*@all(∇V)
    return
end

@parallel function compute_τ!(∇V, τxx, τyy, τxy, Vx, Vy, μ_p, μ_τxy, dx, dy)
    @all(τxx) = 2.0*@all(μ_p)*(@d_xi(Vx)/dx - 1.0/3.0*@all(∇V))
    @all(τyy) = 2.0*@all(μ_p)*(@d_yi(Vy)/dy - 1.0/3.0*@all(∇V))
    @all(τxy) = 2.0*@all(μ_τxy)*(0.5*(@d_ya(Vx)/dy + @d_xa(Vy)/dx))
    return
end

@parallel function compute_dV!(Rx, Ry, dVxdτ, dVydτ, Pt, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, dampX, dampY, dx, dy, dt)
    @all(Rx)    = @d_xa(τxx)/dx + @d_yi(τxy)/dy - @d_xa(Pt)/dx
    @all(Ry)    = @d_ya(τyy)/dy + @d_xi(τxy)/dx - @d_ya(Pt)/dy + 
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
