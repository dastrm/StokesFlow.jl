# ParallelStencil must already be initalized !!!
# ImplicitGlobalGrid must be initialized before calling solveStokes!(..)
using Printf, ParallelStencil, ImplicitGlobalGrid
import MPI, Statistics

"""
    solveStokes!(P, Vx_pad, Vy_pad, ρ_vy, μ_b, μ_p, τxx, τyy, τxy, ∇V, dτPt, Rx, Ry, dVxdτ, dVydτ, dτVx, dτVy, Vx_s, Vy_s,
        g_y, dx, dy, Nx, Ny, dt, maxdisp, comm; use_free_surface_stabilization::Bool=true, ϵ=1e-8, print_info::Bool=true)

Implements a parallel pseudo-transcient Stokes solver

`Vx_s` & `Vy_s` denote the 'small' (unpadded) arrays, Vx_pad, Vy_pad are padded.
"""
@views function solveStokes!(P, Vx_pad, Vy_pad, ρ_vy, μ_b, μ_p, τxx, τyy, τxy, ∇V, dτPt, Rx, Ry, dVxdτ, dVydτ, dτVx, dτVy, Vx_s, Vy_s,
    g_y, dx, dy, Nx, Ny, dt, maxdisp, comm; use_free_surface_stabilization::Bool=true, ϵ=1e-8, print_info::Bool=true)

    Vdmp = 4.0
    Vsc = 0.45                  # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc = 1.0 / 2.0
    min_dxy2 = min(dx, dy)^2
    max_nxy = max(nx_g(), ny_g())
    dampX = 1.0 - Vdmp / nx_g() # damping term for the x-momentum equation
    dampY = 1.0 - Vdmp / ny_g() # damping term for the y-momentum equation

    # numerical helper values
    _dx = 1.0 / dx
    _dy = 1.0 / dy

    if !use_free_surface_stabilization
        dt = 0.0
    end

    # PT setup and iterations
    @parallel compute_timesteps!(dτVx, dτVy, dτPt, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    ncheck = min(5 * min(nx_g(), ny_g()), 2000)
    ndtupdate = 10 * min(nx_g(), ny_g())
    t1 = 0
    itert1 = 11
    #err_evo1=[]; err_evo2=[]
    err = 2ϵ
    iter = 1
    niter = 0
    iterMax = 100000
    while err > ϵ && iter <= iterMax
        if (iter == itert1)
            t1 = Base.time()
        end

        @parallel compute_P_τ!(∇V, P, Vx_s, Vy_s, dτPt, τxx, τyy, τxy, μ_p, μ_b, _dx, _dy)
        @parallel compute_dV!(dVxdτ, dVydτ, P, ρ_vy, τxx, τyy, τxy, Vx_s, Vy_s, g_y, dampX, dampY, _dx, _dy, dt)
        @hide_communication (8, 8) begin
            @parallel compute_V!(Vx_s, Vy_s, dVxdτ, dVydτ, dτVx, dτVy)
            # Free slip BC
            @parallel (1:size(Vx_s, 1)) bc_y_noflux!(Vx_s)
            @parallel (1:size(Vx_s, 2)) bc_x_zero!(Vx_s)
            @parallel (1:size(Vy_s, 1)) bc_y_zero!(Vy_s)
            @parallel (1:size(Vy_s, 2)) bc_x_noflux!(Vy_s)
            update_halo!(Vx_s, Vy_s)
        end

        if use_free_surface_stabilization && (iter % ndtupdate == 0)
            dt = compute_dt(Vx_s, Vy_s, maxdisp, dx, dy, comm)
        end

        if iter % ncheck == 0
            dt_check = if use_free_surface_stabilization
                compute_dt(Vx_s, Vy_s, maxdisp, dx, dy, comm)
            else
                0.0
            end
            @parallel compute_Residuals!(Rx, Ry, P, ρ_vy, τxx, τyy, τxy, Vx_s, Vy_s, g_y, _dx, _dy, dt_check)
            err = compute_err(Rx, Ry, ∇V, comm)
            #push!(err_evo1, err); push!(err_evo2,iter)
            #@printf("Total steps = %d, err = %1.3e [mean_Rx=%1.3e, mean_Ry=%1.3e, mean_∇V=%1.3e] \n", iter, err, mean_Rx, mean_Ry, mean_∇V)
        end

        iter += 1
        niter += 1
    end
    t2 = Base.time()
    t_it = (t2 - t1) / (niter - itert1 + 1)
    A_eff = (3 * 2) / 1e9 * Nx * Ny * sizeof(Data.Number)
    T_eff = A_eff / t_it
    if print_info
        @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", niter, err, t2 - t1, round(T_eff, sigdigits=2))
    end

    # Since update_halo!() from ImplicitGlobalGrid can only update the first/last index in any dimension,
    # there must be kept a separate array to apply the final "zero second derivative" BC on the physical domain boundaries.
    # This artificial BC is necessary for the velocity interpolation to markers.
    Vx_pad[2:end-1, :] .= Vx_s
    Vy_pad[:, 2:end-1] .= Vy_s
    @parallel (1:size(Vx_pad, 2)) bc_x_mirror!(Vx_pad)
    @parallel (1:size(Vy_pad, 1)) bc_y_mirror!(Vy_pad)
    update_halo!(Vx_pad, Vy_pad)

    return compute_dt(Vx_s, Vy_s, maxdisp, dx, dy, comm), T_eff
end

"""
    compute_err(Rx, Ry, ∇V, comm)

Computes the maximum error across all ranks
"""
@views function compute_err(Rx, Ry, ∇V, comm)
    comm_size = MPI.Comm_size(comm)
    mean_Rx = MPI.Allreduce(Statistics.mean(abs.(Rx)), MPI.SUM, comm) / comm_size
    mean_Ry = MPI.Allreduce(Statistics.mean(abs.(Ry)), MPI.SUM, comm) / comm_size
    mean_∇V = MPI.Allreduce(Statistics.mean(abs.(∇V)), MPI.SUM, comm) / comm_size
    return max(mean_Rx, mean_Ry, mean_∇V)
end

"""
    compute_dt(Vx, Vy, maxdisp, dx, dy, comm)

Computes a valid timestep from all ranks

This function is very slow and could be improved significantly, but is not often called
"""
@views function compute_dt(Vx, Vy, maxdisp, dx, dy, comm)
    min_l = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)))
    return maxdisp * MPI.Allreduce(min_l, MPI.MIN, comm)
end

"""
    compute_timesteps!(dτVx, dτVy, dτP, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)

Computes timesteps for the pseudo-transcient solver
"""
@parallel function compute_timesteps!(dτVx, dτVy, dτP, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    @all(dτVx) = Vsc * min_dxy2 / @av_xa(μ_p) / 4.1
    @all(dτVy) = Vsc * min_dxy2 / @av_ya(μ_p) / 4.1
    @all(dτP) = Ptsc * 4.1 * @all(μ_p) / max_nxy
    return
end

"""
    compute_P_τ!(∇V, P, Vx, Vy, dτP, τxx, τyy, τxy, μ_p, μ_b, _dx, _dy)

Computes `P` and `τ` quantities
"""
@parallel_indices (ix, iy) function compute_P_τ!(∇V, P, Vx, Vy, dτP, τxx, τyy, τxy, μ_p, μ_b, _dx, _dy)
    xmax, ymax = size(τxy)
    if ix <= xmax && iy <= ymax
        # read V arrays
        vx0 = Vx[ix, iy]
        vx1 = Vx[ix, iy+1]
        vy0 = Vy[ix, iy]
        vy1 = Vy[ix+1, iy]
        # update τxx, τxx, P, ∇V
        if ix < xmax && iy < ymax # size of P-nodes is 1 smaller than basic nodes (τxy) in each dimension
            # read additional V
            vx2 = Vx[ix+1, iy+1]
            vy2 = Vy[ix+1, iy+1]
            # update
            dVx_dx = (vx2 - vx1) * _dx
            dVy_dy = (vy2 - vy1) * _dy
            div_V = dVx_dx + dVy_dy
            ∇V[ix, iy] = div_V
            P[ix, iy] -= dτP[ix, iy] * div_V
            μ = μ_p[ix, iy]
            τxx[ix, iy] = 2.0 * μ * (dVx_dx - 1.0 / 3.0 * div_V)
            τyy[ix, iy] = 2.0 * μ * (dVy_dy - 1.0 / 3.0 * div_V)
        end
        # update τxy
        dVx_dy = (vx1 - vx0) * _dy
        dVy_dx = (vy1 - vy0) * _dx
        τxy[ix, iy] = 2.0 * μ_b[ix, iy] * 0.5 * (dVx_dy + dVy_dx)
    end
    return nothing
end

"""
    compute_ResX(ix, iy, τxx, τxy, P, _dx, _dy)

Computes intermediate `dV` result in x
"""
@inline function compute_ResX(ix, iy, τxx, τxy, P, _dx, _dy)
    return ((τxx[ix+1, iy] - τxx[ix, iy]) * _dx
            +
            (τxy[ix+1, iy+1] - τxy[ix+1, iy]) * _dy
            -
            (P[ix+1, iy] - P[ix, iy]) * _dx)
end

"""
    compute_ResY(ix, iy, τyy, τxy, P, ρ_vy, Vx, Vy, g_y, _dx, _dy, dt)

Computes intermediate `dV` result in y
"""
@inline function compute_ResY(ix, iy, τyy, τxy, P, ρ_vy, Vx, Vy, g_y, _dx, _dy, dt)
    av_inn_y_Vx = 0.25 * (Vx[ix, iy+1] + Vx[ix+1, iy+1] + Vx[ix, iy+2] + Vx[ix+1, iy+2])
    d_xi_2_ρ_vy = ρ_vy[ix+2, iy+1] - ρ_vy[ix, iy+1]
    d_yi_2_ρ_vy = ρ_vy[ix+1, iy+2] - ρ_vy[ix+1, iy]
    return ((τyy[ix, iy+1] - τyy[ix, iy]) * _dy
            +
            (τxy[ix+1, iy+1] - τxy[ix, iy+1]) * _dx
            -
            (P[ix, iy+1] - P[ix, iy]) * _dy
            +
            g_y * (ρ_vy[ix+1, iy+1] - dt * (av_inn_y_Vx * d_xi_2_ρ_vy * 0.5 * _dx
                                            +
                                            Vy[ix+1, iy+1] * d_yi_2_ρ_vy * 0.5 * _dy)))
end

"""
    compute_dV!(dVxdτ, dVydτ, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, dampX, dampY, _dx, _dy, dt)

Computes `dVxdτ` and `dVydτ`
"""
@parallel_indices (ix, iy) function compute_dV!(dVxdτ, dVydτ, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, dampX, dampY, _dx, _dy, dt)
    if ix <= size(dVxdτ, 1) && iy <= size(dVxdτ, 2)
        Rx_temp = compute_ResX(ix, iy, τxx, τxy, P, _dx, _dy)
        dVxdτ[ix, iy] = dampX * dVxdτ[ix, iy] + Rx_temp
    end
    if ix <= size(dVydτ, 1) && iy <= size(dVydτ, 2)
        Ry_temp = compute_ResY(ix, iy, τyy, τxy, P, ρ_vy, Vx, Vy, g_y, _dx, _dy, dt)
        dVydτ[ix, iy] = dampY * dVydτ[ix, iy] + Ry_temp
    end
    return nothing
end

"""
    compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)

Computes `Vx` and `Vy`
"""
@parallel_indices (ix, iy) function compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
    if ix <= size(dτVx, 1) && iy <= size(dτVx, 2)
        Vx[ix+1, iy+1] += dτVx[ix, iy] * dVxdτ[ix, iy]
    end
    if ix <= size(dτVy, 1) && iy <= size(dτVy, 2)
        Vy[ix+1, iy+1] += dτVy[ix, iy] * dVydτ[ix, iy]
    end
    return
end

"""
    compute_Residuals!(Rx, Ry, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, _dx, _dy, dt)

Computes residuals `Rx` and `Ry`
"""
@parallel_indices (ix, iy) function compute_Residuals!(Rx, Ry, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, _dx, _dy, dt)
    if ix <= size(Rx, 1) && iy <= size(Rx, 2)
        Rx[ix, iy] = compute_ResX(ix, iy, τxx, τxy, P, _dx, _dy)
    end
    if ix <= size(Ry, 1) && iy <= size(Ry, 2)
        Ry[ix, iy] = compute_ResY(ix, iy, τyy, τxy, P, ρ_vy, Vx, Vy, g_y, _dx, _dy, dt)
    end
    return
end

"""
    bc_x_zero!(A::Data.Array)

Implements zero side boundaries
"""
@parallel_indices (iy) function bc_x_zero!(A::Data.Array)
    A[1, iy] = 0.0
    A[end, iy] = 0.0
    return
end

"""
    bc_x_noflux!(A::Data.Array)

Implements no flux side boundaries
"""
@parallel_indices (iy) function bc_x_noflux!(A::Data.Array)
    A[1, iy] = A[2, iy]
    A[end, iy] = A[end-1, iy]
    return
end

"""
    bc_x_mirror!(A::Data.Array)

Implements mirror side boundaries
"""
@parallel_indices (iy) function bc_x_mirror!(A::Data.Array)
    A[1, iy] = -A[3, iy]
    A[end, iy] = -A[end-2, iy]
    return
end

"""
    bc_y_zero!(A::Data.Array)

Implements zero horizontal boundaries
"""
@parallel_indices (ix) function bc_y_zero!(A::Data.Array)
    A[ix, 1] = 0.0
    A[ix, end] = 0.0
    return
end
"""
    bc_y_noflux!(A::Data.Array)

Implements no flux horizontal boundaries
"""
@parallel_indices (ix) function bc_y_noflux!(A::Data.Array)
    A[ix, 1] = A[ix, 2]
    A[ix, end] = A[ix, end-1]
    return
end
"""
    bc_y_mirror!(A::Data.Array)

Implements mirror horizontal boundaries
"""
@parallel_indices (ix) function bc_y_mirror!(A::Data.Array)
    A[ix, 1] = -A[ix, 3]
    A[ix, end] = -A[ix, end-2]
    return
end
