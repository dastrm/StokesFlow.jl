# ParallelStencil must already be initalized !!!

using Printf, Statistics

# STOKES SOLVER
@views function solveStokes!(P,Vx,Vy,ρ_vy,μ_b,μ_p,
                            τxx, τyy, τxy, ∇V, dτPt, Rx, Ry, dVxdτ, dVydτ, dτVx, dτVy,
                            g_y, dx, dy, Nx, Ny,
                            dt, maxdisp; use_free_surface_stabilization::Bool=true,
                            ϵ=1e-8,
                            print_info::Bool=true)

    Vdmp      = 4.0
    Vsc       = 0.45        # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc      = 1.0/2.0 
    min_dxy2  = min(dx,dy)^2
    max_nxy   = max(Nx,Ny)
    dampX     = 1.0-Vdmp/Nx # damping term for the x-momentum equation
    dampY     = 1.0-Vdmp/Ny # damping term for the y-momentum equation

    # numerical helper values
    _dx = 1.0/dx
    _dy = 1.0/dy

    if !use_free_surface_stabilization
        dt=0.0
    end

    # PT setup and iterations
    @parallel compute_timesteps!(dτVx, dτVy, dτPt, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    ncheck    = min(5*min(Nx,Ny),2000)
    ndtupdate = 10*min(Nx,Ny)
    t1 = 0; itert1 = 11
    #ϵ = 0.01 # tol
    err_evo1=[]; err_evo2=[]
    err = 2ϵ; iter=1; niter=0; iterMax=100000
    while err > ϵ && iter <= iterMax
        if (iter==itert1) t1 = Base.time() end

        @parallel compute_P_τ!(∇V, P, Vx, Vy, dτPt, τxx, τyy, τxy, μ_p, μ_b, _dx, _dy)
        @parallel compute_dV!(dVxdτ, dVydτ, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, dampX, dampY, _dx, _dy, dt)
        @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)

        # Free slip BC
        @parallel (1:size(Vx,1)) bc_y_noflux!(Vx)
        @parallel (1:size(Vx,2)) bc_x_zero!(Vx)
        @parallel (1:size(Vy,1)) bc_y_zero!(Vy)
        @parallel (1:size(Vy,2)) bc_x_noflux!(Vy)

        if use_free_surface_stabilization && (iter%ndtupdate == 0)
            dt = compute_dt(Vx,Vy,maxdisp,dx,dy)
        end

        if iter%ncheck == 0
            dt_check = if use_free_surface_stabilization compute_dt(Vx,Vy,maxdisp,dx,dy) else 0.0 end
            @parallel compute_Residuals!(Rx, Ry, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, _dx, _dy, dt_check)
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
    if print_info
        @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", niter, err, t2-t1, round(T_eff, sigdigits=2))
    end

    return compute_dt(Vx,Vy,maxdisp,dx,dy), T_eff
end


# this function is very slow and could be improved significantly, but is not called often
@views function compute_dt(Vx,Vy,maxdisp,dx,dy)
    return maxdisp*min(dx/maximum(Vx[2:end-1,:]),dy/maximum(Vy[:,2:end-1]))
end

@parallel function compute_timesteps!(dτVx, dτVy, dτP, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    @all(dτVx) = Vsc*min_dxy2/@av_xa(μ_p)/4.1
    @all(dτVy) = Vsc*min_dxy2/@av_ya(μ_p)/4.1
    @all(dτP)  = Ptsc*4.1*@all(μ_p)/max_nxy
    return
end

@parallel_indices (ix,iy) function compute_P_τ!(∇V, P, Vx, Vy, dτP, τxx, τyy, τxy, μ_p, μ_b, _dx, _dy)
    xmax, ymax = size(τxy)
    if ix <= xmax && iy <= ymax
        # read V arrays
        vx0 = Vx[ix+1,iy  ]
        vx1 = Vx[ix+1,iy+1]
        vy0 = Vy[ix  ,iy+1]
        vy1 = Vy[ix+1,iy+1]
        # update τxx, τxx, P, ∇V
        if ix < xmax && iy < ymax # size of P-nodes is 1 smaller than basic nodes (τxy) in each dimension
            # read additional V
            vx2 = Vx[ix+2,iy+1]
            vy2 = Vy[ix+1,iy+2]
            # update
            dVx_dx = (vx2 - vx1)*_dx
            dVy_dy = (vy2 - vy1)*_dy
            div_V  = dVx_dx + dVy_dy
            ∇V[ix,iy] = div_V
            P[ix,iy] -= dτP[ix,iy]*div_V
            μ = μ_p[ix,iy]
            τxx[ix,iy] = 2.0*μ*(dVx_dx - 1.0/3.0*div_V)
            τyy[ix,iy] = 2.0*μ*(dVy_dy - 1.0/3.0*div_V)
        end
        # update τxy
        dVx_dy = (vx1 - vx0)*_dy
        dVy_dx = (vy1 - vy0)*_dx
        τxy[ix,iy] = 2.0*μ_b[ix,iy]*0.5*(dVx_dy + dVy_dx)
    end
    return
end

@inline function compute_ResX(ix,iy,τxx,τxy,P,_dx,_dy)
    return ((τxx[ix+1,iy  ] - τxx[ix  ,iy])*_dx
           +(τxy[ix+1,iy+1] - τxy[ix+1,iy])*_dy
           -(  P[ix+1,iy  ] -   P[ix  ,iy])*_dx)
end

@inline function compute_ResY(ix,iy,τyy,τxy,P,ρ_vy,Vx,Vy,g_y,_dx,_dy,dt)
    av_inn_y_Vx = 0.25*(Vx[ix+1,iy+1] + Vx[ix+2,iy+1] + Vx[ix+1,iy+2] + Vx[ix+2,iy+2])
    d_xi_2_ρ_vy = ρ_vy[ix+2,iy+1] - ρ_vy[ix,iy+1]
    d_yi_2_ρ_vy = ρ_vy[ix+1,iy+2] - ρ_vy[ix+1,iy]
    return ((τyy[ix  ,iy+1] - τyy[ix,iy  ])*_dy
           +(τxy[ix+1,iy+1] - τxy[ix,iy+1])*_dx
           -(  P[ix  ,iy+1] -   P[ix,iy  ])*_dy
           +g_y*(ρ_vy[ix+1,iy+1] - dt*(  av_inn_y_Vx   * d_xi_2_ρ_vy*0.5*_dx
                                       + Vy[ix+1,iy+2] * d_yi_2_ρ_vy*0.5*_dy)))
end

@parallel_indices (ix,iy) function compute_dV!(dVxdτ, dVydτ, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, dampX, dampY, _dx, _dy, dt)
    if ix <= size(dVxdτ,1) && iy <= size(dVxdτ,2)
        Rx_temp = compute_ResX(ix,iy,τxx,τxy,P,_dx,_dy)
        dVxdτ[ix,iy] = dampX*dVxdτ[ix,iy] + Rx_temp
    end
    if ix <= size(dVydτ,1) && iy <= size(dVydτ,2)
        Ry_temp = compute_ResY(ix,iy,τyy,τxy,P,ρ_vy,Vx,Vy,g_y,_dx,_dy,dt)
        dVydτ[ix,iy] = dampY*dVydτ[ix,iy] + Ry_temp
    end
    return
end

@parallel_indices (ix,iy) function compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
    if ix <= size(dτVx,1) && iy <= size(dτVx,2)
        Vx[ix+2,iy+1] += dτVx[ix,iy]*dVxdτ[ix,iy]
    end
    if ix <= size(dτVy,1) && iy <= size(dτVy,2)
        Vy[ix+1,iy+2] += dτVy[ix,iy]*dVydτ[ix,iy]
    end
    return
end

@parallel_indices (ix,iy) function compute_Residuals!(Rx, Ry, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, _dx, _dy, dt)
    if ix <= size(Rx,1) && iy <= size(Rx,2)
        Rx[ix,iy] = compute_ResX(ix,iy,τxx,τxy,P,_dx,_dy)
    end
    if ix <= size(Ry,1) && iy <= size(Ry,2)
        Ry[ix,iy] = compute_ResY(ix,iy,τyy,τxy,P,ρ_vy,Vx,Vy,g_y,_dx,_dy,dt)
    end
    return
end

# side boundaries
@parallel_indices (iy) function bc_x_zero!(A::Data.Array)
    A[1    , iy] = -A[3    , iy]
    A[2    , iy] = 0.0
    A[end  , iy] = -A[end-2, iy]
    A[end-1, iy] = 0.0
    return
end
@parallel_indices (iy) function bc_x_noflux!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

# horizontal boundaries
@parallel_indices (ix) function bc_y_zero!(A::Data.Array)
    A[ix, 1    ] = -A[ix, 3    ]
    A[ix, 2    ] = 0.0
    A[ix, end  ] = -A[ix, end-2]
    A[ix, end-1] = 0.0
    return
end
@parallel_indices (ix) function bc_y_noflux!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end
