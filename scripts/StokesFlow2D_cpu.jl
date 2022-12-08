using Plots


@views function StokesFlow2D()

    # --- PARAMETERS ---
    # time
    nt = 10                                     # number of timesteps
    maxdisp = 0.5                               # dt is determined s.t. no marker moves further than maxdisp cells
    # physical parameters
    g_y = 9.81                                  # earth gravity, m/s^2
    lx, ly = 100000, 100000                     # domain size
    μ_air, μ_matrix, μ_plume = 1e17, 1e19, 1e18 # Viscosity, Pa*s
    ρ_air, ρ_matrix, ρ_plume = 1   , 3300, 3200 # Density, kg/m^3
    plume_x, plume_y = lx/2, ly/2               # plume midpoint
    plume_r = 20000                             # plume radius
    air_height = 0.2*ly                         # height of the 'sticky air' layer on top
    # discretization parameters
    Nx, Ny = 35, 45                             # number of grid points
    RANDOMIZE_MARKER_POSITIONS = true           # add random perturbation to initial marker coords
    marker_density = 5                          # use this amount of markers per grid step per dimension

    # derived quantities
    dx, dy = lx/(Nx-1), ly/(Ny-1)               # grid resolution
    Nmx = (Nx-1)*marker_density                 # number of markers
    Nmy = (Ny-1)*marker_density
    Nm  = Nmx*Nmy

    # --- ARRAYS ---
    # marker array allocations
    xy_m       = zeros(Nm,2)                    # marker coords, TODO: maybe better 2 arrays? or dimensions (2,Nm)? or array of structs?
    ρ_m        = zeros(Nm)                      # marker property: density
    μ_m        = zeros(Nm)                      # marker property: viscosity
    vx_m, vy_m = zeros(Nm), zeros(Nm)           # marker velocities

    # grid array allocations
    # maybe Vx & Vy sizes need adjustments for marker interpolation (multi-GPU case, simplify single GPU). TODO
    P    = zeros(Nx-1,Ny-1)
    Vx   = zeros(Nx  ,Ny+1)
    Vy   = zeros(Nx+1,Ny  )
    ρ_vy = zeros(Nx+1,Ny  )
    μ_b  = zeros(Nx  ,Ny  ) # μ on basic nodes
    μ_p  = zeros(Nx-1,Ny-1) # μ on pressure nodes

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
    @assert size(ρ_vy) == size(Vy)

    # --- INITIAL CONDITIONS ---
    setInitialMarkerCoords!(xy_m, Nmx, Nmy, x, y, RANDOMIZE_MARKER_POSITIONS::Bool)
    setInitialMarkerProperties!(xy_m, ρ_m, μ_m, Nm, μ_air, μ_matrix, μ_plume, ρ_air, ρ_matrix, ρ_plume, plume_x, plume_y, plume_r, air_height)
    vline(x)
    hline!(y)
    display(scatter!(xy_m[:,1],xy_m[:,2],color=Int.(round.(ρ_m)),xlims=(x[1],x[end]),ylims=(y[1],y[end]),aspect_ratio=1,yflip=true,legend=false))

    # --- TIMESTEPPING ---
    for t=1:nt
        # interpolate material properties to grid
        bilinearMarkerToGrid!(x_vy,y_vy,ρ_vy,xy_m,ρ_m,dx,dy)
        bilinearMarkerToGrid!(x   ,y   ,μ_b ,xy_m,μ_m,dx,dy)
        bilinearMarkerToGrid!(x_p ,y_p ,μ_p ,xy_m,μ_m,dx,dy)

        # calculate velocities on grid
        solveStokes!(Vx,Vy)

        # plot current state
        showPlot(x,y,x_p,y_p,x_vx,y_vx,x_vy,y_vy, P,Vx,Vy,ρ_vy,μ_b,μ_p, lx,ly)

        # move markers
        dt = maxdisp*min(dx/maximum(Vx),dy/maximum(Vy))
        moveMarkersRK4!(xy_m,Nm,dt,lx,ly)
    end


    return nothing
end


@views function solveStokes!(Vx,Vy)
    # TODO
    Vx .= 1
    Vy .= 1
    return nothing
end


@views function velocitiesGridToMarker!()
    # TODO
    return nothing
end


@views function moveMarkersRK4!(xy_m,Nm,dt,lx,ly)
    ## interpolate Vx, Vy => effective marker velocities vxm, vym according to RK4
    vxm,vym = zeros(Nm),zeros(Nm)
    # Runge-Kutta 4th order
    rk4_dt = [0.0, 0.5, 0.5, 1.0] * dt;
    rk4_wt = [1, 2, 2, 1];
    rk4_wt = rk4_wt/sum(rk4_wt);

    vx_rk = zeros(Nm);
    vy_rk = zeros(Nm);
    x_rk = zeros(Nm);
    y_rk = zeros(Nm);
    for it=eachindex(rk4_wt) # loop over points A-D
        # compute coordinates of current point
        x_rk .= xy_m[:,1] + rk4_dt[it]*vx_rk;
        y_rk .= xy_m[:,2] + rk4_dt[it]*vy_rk;
        # compute velocities at current point => get vx_rk, vy_rk
        velocitiesGridToMarker!() # TODO vx_rk
        velocitiesGridToMarker!() # TODO vy_rk
        # add weighted current velocities to 'effective' velocity
        vxm = vxm + rk4_wt[it]*vx_rk;
        vym = vym + rk4_wt[it]*vy_rk;
    end

    ## actually move particles
    xy_m[:,1] .+= vxm*dt;
    xy_m[:,1] .+= vym*dt;

    # explicitly restrict particles to stay on domain
    # (optional, does not really change anything if BC correctly implemented and dt small enough)
    # !! CHANGE, if global domain is not 0-lx and 0-ly !!
    for m=1:Nm
        xy_m[m,:] .= min.(max.(xy_m[m,:],0),[lx,ly]);
    end

    return nothing
end


function showPlot(x,y,x_p,y_p,x_vx,y_vx,x_vy,y_vy, P,Vx,Vy,ρ_vy,μ_b,μ_p, lx,ly)
    # Visualization
    #p1 = heatmap(x_p ,  y_p, Array(P)' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Pressure")
    #p2 = heatmap(x_vy, y_vy, Array(Vy)', yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Vy")
    #p4 = heatmap(X[2:end-1], Yv[2:end-1], log10.(abs.(Array(Ry)')), aspect_ratio=1, xlims=(X[2],X[end-1]), ylims=(Yv[2],Yv[end-1]), c=:inferno, title="log10(Ry)")
    #p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
    #display(plot(p1, p2, p4, p5))

    p6 = heatmap(x_vy ,  y_vy, Array(ρ_vy)' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="ρ_vy")
    p7 = heatmap(x    ,  y   , Array(μ_b )' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="μ_b" )
    p8 = heatmap(x_p  ,  y_p , Array(μ_p )' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="μ_p" )

    display(plot(p6,p7,p8))
    return nothing
end


@views function bilinearMarkerToGrid!(x_grid,y_grid,val_grid,xy_m,val_m,dx,dy)
    Nx,Ny = size(val_grid)
    Nm = size(xy_m,1)
    val_wt_sum = zeros(Nx,Ny)
    wt_sum = zeros(Nx,Ny)

    for m=1:1:Nm

        xm,ym = xy_m[m,:]

        # get indices of top left node w.r.t marker m.
        # may be 0, when the marker is further left or up than the first grid node
        ix = floor(Int,(xm-x_grid[1])/dx) + 1
        iy = floor(Int,(ym-y_grid[1])/dy) + 1
        @assert ix>=0 && ix<=Nx && iy>=0 && iy<=Ny

        # compute relative distances to the top left node
        dxmij = if ix == 0 (xm-x_grid[1]+dx)/dx else (xm-x_grid[ix])/dx end
        dymij = if iy == 0 (ym-y_grid[1]+dy)/dy else (ym-y_grid[iy])/dy end
        @assert dxmij>=0 && dxmij<=1 && dymij>=0 && dymij<=1

        # sum up weights, if the respective node exists
        if iy>0
            if ix>0
                # 1) top left
                w = (1-dxmij)*(1-dymij)
                wt_sum[ix,iy] = wt_sum[ix,iy] + w
                val_wt_sum[ix,iy] = val_wt_sum[ix,iy] + w*val_m[m]
            end
            if ix<Nx
                # 2) top right
                w = dxmij*(1-dymij)
                wt_sum[ix+1,iy] = wt_sum[ix+1,iy] + w
                val_wt_sum[ix+1,iy] = val_wt_sum[ix+1,iy] + w*val_m[m]
            end
        end
        if iy<Ny
            if ix>0
                # 3) bottom left
                w = (1-dxmij)*dymij
                wt_sum[ix,iy+1] = wt_sum[ix,iy+1] + w
                val_wt_sum[ix,iy+1] = val_wt_sum[ix,iy+1] + w*val_m[m]
            end
            if ix<Nx
                # 4) bottom right
                w = dxmij*dymij
                wt_sum[ix+1,iy+1] = wt_sum[ix+1,iy+1] + w
                val_wt_sum[ix+1,iy+1] = val_wt_sum[ix+1,iy+1] + w*val_m[m]
            end
        end
    end

    # finally compute actual value from the sums
    for ix = 1:Nx
        for iy = 1:Ny
            if wt_sum[ix,iy] > 0.0
                val_grid[ix,iy] = val_wt_sum[ix,iy] / wt_sum[ix,iy];
            else
                # This happens if no markers are close enough, which should
                # be avoided. do not update in this case
                print("PROBLEM: no markers close enough for value interpolation:")
                @show ix, iy, Nx, Ny
            end
        end
    end
    return nothing
end


"""
Sets initial coordinates and properties of the markers
x, y are coordinates of the basic grid nodes
xlims, ylims contain domain lower and upper limits at indices [1] and [end], respectively
"""
@views function setInitialMarkerCoords!(xy_m, Nmx, Nmy, xlims, ylims, RANDOMIZE_MARKER_POSITIONS::Bool)
    Nm = Nmx*Nmy
    @assert size(xy_m) == (Nm,2)
    dxm = (xlims[end]-xlims[1]) / Nmx
    dym = (ylims[end]-ylims[1]) / Nmy
    xcoords = LinRange(xlims[1]+0.5dxm,xlims[end]-0.5dxm,Nmx)
    ycoords = LinRange(ylims[1]+0.5dym,ylims[end]-0.5dym,Nmy)
    m = 1
    for ix=1:Nmx
        for iy=1:Nmy
            xy_m[m,1] = xcoords[ix]
            xy_m[m,2] = ycoords[iy]
            m += 1
        end
    end
    if RANDOMIZE_MARKER_POSITIONS
        # TODO: maybe add seed and specify rng
        xy_m[:,1] .+= (rand(Nm).-0.5).*dxm
        xy_m[:,2] .+= (rand(Nm).-0.5).*dym
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
    @assert size(xy_m) == (Nm,2)
    @assert size(ρ_m) == size(μ_m) && size(μ_m,1) == Nm
    for m=1:Nm
        x,y = xy_m[m,:]
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
