@assert (@isdefined(USE_GPU)) "USE_GPU must be defined"
typeassert(USE_GPU, Bool)

using ParallelStencil

# Workaround for ParallelStencil Bug: @ps_println only works on GPU
@static if USE_GPU
    macro printline(args...)
        esc(:(@ps_println($(args...))))
    end
else
    macro printline(args...)
        esc(:(println($(args...))))
    end
end

"""
    topleftIndexRelDist(x_grid_min, y_grid_min, x, y, dx, dy)

Compute indices `ix`, `iy` of the top left node with respect to the given 2D position as well as the relative distances
"""
@views function topleftIndexRelDist(x_grid_min, y_grid_min, x, y, dx, dy)
    # indices: may be out of bounds if the grid does not cover (x,y)
    ix = floor(Int, (x - x_grid_min) / dx) + 1
    iy = floor(Int, (y - y_grid_min) / dy) + 1
    # position of top left node with index (ix,iy)
    x_ix = x_grid_min + (ix - 1) * dx
    y_iy = y_grid_min + (iy - 1) * dy
    # compute relative distances
    dxij = (x - x_ix) / dx
    dyij = (y - y_iy) / dy
    return ix, iy, dxij, dyij
end

"""
    initializeMarkersCPU(comm, dims, coords, marker_density::Integer, lx, ly, dx, dy, Nx, Ny, RAND_MARKER_POS; rng=Random.GLOBAL_RNG)

Declares marker arrays and initializes marker coordinates
"""
@views function initializeMarkersCPU(comm, dims, coords, marker_density::Integer, lx, ly, dx, dy, Nx, Ny, RAND_MARKER_POS; rng=Random.GLOBAL_RNG)

    dxm = dx / marker_density
    dym = dy / marker_density

    xloBNDRY = coords[1] == 0
    xhiBNDRY = coords[1] == dims[1] - 1
    yloBNDRY = coords[2] == 0
    yhiBNDRY = coords[2] == dims[2] - 1

    x_less_hi = marker_density % 2 == 0 ? dxm / 2 : dxm
    y_less_hi = marker_density % 2 == 0 ? dym / 2 : dym

    x_less_lo = marker_density % 2 == 0 ? dxm / 2 : 0.0
    y_less_lo = marker_density % 2 == 0 ? dym / 2 : 0.0

    xlimlo = xloBNDRY ? dxm / 2 : dx / 2 + x_less_lo
    xlimhi = xhiBNDRY ? lx - dxm / 2 : lx - dx / 2 - x_less_hi
    ylimlo = yloBNDRY ? dym / 2 : dy / 2 + y_less_lo
    ylimhi = yhiBNDRY ? ly - dym / 2 : ly - dy / 2 - y_less_hi

    lo_less = Int(floor(marker_density / 2))
    hi_less = Int(ceil(marker_density / 2))

    Nmx = marker_density * (Nx - 1) - (xloBNDRY ? 0 : lo_less) - (xhiBNDRY ? 0 : hi_less)
    Nmy = marker_density * (Ny - 1) - (yloBNDRY ? 0 : lo_less) - (yhiBNDRY ? 0 : hi_less)
    Nm = Nmx * Nmy

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
        if marker_density % 2 == 1
            x_m, y_m, ρ_m, μ_m = exchangeMarkers(comm, dims, [lx, ly], dx, dy, x_m, y_m, ρ_m, μ_m)
        end
    end

    return x_m, y_m, ρ_m, μ_m
end
