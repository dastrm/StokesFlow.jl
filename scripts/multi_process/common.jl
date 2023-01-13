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
function topleftIndexRelDist(x_grid_min, y_grid_min, x, y, dx, dy)
    # TODO: ix,iy may be out of bounds if the grid does not cover (x,y), this is not checked here
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
    setInitialMarkerCoords!(coords, dims, dx, dy, x_m, y_m, Nmx, Nmy, xlims, ylims, RAND_MARKER_POS::Bool; rng=Random.GLOBAL_RNG)

Sets initial coordinates and properties of the markers

`x` and `y` are coordinates of the basic grid nodes

`xlims` and `ylims` contain domain lower and upper limits at start and end indices respectively
"""
@views function setInitialMarkerCoords!(coords, dims, dx, dy, x_m, y_m, Nmx, Nmy, xlims, ylims, RAND_MARKER_POS::Bool; rng=Random.GLOBAL_RNG)
    Nm = Nmx * Nmy
    @assert size(x_m, 1) == (Nm)
    @assert size(y_m, 1) == (Nm)
    xlimslower = coords[1] == 0 ? xlims[1] : xlims[1] + dx / 2
    xlimsupper = coords[1] == dims[1] - 1 ? xlims[end] : xlims[end] - dx / 2
    ylimslower = coords[2] == 0 ? ylims[1] : ylims[1] + dy / 2
    ylimsupper = coords[2] == dims[2] - 1 ? ylims[end] : ylims[end] - dy / 2
    dxm = (xlimsupper - xlimslower) / Nmx
    dym = (ylimsupper - ylimslower) / Nmy
    xcoords = LinRange(xlimslower + 0.5dxm, xlimsupper - 0.5dxm, Nmx)
    ycoords = LinRange(ylimslower + 0.5dym, ylimsupper - 0.5dym, Nmy)
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
    end
    return nothing
end
