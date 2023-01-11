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
