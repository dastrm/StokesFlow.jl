@assert (@isdefined(USE_GPU)) "USE_GPU must be defined"
typeassert(USE_GPU, Bool)

# ParallelStencil must already be initalized !!!
using ParallelStencil
import StaticArrays

include("common.jl")

"""
    moveMarkersRK4!(x_m, y_m, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dt, lx, ly, dx, dy)

Moves markers according to a fourth order Runge-Kutta method
"""
@views function moveMarkersRK4!(x_m, y_m, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dt, lx, ly, dx, dy)

    Nm = length(x_m)
    @assert (Nm == length(y_m)) "x and y Marker coords have not same length"

    @parallel (1:Nm) moveMarkersRK4_Kernel!(x_m, y_m, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dt, lx, ly, dx, dy)

    return nothing
end

"""
    moveMarkersRK4_Kernel!(x_m, y_m, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dt, lx, ly, dx, dy)

Implements a parallel fourth order Runge-Kutta method
"""
@parallel_indices (m) function moveMarkersRK4_Kernel!(x_m, y_m, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dt, lx, ly, dx, dy)

    # Runge-Kutta 4th order
    rk4_dt = StaticArrays.@SVector [0.0, 0.5dt, 0.5dt, dt]
    rk4_wt = StaticArrays.@SVector [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]

    x_old = x_m[m] # old position
    y_old = y_m[m]
    vx_eff, vy_eff = 0.0, 0.0 # 'effective' velocity for explicit update: x_new = x_old + v_eff*dt
    vx_rk, vy_rk = 0.0, 0.0 # velocity at previous/current point

    for it = 1:4 # loop over points A-D
        # position of current point based on previous point velocities
        x_rk = x_old + rk4_dt[it] * vx_rk
        y_rk = y_old + rk4_dt[it] * vy_rk

        # interpolate velocity to current point
        vx_rk, vy_rk = interpolateV(x_rk, y_rk, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dx, dy)

        # apply RK4 scheme: add up weighted velocities
        vx_eff += rk4_wt[it] * vx_rk
        vy_eff += rk4_wt[it] * vy_rk
    end

    # move particle
    x_new = x_old + vx_eff * dt
    y_new = y_old + vy_eff * dt

    # explicitly restrict particles to stay on domain
    # (optional, does not really change anything if BC correctly implemented and dt small enough)
    # !! TODO: CHANGE, if global domain is not 0-lx and 0-ly !!
    x_new = min(max(x_new, 0), lx)
    y_new = min(max(y_new, 0), ly)

    # write back updated positions
    x_m[m] = x_new
    y_m[m] = y_new

    return nothing
end

"""
    interpolateV(x, y, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dx, dy)

Interpolates grid velocities to 2D positions
"""
function interpolateV(x, y, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dx, dy)
    # Interpolate Vx
    ix, iy, dxij, dyij = topleftIndexRelDist(x_vx_min, y_vx_min, x, y, dx, dy)
    #index range failsafe (that should never be used)
    if ix < 2
        @printline("WARNING: Vx-interpolation, ix=$(ix) too small")
        ix = 2
        dxij = 0.0
    end
    if iy < 1
        @printline("WARNING: Vx-interpolation, iy=$(iy) too small")
        iy = 1
        dyij = 0.0
    end
    if ix >= size(Vx, 1) - 1
        @printline("WARNING: Vx-interpolation, ix=$(ix) too big")
        ix = size(Vx, 1) - 2
        dxij = 1.0
    end
    if iy >= size(Vx, 2)
        @printline("WARNING: Vx-interpolation, iy=$(iy) too big")
        iy = size(Vx, 2) - 1
        dyij = 1.0
    end
    # bilinear Interpolation
    v1 = Vx[ix, iy]
    v2 = Vx[ix+1, iy]
    v3 = Vx[ix, iy+1]
    v4 = Vx[ix+1, iy+1]
    vx = bilinearInterp(v1, v2, v3, v4, dxij, dyij)
    # Continuity-based velocity correction for Vx: the interpolated field will have zero divergence!
    correction = 0.0
    # right half of cell => extend stencil to the right
    if dxij >= 0.5
        v5 = Vx[ix+2, iy]
        v6 = Vx[ix+2, iy+1]
        correction = 0.5 * (dxij - 0.5)^2 * (
                         (1 - dyij) * (v1 - 2v2 + v5) +
                         dyij * (v3 - 2v4 + v6))
        # left  half of cell => extend stencil to the left
    else
        v5 = Vx[ix-1, iy]
        v6 = Vx[ix-1, iy+1]
        correction = 0.5 * (dxij - 0.5)^2 * (
                         (1 - dyij) * (v5 - 2v1 + v2) +
                         dyij * (v6 - 2v3 + v4))
    end
    vx += correction

    # Interpolate Vy
    ix, iy, dxij, dyij = topleftIndexRelDist(x_vy_min, y_vy_min, x, y, dx, dy)
    #index range failsafe (that should never be used)
    if ix < 1
        @printline("WARNING: Vy-interpolation, ix=$(ix) too small")
        ix = 1
        dxij = 0.0
    end
    if iy < 2
        @printline("WARNING: Vy-interpolation, iy=$(iy) too small")
        iy = 2
        dyij = 0.0
    end
    if ix >= size(Vy, 1)
        @printline("WARNING: Vy-interpolation, ix=$(ix) too big")
        ix = size(Vy, 1) - 1
        dxij = 1.0
    end
    if iy >= size(Vy, 2) - 1
        @printline("WARNING: Vy-interpolation, iy=$(iy) too big")
        iy = size(Vy, 2) - 2
        dyij = 1.0
    end
    # bilinear Interpolation
    v1 = Vy[ix, iy]
    v2 = Vy[ix+1, iy]
    v3 = Vy[ix, iy+1]
    v4 = Vy[ix+1, iy+1]
    vy = bilinearInterp(v1, v2, v3, v4, dxij, dyij)
    # Continuity-based velocity correction for Vy: the interpolated field will have zero divergence!
    correction = 0.0
    # lower half of cell => extend stencil to the bottom
    if dyij >= 0.5
        v5 = Vy[ix, iy+2]
        v6 = Vy[ix+1, iy+2]
        correction = 0.5 * (dyij - 0.5)^2 * (
                         (1 - dxij) * (v1 - 2v3 + v5) +
                         dxij * (v2 - 2v4 + v6))
        # upper half of cell => extend stencil to the top
    else
        v5 = Vy[ix, iy-1]
        v6 = Vy[ix+1, iy-1]
        correction = 0.5 * (dyij - 0.5)^2 * (
                         (1 - dxij) * (v5 - 2v1 + v3) +
                         dxij * (v6 - 2v2 + v4))
    end
    vy += correction

    return vx, vy
end

"""
    bilinearInterp(v1, v2, v3, v4, dxij, dyij)

Performs bilinear interpolation using repeated linear interpolation
"""
function bilinearInterp(v1, v2, v3, v4, dxij, dyij)
    s1 = (1 - dxij) * (1 - dyij) * v1
    s2 = dxij * (1 - dyij) * v2
    s3 = (1 - dxij) * dyij * v3
    s4 = dxij * dyij * v4
    return s1 + s2 + s3 + s4
end
