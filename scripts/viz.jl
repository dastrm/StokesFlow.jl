using Plots, Plots.Measures, Printf, MAT

const animfps = 2 # animation fps
default(framestyle=:box, label=false, grid=false, yflip=true, aspect_ratio=1)

"""
  loadStats()

Loads visualization relevant stats previously saved to disk
"""
@views function loadStats()
  file = matopen("viz_out/stats.mat")
  dims = read(file, "dims")
  nt = read(file, "nt")
  close(file)
  return dims, nt
end

"""
  vizMarkers(dims, nt)

Creates an animation from the markers previously saved to disk
"""
@views function vizMarkers(dims, nt)
  nprocs = dims[1] * dims[2]
  anim = @animate for it = 0:nt
    x_m = []
    y_m = []
    ρ_m = []

    for ip = 1:nprocs
      file = matopen(string(@sprintf("viz_out/markers_%04d_%04d", it, ip - 1), ".mat"))
      x_m = append!(x_m, read(file, "x_m"))
      y_m = append!(y_m, read(file, "y_m"))
      ρ_m = append!(ρ_m, read(file, "rho_m"))
      close(file)
    end
    @assert size(x_m, 1) == size(y_m, 1)
    @assert size(x_m, 1) == size(ρ_m, 1)

    opts = (color=Int.(round.(exp.(ρ_m))), size=(1200, 1150), margin=1mm, legend=false, markersize=3, markerstrokewidth=0)
    @assert !(any(isnan, x_m) || any(isnan, y_m) || any(isnan, ρ_m))
    scatter(x_m, y_m; opts...)
  end

  gif(anim, "viz_out/markers.gif"; fps=animfps)

  return nothing
end

"""
  vizGrid(dims, nt)

Creates an animation from the grid values previously saved to disk
"""
@views function vizGrid(dims, nt)
  anim = @animate for it = 1:nt
    x = []
    y = []
    μ_b = []
    x_p = []
    y_p = []
    P = []
    x_vx = []
    y_vx = []
    Vx = []
    x_vy = []
    y_vy = []
    Vy = []

    ip = 1
    for ipx = 1:dims[1]
      for ipy = 1:dims[2]
        file = matopen(string(@sprintf("viz_out/grid_%04d_%04d", it, ip - 1), ".mat"))
        x_loc = read(file, "x")
        y_loc = read(file, "y")
        μ_b_loc = read(file, "mu_b")
        x_p_loc = read(file, "x_p")
        y_p_loc = read(file, "y_p")
        P_loc = read(file, "P")
        x_vx_loc = read(file, "x_vx")
        y_vx_loc = read(file, "y_vx")
        Vx_loc = read(file, "Vx")
        x_vy_loc = read(file, "x_vy")
        y_vy_loc = read(file, "y_vy")
        Vy_loc = read(file, "Vy")
        close(file)

        if (ipy == 1)
          x = append!(x, x_loc .+ ((ipx - 1) * x_loc[end]))
          x_p = append!(x_p, x_p_loc .+ ((ipx - 1) * x_p_loc[end]))
          x_vx = append!(x_vx, x_vx_loc .+ ((ipx - 1) * x_vx_loc[end]))
          x_vy = append!(x_vy, x_vy_loc .+ ((ipx - 1) * x_vy_loc[end]))
        end
        if (ipx == 1)
          y = append!(y, y_loc .+ ((ipy - 1) * y_loc[end]))
          y_p = append!(y_p, y_p_loc .+ ((ipy - 1) * y_p_loc[end]))
          y_vx = append!(y_vx, y_vx_loc .+ ((ipy - 1) * y_vx_loc[end]))
          y_vy = append!(y_vy, y_vy_loc .+ ((ipy - 1) * y_vy_loc[end]))
        end

        nx_i = size(μ_b_loc, 1)
        ny_i = size(μ_b_loc, 2)
        ix1 = 1 + (ipx - 1) * nx_i
        iy1 = 1 + (ipy - 1) * ny_i
        if (ip == 1)
          μ_b = zeros(dims[1] * nx_i, dims[2] * ny_i)
        end
        μ_b[ix1:ix1+nx_i-1, iy1:iy1+ny_i-1] .= μ_b_loc[:, :]

        nx_i = size(P_loc, 1)
        ny_i = size(P_loc, 2)
        ix1 = 1 + (ipx - 1) * nx_i
        iy1 = 1 + (ipy - 1) * ny_i
        if (ip == 1)
          P = zeros(dims[1] * nx_i, dims[2] * ny_i)
        end
        P[ix1:ix1+nx_i-1, iy1:iy1+ny_i-1] .= P_loc[:, :]

        nx_i = size(Vx_loc, 1)
        ny_i = size(Vx_loc, 2)
        ix1 = 1 + (ipx - 1) * nx_i
        iy1 = 1 + (ipy - 1) * ny_i
        if (ip == 1)
          Vx = zeros(dims[1] * nx_i, dims[2] * ny_i)
        end
        Vx[ix1:ix1+nx_i-1, iy1:iy1+ny_i-1] .= Vx_loc[:, :]

        nx_i = size(Vy_loc, 1)
        ny_i = size(Vy_loc, 2)
        ix1 = 1 + (ipx - 1) * nx_i
        iy1 = 1 + (ipy - 1) * ny_i
        if (ip == 1)
          Vy = zeros(dims[1] * nx_i, dims[2] * ny_i)
        end
        Vy[ix1:ix1+nx_i-1, iy1:iy1+ny_i-1] .= Vy_loc[:, :]

        ip += 1
      end
    end

    opts = (size=(1200, 1000), margin=10mm, c=:inferno)
    @assert !(any(isnan, x) || any(isnan, y) || any(isnan, μ_b))
    p1 = heatmap(x, y, μ_b'; title="μ_b", opts...)
    @assert !(any(isnan, x_p) || any(isnan, y_p) || any(isnan, P))
    p2 = heatmap(x_p, y_p, P'; title="Pressure", opts...)
    @assert !(any(isnan, x_vx) || any(isnan, y_vx) || any(isnan, Vx))
    p3 = heatmap(x_vx, y_vx, Vx'; title="Vx", opts...)
    @assert !(any(isnan, x_vy) || any(isnan, y_vy) || any(isnan, Vy))
    p4 = heatmap(x_vy, y_vy, Vy'; title="Vy", opts...)
    plot(p1, p2, p3, p4)
  end

  gif(anim, "viz_out/grid.gif"; fps=animfps)

  return nothing
end

"""
  run()

Loads stats and executes each visualization function
"""
@views function run()
  dims, nt = loadStats()
  vizMarkers(dims, nt)
  vizGrid(dims, nt)
  return nothing
end

run()
