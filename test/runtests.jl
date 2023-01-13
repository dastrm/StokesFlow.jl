using Test
import MPI

# This test driver is inspired by the tests from MPI.jl

testdir = @__DIR__
istest(f) = endswith(f, ".jl") && startswith(f, "test_")
testfiles = sort(filter(istest, readdir(testdir)))

nprocs = 4
USE_GPU = false

@testset "$f" for f in testfiles
    MPI.mpiexec() do mpirun
        cmd(n=nprocs) = `$mpirun -n $n $(Base.julia_cmd()) $(joinpath(testdir, f))`

        withenv("USE_GPU" => USE_GPU) do

            if f == "test_StokesSolver.jl"
                # run this test with up to 4 processes
                # TODO: this test passes, except when BOTH --check-bounds=true and USE_GPU==true (however no bounds-check fails...)
                for npr = 1:min(nprocs, 4)
                    r = run(ignorestatus(cmd(npr)))
                    @test success(r)
                end
            elseif f == "test_MarkerExchange_1P.jl"
                # run this test with exactly 1 process
                r = run(ignorestatus(cmd(1)))
                @test success(r)
            elseif f == "test_MarkerExchange_9P.jl"
                # run this test with exactly 9 processes
                r = run(ignorestatus(cmd(9)))
                @test success(r)
            else
                r = run(ignorestatus(cmd()))
                @test success(r)
            end
        end

    end
end
