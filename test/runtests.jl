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
                for npr=1:min(nprocs,4)
                    r = run(ignorestatus(cmd(npr)))
                    @test success(r)
                end
            else
                r = run(ignorestatus(cmd()))
                @test success(r)
            end
        end

    end
end
