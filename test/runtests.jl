using RemotePhaseRetrieval
using RemotePhaseRetrieval: PSFWorkspace, make_psfGenerator_vector_mk, CudaPSFWorkspace
using HDF5
using FFTW
using Test
using LsqFit
using CUDA

@testset "RemotePhaseRetrieval.jl" begin
    f = h5open("mock_psf_test.h5", "r")
    w = PSFWorkspace(f["a"][], f["aber"][], f["zComp"][])
    cw = CudaPSFWorkspace(f["a"][], f["aber"][], f["zComp"][])
    cw32 = CudaPSFWorkspace{Float32}(f["a"][], f["aber"][], f["zComp"][])
    ub = f["ub"][]
    lb = f["lb"][]
    psf = f["psf"][]
    psf_vec = vec(similar(psf))
    param0 = f["param0"][]
    param = f["param"][]
    close(f)
    objfun = make_psfGenerator_vector_mk(w)
    @time result = curve_fit(objfun, [], ifftshift(ifftshift(psf, 1), 2)[:], copy(param0[:]); upper = ub[:], lower = lb[:], inplace=true, x_tol=5e-3)
end
