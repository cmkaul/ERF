#include <ERF_VertInterp.H>
#include <AMReX_REAL.H>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool nearly_equal(amrex::Real a, amrex::Real b, amrex::Real tol = amrex::Real(1.0e-12))
{
    return std::abs(static_cast<double>(a - b)) <= static_cast<double>(tol);
}

int test_linear_exact()
{
    // f(z) = 2z + 3 on z=[0,1,2,3]
    std::vector<amrex::Real> z{amrex::Real(0.0), amrex::Real(1.0), amrex::Real(2.0), amrex::Real(3.0)};
    std::vector<amrex::Real> f{amrex::Real(3.0), amrex::Real(5.0), amrex::Real(7.0), amrex::Real(9.0)};

    amrex::Real v = realbdy_interp1d_linear_clamped(z.data(), f.data(), static_cast<int>(z.size()), amrex::Real(1.25));
    amrex::Real e = amrex::Real(5.5);
    if (!nearly_equal(v, e)) {
        std::cerr << "test_linear_exact failed: got " << v << " expected " << e << "\n";
        return 1;
    }
    return 0;
}

int test_clamp_low_high()
{
    std::vector<amrex::Real> z{amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0)};
    std::vector<amrex::Real> f{amrex::Real(1.0), amrex::Real(2.0), amrex::Real(4.0)};

    amrex::Real vlo = realbdy_interp1d_linear_clamped(z.data(), f.data(), static_cast<int>(z.size()), amrex::Real(0.0));
    amrex::Real vhi = realbdy_interp1d_linear_clamped(z.data(), f.data(), static_cast<int>(z.size()), amrex::Real(100.0));
    if (!nearly_equal(vlo, amrex::Real(1.0)) || !nearly_equal(vhi, amrex::Real(4.0))) {
        std::cerr << "test_clamp_low_high failed: got vlo=" << vlo << " vhi=" << vhi << "\n";
        return 1;
    }
    return 0;
}

int test_node_exact()
{
    std::vector<amrex::Real> z{amrex::Real(0.0), amrex::Real(5.0), amrex::Real(9.0)};
    std::vector<amrex::Real> f{amrex::Real(2.0), amrex::Real(7.0), amrex::Real(1.0)};
    amrex::Real v = realbdy_interp1d_linear_clamped(z.data(), f.data(), static_cast<int>(z.size()), amrex::Real(5.0));
    if (!nearly_equal(v, amrex::Real(7.0))) {
        std::cerr << "test_node_exact failed: got " << v << " expected 7\n";
        return 1;
    }
    return 0;
}

} // namespace

int main()
{
    int ierr = 0;
    ierr += test_linear_exact();
    ierr += test_clamp_low_high();
    ierr += test_node_exact();

    if (ierr == 0) {
        std::cout << "ERF_RealBdyInterpUnit: all tests passed.\n";
        return 0;
    }

    std::cerr << "ERF_RealBdyInterpUnit: " << ierr << " test(s) failed.\n";
    return 1;
}
