#ifndef ERF_VERT_INTERP_H_
#define ERF_VERT_INTERP_H_

#include "AMReX_REAL.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Math.H"

/**
 * Interpolate a 1D profile from source z-levels to a target z.
 *
 * Requirements:
 * - z_src has length nsrc >= 2
 * - z_src is monotonic increasing
 *
 * Behavior:
 * - If z_tgt is outside [z_src[0], z_src[nsrc-1]], returns endpoint value
 *   (clamped extrapolation).
 * - Otherwise performs piecewise-linear interpolation.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
realbdy_interp1d_linear_clamped (const amrex::Real* z_src,
                                 const amrex::Real* f_src,
                                 const int nsrc,
                                 const amrex::Real z_tgt) noexcept
{
    if (nsrc <= 1) { return f_src[0]; }

    if (z_tgt <= z_src[0])        { return f_src[0]; }
    if (z_tgt >= z_src[nsrc - 1]) { return f_src[nsrc - 1]; }

    int klo = 0;
    int khi = nsrc - 1;
    while (khi - klo > 1) {
        int km = (klo + khi) / 2;
        if (z_tgt < z_src[km]) {
            khi = km;
        } else {
            klo = km;
        }
    }

    amrex::Real z0 = z_src[klo];
    amrex::Real z1 = z_src[khi];
    amrex::Real f0 = f_src[klo];
    amrex::Real f1 = f_src[khi];
    amrex::Real dz = z1 - z0;

    if (amrex::Math::abs(dz) <= amrex::Real(1.0e-14)) {
        return f0;
    }

    amrex::Real alpha = (z_tgt - z0) / dz;
    return (amrex::Real(1.0) - alpha) * f0 + alpha * f1;
}

#endif

