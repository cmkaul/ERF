#ifndef ERF_UTILS_H_
#define ERF_UTILS_H_

#include "AMReX.H"
#include "AMReX_MultiFab.H"
#include "AMReX_BCRec.H"
#include "ERF_DataStruct.H"
#include "ERF_IndexDefines.H"
#include "ERF_SurfaceLayer.H"
#include "ERF_FillPatcher.H"
#include "ERF_ReadBndryPlanes.H"

/*
 * Create a new BoxArray in which every grid touches the bottom boundary
 */
void ChopGrids2D (amrex::BoxArray& ba, const amrex::Box& domain, int target_size);

/*
 * Create a new BoxArray with exactly the number of boxes as MPI ranks.
 * This is designed to work only on the base level which covers the entire domain.
 */
amrex::BoxArray
ERFPostProcessBaseGrids (const amrex::Box& domain, bool decompose_in_z);

/*
 * Create the Jacobian for the metric transformation when use_terrain is true
 */
void make_J (const amrex::Geometry& geom,
             amrex::MultiFab& z_phys_nd,
             amrex::MultiFab& detJ_cc);

void make_areas (const amrex::Geometry& geom,
                 amrex::MultiFab& z_phys_nd,
                 amrex::MultiFab& ax,
                 amrex::MultiFab& ay,
                 amrex::MultiFab& az);

/*
 * Average z_phys_nd on nodes to cell centers
 */
void make_zcc (const amrex::Geometry& geom,
               amrex::MultiFab& z_phys_nd,
               amrex::MultiFab& z_phys_cc);

/*
 * Convert momentum to velocity by dividing by density averaged onto faces
 */
void MomentumToVelocity (amrex::MultiFab& xvel_out,
                         amrex::MultiFab& yvel_out,
                         amrex::MultiFab& zvel_out,
                         const amrex::MultiFab& cons_in,
                         const amrex::MultiFab& xmom_in,
                         const amrex::MultiFab& ymom_in,
                         const amrex::MultiFab& zmom_in,
                         const amrex::Box& domain,
                         const amrex::Vector<amrex::BCRec>& domain_bcs_type_h,
                         const amrex::MultiFab* c_vfrac = nullptr // optional
                        );

/*
 * Convert velocity to momentum by multiplying by density averaged onto faces
 */
void VelocityToMomentum (const amrex::MultiFab& xvel_in,
                         const amrex::IntVect & xvel_ngrow,
                         const amrex::MultiFab& yvel_in,
                         const amrex::IntVect & yvel_ngrow,
                         const amrex::MultiFab& zvel_in,
                         const amrex::IntVect & zvel_ngrow,
                         const amrex::MultiFab& cons_in,
                         amrex::MultiFab& xmom_out,
                         amrex::MultiFab& ymom_out,
                         amrex::MultiFab& zmom_out,
                         const amrex::Box& domain,
                         const amrex::Vector<amrex::BCRec>& domain_bcs_type_h,
                         const amrex::MultiFab* c_vfrac = nullptr // optional
                        );

/*
 * Convert (den_div u) to (den_mlt u) by multiplying by (den_mlt/den_div)
 */
void
ConvertForProjection (const amrex::MultiFab& den_div, const amrex::MultiFab& den_mlt,
                      amrex::MultiFab& xmom, amrex::MultiFab& ymom, amrex::MultiFab& zmom,
                      const amrex::Box& domain, const amrex::Vector<amrex::BCRec>& domain_bcs_type_h);

/*
 * \brief Enforces solvability by scaling outflow to match with inflow.
 *
 */
void enforceInOutSolvability (int lev,
                              amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& vels_vec,
                              amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& area_vec,
                              const amrex::Geometry& geom);

/*
 * Compute boxes for looping over interior/exterior ghost cells
 * for use by fillpatch, erf_slow_rhs_pre, and erf_slow_rhs_post
 */
void realbdy_interior_bxs_xy (const amrex::Box& bx,
                              const amrex::Box& domain,
                              const int& width,
                              amrex::Box& bx_xlo,
                              amrex::Box& bx_xhi,
                              amrex::Box& bx_ylo,
                              amrex::Box& bx_yhi,
                              const amrex::IntVect& ng_vect=amrex::IntVect(0,0,0),
                              const bool get_int_ng=false,
                              const bool y_face_owns_corners=false);

/*
 * Compute relaxation region RHS with wrfbdy
 */
void realbdy_compute_interior_ghost_rhs (const amrex::Real& total_time,
                                         const amrex::Real& delta_t,
                                         const amrex::Real& start_bdy_time,
                                         const amrex::Real& final_bdy_time,
                                         const amrex::Real& bdy_time_interval,
                                         const amrex::Real& nudge_factor,
                                         int width,
                                         bool do_upwind,
                                         const amrex::Geometry& geom,
                                         amrex::Vector<amrex::MultiFab>& S_rhs,
                                         amrex::Vector<amrex::MultiFab>& S_cur_data,
                                         amrex::Vector<amrex::Vector<amrex::FArrayBox>>& bdy_data_xlo,
                                         amrex::Vector<amrex::Vector<amrex::FArrayBox>>& bdy_data_xhi,
                                         amrex::Vector<amrex::Vector<amrex::FArrayBox>>& bdy_data_ylo,
                                         amrex::Vector<amrex::Vector<amrex::FArrayBox>>& bdy_data_yhi,
                                         std::unique_ptr<ReadBndryPlanes>& m_r2d);

/*
 * Compute relaxation region RHS at fine-crse interface
 */
void
fine_compute_interior_ghost_rhs (const amrex::Real& time,
                                 const amrex::Real& delta_t,
                                 const int& width,
                                 const int& set_width,
                                 const amrex::Geometry& geom,
                                 ERFFillPatcher* FPr_c,
                                 ERFFillPatcher* FPr_u,
                                 ERFFillPatcher* FPr_v,
                                 ERFFillPatcher* FPr_w,
                                 amrex::Vector<amrex::BCRec>& domain_bcs_type,
                                 amrex::Vector<amrex::MultiFab>& S_rhs_f,
                                 amrex::Vector<amrex::MultiFab>& S_data_f);

/*
 * Accumulate time averaged velocity fields
 */
void
Time_Avg_Vel_atCC (const amrex::Real& dt,
                   amrex::Real& t_avg_cnt,
                   amrex::MultiFab* vel_t_avg,
                   amrex::MultiFab& xvel,
                   amrex::MultiFab& yvel,
                   amrex::MultiFab& zvel);

/**
 * Compute the nudging relaxation
 *
 * @param[in] delta_t time step
 * @param[in] icomp component offset
 * @param[in] num_var number of variables to loop
 * @param[in] width width of wrf bdy file
 * @param[in] dom_lo low bound of domain
 * @param[in] dom_hi high bound of domain
 * @param[in] F1 drift relaxation parameter
 * @param[in] F2 Laplacian relaxation parameter
 * @param[in] bx_xlo box for low x relaxation
 * @param[in] bx_xhi box for high x relaxation
 * @param[in] bx_ylo box for low y relaxation
 * @param[in] bx_yhi box for high y relaxation
 * @param[in] arr_xlo array for low x relaxation
 * @param[in] arr_xhi array for high x relaxation
 * @param[in] arr_ylo array for low y relaxation
 * @param[in] arr_yhi array for high y relaxation
 * @param[in] data_arr data array
 * @param[out] rhs_arr RHS array
 */
AMREX_GPU_HOST
AMREX_FORCE_INLINE
amrex::Real
realbdy_weight_shape (amrex::Real s,
                      const int profile_id,
                      const amrex::Real tanh_beta)
{
    // Clamp to [0,1] inside nudging strip.
    s = amrex::max(amrex::Real(0.0), amrex::min(amrex::Real(1.0), s));

    // 0: quadratic (legacy), 1: cosine, 2: tanh-mapped
    if (profile_id == 1) {
        constexpr amrex::Real pi = amrex::Real(3.1415926535897932384626433832795);
        return amrex::Real(0.5) * (amrex::Real(1.0) - std::cos(pi * s));
    } else if (profile_id == 2) {
        amrex::Real b = amrex::max(tanh_beta, amrex::Real(1.0e-6));
        amrex::Real den = std::tanh(b);
        return (std::tanh(b * (amrex::Real(2.0)*s - amrex::Real(1.0))) + den) / (amrex::Real(2.0)*den);
    }
    return s * s;
}

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void
realbdy_compute_relaxation (const int& icomp,
                            const int& num_var,
                            const int& width,
                            const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>& dx,
                            const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>& ProbLo,
                            const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>& ProbHi,
                            const amrex::Real& F1,
                            const amrex::Box& domain_cc,
                            const amrex::Box& bx_xlo,
                            const amrex::Box& bx_xhi,
                            const amrex::Box& bx_ylo,
                            const amrex::Box& bx_yhi,
                            const amrex::Array4<const amrex::Real>& arr_xlo,
                            const amrex::Array4<const amrex::Real>& arr_xhi,
                            const amrex::Array4<const amrex::Real>& arr_ylo,
                            const amrex::Array4<const amrex::Real>& arr_yhi,
                            const amrex::Array4<const amrex::Real>& u_xlo,
                            const amrex::Array4<const amrex::Real>& u_xhi,
                            const amrex::Array4<const amrex::Real>& v_xlo,
                            const amrex::Array4<const amrex::Real>& v_xhi,
                            const amrex::Array4<const amrex::Real>& v_ylo,
                            const amrex::Array4<const amrex::Real>& v_yhi,
                            const amrex::Array4<const amrex::Real>& rho_cc_arr,
                            const amrex::Array4<const amrex::Real>& data_arr,
                            const amrex::Array4<amrex::Real>& rhs_arr,
                            const amrex::Real& nudge_scale,
                            const bool& apply_x_nudging,
                            const bool& apply_y_nudging,
                            const bool& exclude_x_corners,
                            const amrex::Real& force_factor,
                            const bool& do_upwind,
                            const bool y_face_owns_corners = false,
                            const bool y_face_use_max_metric = true,
                            const bool use_primitive_rhs = false,
                            const int weight_profile_id = 0,
                            const amrex::Real weight_tanh_beta = amrex::Real(2.5))
{
    amrex::IntVect iv = bx_xlo.type();
    amrex::Real ioff  = (iv[0]==1) ? zero : myhalf;
    amrex::Real joff  = (iv[1]==1) ? zero : myhalf;
    const auto& dom_cc_lo = lbound(domain_cc);
    const auto& dom_cc_hi = ubound(domain_cc);
    amrex::ParallelFor(bx_xlo, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (!apply_x_nudging) { return; }
        // Corners with x boxes
        amrex::Real x = ProbLo[0] + (i + ioff) * dx[0];
        amrex::Real y = ProbLo[1] + (j + joff) * dx[1];
        amrex::Real x_end  = ProbLo[0] + width * dx[0];
        amrex::Real y_end  = ProbLo[1] + width * dx[1];
        amrex::Real y_strt = ProbHi[1] - width * dx[1];
        amrex::Real xi     = (x_end - x) / (x_end - ProbLo[0]);
        amrex::Real eta_lo = (y < y_end ) ? (y_end  - y) / (y_end  - ProbLo[1]) : zero;
        amrex::Real eta_hi = (y > y_strt) ? (y - y_strt) / (ProbHi[1] - y_strt) : zero;
        amrex::Real eta    = std::max(eta_lo,eta_hi);
        if (exclude_x_corners && eta > zero) { return; }
        amrex::Real wx = realbdy_weight_shape(xi,  weight_profile_id, weight_tanh_beta);
        amrex::Real wy = realbdy_weight_shape(eta, weight_profile_id, weight_tanh_beta);
        amrex::Real Factor = std::max(wx,wy);
        if (force_factor >= zero) { Factor = force_factor; }

        int ii = std::min(std::max(i,dom_cc_lo.x),dom_cc_hi.x);
        int jj = std::min(std::max(j,dom_cc_lo.y),dom_cc_hi.y);
        amrex::Real rho_loc = amrex::max(rho_cc_arr(ii,jj,k,Rho_comp), amrex::Real(1.e-16));
        if (iv[0] == 1 && iv[1] == 0) {
            int iim1 = std::min(std::max(i-1,dom_cc_lo.x),dom_cc_hi.x);
            rho_loc = amrex::max(myhalf * (rho_cc_arr(iim1,jj,k,Rho_comp) + rho_cc_arr(ii,jj,k,Rho_comp)), amrex::Real(1.e-16));
        } else if (iv[0] == 0 && iv[1] == 1) {
            int jjm1 = std::min(std::max(j-1,dom_cc_lo.y),dom_cc_hi.y);
            rho_loc = amrex::max(myhalf * (rho_cc_arr(ii,jjm1,k,Rho_comp) + rho_cc_arr(ii,jj,k,Rho_comp)), amrex::Real(1.e-16));
        }
        amrex::Real delta = arr_xlo(i  ,j  ,k,n) - data_arr(i  ,j  ,k  ,n+icomp);
        if (use_primitive_rhs) {
            amrex::Real targ_prim = arr_xlo(i,j,k,n) / rho_loc;
            amrex::Real cur_prim  = data_arr(i,j,k,n+icomp) / rho_loc;
            delta = targ_prim - cur_prim;
        }
        amrex::Real Temp  = nudge_scale * Factor*F1*delta;
        if (!do_upwind) {
            rhs_arr(i,j,k,n+icomp) += Temp;
        } else {
            int jju = std::min(std::max(j,dom_cc_lo.y),dom_cc_hi.y);
            int iiv = std::min(std::max(i,dom_cc_lo.x),dom_cc_hi.x);
            if ( (u_xlo(dom_cc_lo.x,jju,k) >= zero) ||
                 ((j == dom_cc_lo.y      ) && (v_xlo(iiv,dom_cc_lo.y  ,k) >= zero)) ||
                 ((j == dom_cc_hi.y+iv[1]) && (v_xlo(iiv,dom_cc_hi.y+1,k) <= zero)) ) {
                rhs_arr(i,j,k,n+icomp) += Temp;
            }
        }
    },
    bx_xhi, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (!apply_x_nudging) { return; }
        // Corners with x boxes
        amrex::Real x = ProbLo[0] + (i + ioff) * dx[0];
        amrex::Real y = ProbLo[1] + (j + joff) * dx[1];
        amrex::Real x_strt = ProbHi[0] - width * dx[0];
        amrex::Real y_strt = ProbHi[1] - width * dx[1];
        amrex::Real y_end  = ProbLo[1] + width * dx[1];
        amrex::Real xi     = (x - x_strt) / (ProbHi[0] - x_strt);
        amrex::Real eta_lo = (y < y_end ) ? (y_end  - y) / (y_end  - ProbLo[1]) : zero;
        amrex::Real eta_hi = (y > y_strt) ? (y - y_strt) / (ProbHi[1] - y_strt) : zero;
        amrex::Real eta    = std::max(eta_lo,eta_hi);
        if (exclude_x_corners && eta > zero) { return; }
        amrex::Real wx = realbdy_weight_shape(xi,  weight_profile_id, weight_tanh_beta);
        amrex::Real wy = realbdy_weight_shape(eta, weight_profile_id, weight_tanh_beta);
        amrex::Real Factor = std::max(wx,wy);
        if (force_factor >= zero) { Factor = force_factor; }

        int ii = std::min(std::max(i,dom_cc_lo.x),dom_cc_hi.x);
        int jj = std::min(std::max(j,dom_cc_lo.y),dom_cc_hi.y);
        amrex::Real rho_loc = amrex::max(rho_cc_arr(ii,jj,k,Rho_comp), amrex::Real(1.e-16));
        if (iv[0] == 1 && iv[1] == 0) {
            int iim1 = std::min(std::max(i-1,dom_cc_lo.x),dom_cc_hi.x);
            rho_loc = amrex::max(myhalf * (rho_cc_arr(iim1,jj,k,Rho_comp) + rho_cc_arr(ii,jj,k,Rho_comp)), amrex::Real(1.e-16));
        } else if (iv[0] == 0 && iv[1] == 1) {
            int jjm1 = std::min(std::max(j-1,dom_cc_lo.y),dom_cc_hi.y);
            rho_loc = amrex::max(myhalf * (rho_cc_arr(ii,jjm1,k,Rho_comp) + rho_cc_arr(ii,jj,k,Rho_comp)), amrex::Real(1.e-16));
        }
        amrex::Real delta = arr_xhi(i  ,j  ,k,n) - data_arr(i  ,j  ,k  ,n+icomp);
        if (use_primitive_rhs) {
            amrex::Real targ_prim = arr_xhi(i,j,k,n) / rho_loc;
            amrex::Real cur_prim  = data_arr(i,j,k,n+icomp) / rho_loc;
            delta = targ_prim - cur_prim;
        }
        amrex::Real Temp  = nudge_scale * Factor*F1*delta;
        if (!do_upwind) {
            rhs_arr(i,j,k,n+icomp) += Temp;
        } else {
            int jju = std::min(std::max(j,dom_cc_lo.y),dom_cc_hi.y);
            int iiv = std::min(std::max(i,dom_cc_lo.x),dom_cc_hi.x);
            if ( (u_xhi(dom_cc_hi.x+1,jju,k) <= zero) ||
                 ((j == dom_cc_lo.y      ) && (v_xhi(iiv,dom_cc_lo.y  ,k) >= zero)) ||
                 ((j == dom_cc_hi.y+iv[1]) && (v_xhi(iiv,dom_cc_hi.y+1,k) <= zero)) ) {
                rhs_arr(i,j,k,n+icomp) += Temp;
            }
        }
    });

    amrex::ParallelFor(bx_ylo, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (!apply_y_nudging) { return; }
        // Legacy y-side weighting is eta^2. If y faces own corners, use
        // symmetric corner-aware weighting max(xi^2, eta^2).
        amrex::Real x      = ProbLo[0] + (i + ioff) * dx[0];
        amrex::Real y      = ProbLo[1] + (j + joff) * dx[1];
        amrex::Real x_end  = ProbLo[0] + width * dx[0];
        amrex::Real x_strt = ProbHi[0] - width * dx[0];
        amrex::Real y_end  = ProbLo[1] + width * dx[1];
        amrex::Real xi_lo  = (x < x_end ) ? (x_end  - x) / (x_end  - ProbLo[0]) : zero;
        amrex::Real xi_hi  = (x > x_strt) ? (x - x_strt) / (ProbHi[0] - x_strt) : zero;
        amrex::Real xi     = std::max(xi_lo,xi_hi);
        amrex::Real eta    = (y_end - y) / (y_end - ProbLo[1]);
        amrex::Real wx = realbdy_weight_shape(xi,  weight_profile_id, weight_tanh_beta);
        amrex::Real wy = realbdy_weight_shape(eta, weight_profile_id, weight_tanh_beta);
        amrex::Real Factor = (y_face_owns_corners && y_face_use_max_metric)
                                 ? std::max(wx,wy) : wy;
        if (force_factor >= zero) { Factor = force_factor; }

        int ii = std::min(std::max(i,dom_cc_lo.x),dom_cc_hi.x);
        int jj = std::min(std::max(j,dom_cc_lo.y),dom_cc_hi.y);
        amrex::Real rho_loc = amrex::max(rho_cc_arr(ii,jj,k,Rho_comp), amrex::Real(1.e-16));
        if (iv[0] == 1 && iv[1] == 0) {
            int iim1 = std::min(std::max(i-1,dom_cc_lo.x),dom_cc_hi.x);
            rho_loc = amrex::max(myhalf * (rho_cc_arr(iim1,jj,k,Rho_comp) + rho_cc_arr(ii,jj,k,Rho_comp)), amrex::Real(1.e-16));
        } else if (iv[0] == 0 && iv[1] == 1) {
            int jjm1 = std::min(std::max(j-1,dom_cc_lo.y),dom_cc_hi.y);
            rho_loc = amrex::max(myhalf * (rho_cc_arr(ii,jjm1,k,Rho_comp) + rho_cc_arr(ii,jj,k,Rho_comp)), amrex::Real(1.e-16));
        }
        amrex::Real delta = arr_ylo(i  ,j  ,k,n) - data_arr(i  ,j  ,k  ,n+icomp);
        if (use_primitive_rhs) {
            amrex::Real targ_prim = arr_ylo(i,j,k,n) / rho_loc;
            amrex::Real cur_prim  = data_arr(i,j,k,n+icomp) / rho_loc;
            delta = targ_prim - cur_prim;
        }
        amrex::Real Temp  = nudge_scale * Factor*F1*delta;
        if (!do_upwind) {
            rhs_arr(i,j,k,n+icomp) += Temp;
        } else {
            int iiv = std::min(std::max(i,dom_cc_lo.x+width),dom_cc_hi.x-width);
            if (v_ylo(iiv,dom_cc_lo.y,k) >= zero) {
                rhs_arr(i,j,k,n+icomp) += Temp;
            }
        }
    },
    bx_yhi, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (!apply_y_nudging) { return; }
        // Legacy y-side weighting is eta^2. If y faces own corners, use
        // symmetric corner-aware weighting max(xi^2, eta^2).
        amrex::Real x      = ProbLo[0] + (i + ioff) * dx[0];
        amrex::Real y      = ProbLo[1] + (j + joff) * dx[1];
        amrex::Real x_end  = ProbLo[0] + width * dx[0];
        amrex::Real x_strt = ProbHi[0] - width * dx[0];
        amrex::Real y_strt = ProbHi[1] - width * dx[1];
        amrex::Real xi_lo  = (x < x_end ) ? (x_end  - x) / (x_end  - ProbLo[0]) : zero;
        amrex::Real xi_hi  = (x > x_strt) ? (x - x_strt) / (ProbHi[0] - x_strt) : zero;
        amrex::Real xi     = std::max(xi_lo,xi_hi);
        amrex::Real eta    = (y - y_strt) / (ProbHi[1] - y_strt);
        amrex::Real wx = realbdy_weight_shape(xi,  weight_profile_id, weight_tanh_beta);
        amrex::Real wy = realbdy_weight_shape(eta, weight_profile_id, weight_tanh_beta);
        amrex::Real Factor = (y_face_owns_corners && y_face_use_max_metric)
                                 ? std::max(wx,wy) : wy;
        if (force_factor >= zero) { Factor = force_factor; }

        int ii = std::min(std::max(i,dom_cc_lo.x),dom_cc_hi.x);
        int jj = std::min(std::max(j,dom_cc_lo.y),dom_cc_hi.y);
        amrex::Real rho_loc = amrex::max(rho_cc_arr(ii,jj,k,Rho_comp), amrex::Real(1.e-16));
        if (iv[0] == 1 && iv[1] == 0) {
            int iim1 = std::min(std::max(i-1,dom_cc_lo.x),dom_cc_hi.x);
            rho_loc = amrex::max(myhalf * (rho_cc_arr(iim1,jj,k,Rho_comp) + rho_cc_arr(ii,jj,k,Rho_comp)), amrex::Real(1.e-16));
        } else if (iv[0] == 0 && iv[1] == 1) {
            int jjm1 = std::min(std::max(j-1,dom_cc_lo.y),dom_cc_hi.y);
            rho_loc = amrex::max(myhalf * (rho_cc_arr(ii,jjm1,k,Rho_comp) + rho_cc_arr(ii,jj,k,Rho_comp)), amrex::Real(1.e-16));
        }
        amrex::Real delta = arr_yhi(i  ,j  ,k,n) - data_arr(i  ,j  ,k  ,n+icomp);
        if (use_primitive_rhs) {
            amrex::Real targ_prim = arr_yhi(i,j,k,n) / rho_loc;
            amrex::Real cur_prim  = data_arr(i,j,k,n+icomp) / rho_loc;
            delta = targ_prim - cur_prim;
        }
        amrex::Real Temp  = nudge_scale * Factor*F1*delta;
        if (!do_upwind) {
            rhs_arr(i,j,k,n+icomp) += Temp;
        } else {
            int iiv = std::min(std::max(i,dom_cc_lo.x+width),dom_cc_hi.x-width);
            if (v_yhi(iiv,dom_cc_hi.y+1,k) >= zero) {
                rhs_arr(i,j,k,n+icomp) += Temp;
            }
        }
    });
}

/*
 * Effectively a Multiply for a MultiFab and an iMultiFab mask
 */
AMREX_GPU_HOST
AMREX_FORCE_INLINE
void
ApplyMask (amrex::MultiFab& dst,
           const amrex::iMultiFab& imask,
           const int nghost = 0)
{
    for (amrex::MFIter mfi(dst,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox(nghost);
        if (bx.ok())
        {
            auto        dstFab = dst.array(mfi);
            const auto maskFab = imask.const_array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                dstFab(i,j,k) *= maskFab(i,j,k);
            });
        }
    }
}

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void
ApplyInvertedMask (amrex::MultiFab& dst,
                   const amrex::iMultiFab& imask,
                   const int nghost = 0)
{
    for (amrex::MFIter mfi(dst,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox(nghost);
        if (bx.ok())
        {
            auto        dstFab = dst.array(mfi);
            const auto maskFab = imask.const_array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                dstFab(i,j,k) *= (1-maskFab(i,j,k));
            });
        }
    }
}

void
thinbody_wall_dist (std::unique_ptr<amrex::MultiFab>& wdist,
                    amrex::Vector<amrex::IntVect>& xfaces,
                    amrex::Vector<amrex::IntVect>& yfaces,
                    amrex::Vector<amrex::IntVect>& zfaces,
                    const amrex::Geometry& geomdata,
                    std::unique_ptr<amrex::MultiFab>& z_phys_cc);


void WeatherDataInterpolation(const amrex::Real time);

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
quad_interp_1d(amrex::Real z,
               amrex::Real z0, amrex::Real p0,
               amrex::Real z1, amrex::Real p1,
               amrex::Real z2, amrex::Real p2)
{
    return p0 * ( (z - z1) * (z - z2) ) / ( (z0 - z1) * (z0 - z2) )
         + p1 * ( (z - z0) * (z - z2) ) / ( (z1 - z0) * (z1 - z2) )
         + p2 * ( (z - z0) * (z - z1) ) / ( (z2 - z0) * (z2 - z1) );
}

#endif
