#if defined(ERF_USE_NETCDF)

#include <ERF_SrcHeaders.H>
#include <ERF_Utils.H>

using namespace amrex;

/**
 * Function for setting the slow variables in the "specified" zones at the domain boundary
 *
*/

void
moist_set_rhs (const Geometry& geom,
               const Box& tbx,
               const Array4<Real const>& new_cons,
               const Array4<Real      >& cell_rhs,
               const Array4<Real const>& mub_arr,
               const Array4<Real const>& c1f_arr,
               const Array4<Real const>& c2f_arr,
               const Array4<Real const>& phb_arr,
               const Array4<Real const>& zcc_arr,
               const Real& time,
               const Real& dt,
               const Real& start_bdy_time,
               const Real& final_bdy_time,
               const Real& bdy_time_interval,
               const Real& nudge_factor,
               int  width,
               bool do_upwind,
               const Box& domain,
               Vector<Vector<FArrayBox>>& bdy_data_xlo,
               Vector<Vector<FArrayBox>>& bdy_data_xhi,
               Vector<Vector<FArrayBox>>& bdy_data_ylo,
               Vector<Vector<FArrayBox>>& bdy_data_yhi,
               std::unique_ptr<ReadBndryPlanes>& m_r2d)
{
    // Debug controls for isolating real-boundary nudging behavior.
    static bool dbg_flags_init = false;
    static bool dbg_nudge_tq = true;
    static bool dbg_nudge_x = true;
    static bool dbg_nudge_y = true;
    static bool dbg_nudge_print_stats = false;
    static bool dbg_nudge_freeze_alpha = false;
    static bool dbg_nudge_exclude_x_corners = false;
    static bool dbg_realbdy_yface_corner_owner = false;
    static bool dbg_realbdy_component_corner_owner = false;
    static bool dbg_realbdy_yface_corner_use_max_metric = true;
    static bool realbdy_vertical_remap_theta = false;
    static int realbdy_vertical_remap_mode = 1; // 1=A-2 (default), 2=B, 3=C, 4=D
    static std::string dbg_realbdy_weight_profile = "quadratic";
    static int dbg_realbdy_weight_profile_id = 0; // 0=quadratic,1=cosine,2=tanh
    static Real dbg_realbdy_weight_tanh_beta = Real(2.5);
    static Real dbg_nudge_const_factor = Real(-1.0);
    static Real dbg_nudge_factor_tq = Real(-1.0);
    if (!dbg_flags_init) {
        ParmParse pp("erf");
        pp.query("dbg_nudge_tq", dbg_nudge_tq);
        pp.query("dbg_nudge_x",  dbg_nudge_x);
        pp.query("dbg_nudge_y",  dbg_nudge_y);
        pp.query("dbg_nudge_print_stats", dbg_nudge_print_stats);
        pp.query("dbg_nudge_freeze_alpha", dbg_nudge_freeze_alpha);
        pp.query("dbg_nudge_exclude_x_corners", dbg_nudge_exclude_x_corners);
        pp.query("dbg_realbdy_yface_corner_owner", dbg_realbdy_yface_corner_owner);
        pp.query("dbg_realbdy_component_corner_owner", dbg_realbdy_component_corner_owner);
        pp.query("dbg_realbdy_yface_corner_use_max_metric", dbg_realbdy_yface_corner_use_max_metric);
        pp.query("realbdy_vertical_remap_theta", realbdy_vertical_remap_theta);
        pp.query("realbdy_vertical_remap_mode", realbdy_vertical_remap_mode);
        pp.query("dbg_realbdy_weight_profile", dbg_realbdy_weight_profile);
        pp.query("dbg_realbdy_weight_tanh_beta", dbg_realbdy_weight_tanh_beta);
        pp.query("dbg_nudge_const_factor", dbg_nudge_const_factor);
        pp.query("dbg_nudge_factor_tq", dbg_nudge_factor_tq);

        if (dbg_realbdy_weight_profile == "quadratic" || dbg_realbdy_weight_profile == "quad") {
            dbg_realbdy_weight_profile_id = 0;
        } else if (dbg_realbdy_weight_profile == "cosine" || dbg_realbdy_weight_profile == "cos") {
            dbg_realbdy_weight_profile_id = 1;
        } else if (dbg_realbdy_weight_profile == "tanh") {
            dbg_realbdy_weight_profile_id = 2;
        } else {
            if (ParallelDescriptor::IOProcessor()) {
                Print() << "WARNING: Unknown erf.dbg_realbdy_weight_profile='"
                        << dbg_realbdy_weight_profile
                        << "'. Falling back to 'quadratic'.\n";
            }
            dbg_realbdy_weight_profile_id = 0;
            dbg_realbdy_weight_profile = "quadratic";
        }
        if (realbdy_vertical_remap_mode != 1 &&
            realbdy_vertical_remap_mode != 2 &&
            realbdy_vertical_remap_mode != 3 &&
            realbdy_vertical_remap_mode != 4) {
            if (ParallelDescriptor::IOProcessor()) {
                Print() << "WARNING: Unknown erf.realbdy_vertical_remap_mode="
                        << realbdy_vertical_remap_mode
                        << ". Falling back to mode=1 (A-2).\n";
            }
            realbdy_vertical_remap_mode = 1;
        }
        dbg_flags_init = true;
    }
    bool y_face_owns_corners = dbg_realbdy_component_corner_owner ? false : dbg_realbdy_yface_corner_owner;
    Real nudge_scale = dbg_nudge_tq ? Real(1.0) : Real(0.0);

    // HACK HACK HACK
    // Get bndry data
    int bdy_comp = BCVars::RhoQ1_bc_comp;
    Array4<Real> bdatxlo, bdatxhi, bdatylo, bdatyhi;
    if (m_r2d) {
        Vector<std::unique_ptr<PlaneVector>>& bndry_data = m_r2d->interp_in_time(time);
        bdatxlo = (*bndry_data[0])[0].array();
        bdatylo = (*bndry_data[1])[0].array();
        bdatxhi = (*bndry_data[3])[0].array();
        bdatyhi = (*bndry_data[4])[0].array();
    }

    //
    // Note that time (= start_time+old_stage_time)  is measured as total time
    //           start_bdy_time and final_bdy_time are also measured as total time
    //

    // Relaxation constants
    Real nudge_factor_local = nudge_factor;
    if (dbg_nudge_factor_tq > Real(0.0)) { nudge_factor_local = dbg_nudge_factor_tq; }
    Real F1 = one/(nudge_factor_local*dt);

    // Domain bounds
    const auto& dom_hi = ubound(domain);
    const auto& dom_lo = lbound(domain);
    auto dx = geom.CellSizeArray();
    auto ProbHi = geom.ProbHiArray();
    auto ProbLo = geom.ProbLoArray();

    // Time interpolation
    Real dT = bdy_time_interval;

    int n_time    = static_cast<int>( (time-start_bdy_time) /  dT);
    int n_time_p1 = n_time + 1;
    Real alpha    = ((time-start_bdy_time) - n_time * dT) / dT;

    // Do not over run the last bdy file
    if (time >= final_bdy_time) {
      n_time    = static_cast<int>( (final_bdy_time - start_bdy_time)/ dT);
      n_time_p1 = n_time;
      alpha     = zero;
    }

    AMREX_ALWAYS_ASSERT( alpha >= zero && alpha <= one);
    Real oma   = one - alpha;
    if (dbg_nudge_freeze_alpha) {
        oma = one;
        alpha = zero;
    }
    const bool have_wrfbdy_ph = (bdy_data_xlo[n_time].size() > WRFBdyVars::PH &&
                                 bdy_data_xlo[n_time_p1].size() > WRFBdyVars::PH);
    const bool use_qv_vertical_remap = realbdy_vertical_remap_theta && have_wrfbdy_ph;
    const bool use_qv_modeB = (realbdy_vertical_remap_mode == 2);
    const bool use_qv_modeC = (realbdy_vertical_remap_mode == 3);
    const bool use_qv_modeD = (realbdy_vertical_remap_mode == 4);
    if (realbdy_vertical_remap_theta && !have_wrfbdy_ph && ParallelDescriptor::IOProcessor()) {
        Print() << "[REALBDY qv-remap] PH boundary field unavailable; disabling qv vertical remap.\n";
    }
    if (use_qv_vertical_remap && (use_qv_modeB || use_qv_modeC || use_qv_modeD) && ParallelDescriptor::IOProcessor()) {
        Print() << "[REALBDY qv-remap] mode=" << realbdy_vertical_remap_mode
                << " selected; for cell-centered qv this uses the same "
                << "cc remap operator as mode=1.\n";
    }

    /*
    // UNIT TEST DEBUG
    oma = one; alpha = zero;
    */

    // NOTE: The sizing of the temporary BDY FABS is
    //       GLOBAL and occurs over the entire BDY region.

    // Size the FABs
    //==========================================================
    // NOTE: No ghost cells, we force mask to be idx type (0,0,0)
    IntVect ng_vect(0);
    Box gdom(domain); gdom.grow(ng_vect);
    Box bx_xlo, bx_xhi, bx_ylo, bx_yhi;
    realbdy_interior_bxs_xy(gdom, domain, width,
                            bx_xlo, bx_xhi,
                            bx_ylo, bx_yhi,
                            ng_vect, true,
                            y_face_owns_corners);

    // Temporary FABs for storage (owned/filled on all ranks)
    FArrayBox QV_xlo, QV_xhi, QV_ylo, QV_yhi;
    QV_xlo.resize(bx_xlo,1,The_Async_Arena()); QV_xhi.resize(bx_xhi,1,The_Async_Arena());
    QV_ylo.resize(bx_ylo,1,The_Async_Arena()); QV_yhi.resize(bx_yhi,1,The_Async_Arena());

    // Masks for upwinding
    FArrayBox U_xlo, U_xhi, V_xlo, V_xhi, V_ylo, V_yhi;
    U_xlo.resize(convert(bx_xlo,IntVect(1,0,0)),1,The_Async_Arena());
    U_xhi.resize(convert(bx_xhi,IntVect(1,0,0)),1,The_Async_Arena());
    V_xlo.resize(convert(bx_xlo,IntVect(0,1,0)),1,The_Async_Arena());
    V_xhi.resize(convert(bx_xhi,IntVect(0,1,0)),1,The_Async_Arena());
    V_ylo.resize(convert(bx_ylo,IntVect(0,1,0)),1,The_Async_Arena());
    V_yhi.resize(convert(bx_yhi,IntVect(0,1,0)),1,The_Async_Arena());

    // Populate FABs from bdy interpolation (primitive vars)
    //==========================================================
    const auto& bdatxlo_n   = bdy_data_xlo[n_time   ][WRFBdyVars::QV].const_array();
    const auto& bdatxlo_np1 = bdy_data_xlo[n_time_p1][WRFBdyVars::QV].const_array();
    const auto& bdatxhi_n   = bdy_data_xhi[n_time   ][WRFBdyVars::QV].const_array();
    const auto& bdatxhi_np1 = bdy_data_xhi[n_time_p1][WRFBdyVars::QV].const_array();
    const auto& bdatylo_n   = bdy_data_ylo[n_time   ][WRFBdyVars::QV].const_array();
    const auto& bdatylo_np1 = bdy_data_ylo[n_time_p1][WRFBdyVars::QV].const_array();
    const auto& bdatyhi_n   = bdy_data_yhi[n_time   ][WRFBdyVars::QV].const_array();
    const auto& bdatyhi_np1 = bdy_data_yhi[n_time_p1][WRFBdyVars::QV].const_array();

    const auto& bdatxlo_n_u   = bdy_data_xlo[n_time   ][WRFBdyVars::U].const_array();
    const auto& bdatxlo_np1_u = bdy_data_xlo[n_time_p1][WRFBdyVars::U].const_array();
    const auto& bdatxhi_n_u   = bdy_data_xhi[n_time   ][WRFBdyVars::U].const_array();
    const auto& bdatxhi_np1_u = bdy_data_xhi[n_time_p1][WRFBdyVars::U].const_array();

    const auto& bdatxlo_n_v   = bdy_data_xlo[n_time   ][WRFBdyVars::V].const_array();
    const auto& bdatxlo_np1_v = bdy_data_xlo[n_time_p1][WRFBdyVars::V].const_array();
    const auto& bdatxhi_n_v   = bdy_data_xhi[n_time   ][WRFBdyVars::V].const_array();
    const auto& bdatxhi_np1_v = bdy_data_xhi[n_time_p1][WRFBdyVars::V].const_array();

    const auto& bdatylo_n_v   = bdy_data_ylo[n_time   ][WRFBdyVars::V].const_array();
    const auto& bdatylo_np1_v = bdy_data_ylo[n_time_p1][WRFBdyVars::V].const_array();
    const auto& bdatyhi_n_v   = bdy_data_yhi[n_time   ][WRFBdyVars::V].const_array();
    const auto& bdatyhi_np1_v = bdy_data_yhi[n_time_p1][WRFBdyVars::V].const_array();

    const auto& bdatxlo_mu_n   = bdy_data_xlo[n_time   ][WRFBdyVars::MU].const_array();
    const auto& bdatxlo_mu_np1 = bdy_data_xlo[n_time_p1][WRFBdyVars::MU].const_array();
    const auto& bdatxhi_mu_n   = bdy_data_xhi[n_time   ][WRFBdyVars::MU].const_array();
    const auto& bdatxhi_mu_np1 = bdy_data_xhi[n_time_p1][WRFBdyVars::MU].const_array();
    const auto& bdatylo_mu_n   = bdy_data_ylo[n_time   ][WRFBdyVars::MU].const_array();
    const auto& bdatylo_mu_np1 = bdy_data_ylo[n_time_p1][WRFBdyVars::MU].const_array();
    const auto& bdatyhi_mu_n   = bdy_data_yhi[n_time   ][WRFBdyVars::MU].const_array();
    const auto& bdatyhi_mu_np1 = bdy_data_yhi[n_time_p1][WRFBdyVars::MU].const_array();
    const auto& bdatxlo_ph_n   = have_wrfbdy_ph ? bdy_data_xlo[n_time   ][WRFBdyVars::PH].const_array() : bdy_data_xlo[n_time][WRFBdyVars::QV].const_array();
    const auto& bdatxlo_ph_np1 = have_wrfbdy_ph ? bdy_data_xlo[n_time_p1][WRFBdyVars::PH].const_array() : bdy_data_xlo[n_time_p1][WRFBdyVars::QV].const_array();
    const auto& bdatxhi_ph_n   = have_wrfbdy_ph ? bdy_data_xhi[n_time   ][WRFBdyVars::PH].const_array() : bdy_data_xhi[n_time][WRFBdyVars::QV].const_array();
    const auto& bdatxhi_ph_np1 = have_wrfbdy_ph ? bdy_data_xhi[n_time_p1][WRFBdyVars::PH].const_array() : bdy_data_xhi[n_time_p1][WRFBdyVars::QV].const_array();
    const auto& bdatylo_ph_n   = have_wrfbdy_ph ? bdy_data_ylo[n_time   ][WRFBdyVars::PH].const_array() : bdy_data_ylo[n_time][WRFBdyVars::QV].const_array();
    const auto& bdatylo_ph_np1 = have_wrfbdy_ph ? bdy_data_ylo[n_time_p1][WRFBdyVars::PH].const_array() : bdy_data_ylo[n_time_p1][WRFBdyVars::QV].const_array();
    const auto& bdatyhi_ph_n   = have_wrfbdy_ph ? bdy_data_yhi[n_time   ][WRFBdyVars::PH].const_array() : bdy_data_yhi[n_time][WRFBdyVars::QV].const_array();
    const auto& bdatyhi_ph_np1 = have_wrfbdy_ph ? bdy_data_yhi[n_time_p1][WRFBdyVars::PH].const_array() : bdy_data_yhi[n_time_p1][WRFBdyVars::QV].const_array();
    const int kmax_ph_xlo = have_wrfbdy_ph ? bdy_data_xlo[n_time][WRFBdyVars::PH].box().bigEnd(2) : dom_hi.z;
    const int kmax_ph_xhi = have_wrfbdy_ph ? bdy_data_xhi[n_time][WRFBdyVars::PH].box().bigEnd(2) : dom_hi.z;
    const int kmax_ph_ylo = have_wrfbdy_ph ? bdy_data_ylo[n_time][WRFBdyVars::PH].box().bigEnd(2) : dom_hi.z;
    const int kmax_ph_yhi = have_wrfbdy_ph ? bdy_data_yhi[n_time][WRFBdyVars::PH].box().bigEnd(2) : dom_hi.z;
    const int kmax_qv_xlo = bdy_data_xlo[n_time][WRFBdyVars::QV].box().bigEnd(2);
    const int kmax_qv_xhi = bdy_data_xhi[n_time][WRFBdyVars::QV].box().bigEnd(2);
    const int kmax_qv_ylo = bdy_data_ylo[n_time][WRFBdyVars::QV].box().bigEnd(2);
    const int kmax_qv_yhi = bdy_data_yhi[n_time][WRFBdyVars::QV].box().bigEnd(2);

    // Get Array4 of interpolated values
    Array4<Real> arr_xlo = QV_xlo.array();  Array4<Real> arr_xhi = QV_xhi.array();
    Array4<Real> arr_ylo = QV_ylo.array();  Array4<Real> arr_yhi = QV_yhi.array();

    Array4<Real> u_xlo = U_xlo.array();  Array4<Real> u_xhi = U_xhi.array();
    Array4<Real> v_xlo = V_xlo.array();  Array4<Real> v_xhi = V_xhi.array();
    Array4<Real> v_ylo = V_ylo.array();  Array4<Real> v_yhi = V_yhi.array();

    Box gtbx = grow(tbx,ng_vect);
    Box tbx_xlo, tbx_xhi, tbx_ylo, tbx_yhi;
    realbdy_interior_bxs_xy(gtbx, domain, width,
                            tbx_xlo, tbx_xhi,
                            tbx_ylo, tbx_yhi,
                            ng_vect, true,
                            y_face_owns_corners);

    // Limiting offset
    int offset = width - 1;

    // Populate with interpolation (protect from ghost cells)
    ParallelFor(tbx_xlo, tbx_xhi,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        int ii = std::min(std::max(i , dom_lo.x), dom_lo.x+offset);
        int jj = std::min(std::max(j , dom_lo.y), dom_hi.y       );
        Real rho = new_cons(i,j,k,Rho_comp);
        Real qv_t;
        if (bdatxlo) {
            qv_t = bdatxlo(ii,jj,k,bdy_comp);
        } else {
            Real qv_base = oma * bdatxlo_n(ii,jj,k) + alpha * bdatxlo_np1(ii,jj,k);
            qv_t = qv_base;
            if (use_qv_vertical_remap) {
                // qv remap at cell centers. Modes 1/2 currently share the same operator
                // because qv is already cell-centered in horizontal coordinates.
                const int ksrc_max = amrex::min(kmax_qv_xlo, kmax_ph_xlo-1);
                const int ii_cc = amrex::min(amrex::max(ii, dom_lo.x), dom_hi.x);
                const int jj_cc = amrex::min(amrex::max(jj, dom_lo.y), dom_hi.y);
                const Real z_state = zcc_arr(i,j,k,0);
                auto z_target = [&](int kk) AMREX_GPU_DEVICE noexcept -> Real {
                    Real mu_t = oma * bdatxlo_mu_n(ii_cc,jj_cc,0,0) + alpha * bdatxlo_mu_np1(ii_cc,jj_cc,0,0) + mub_arr(ii_cc,jj_cc,0);
                    Real xmu_f = c1f_arr(0,0,kk,0) * mu_t + c2f_arr(0,0,kk,0);
                    Real ph_t = oma * bdatxlo_ph_n(ii_cc,jj_cc,kk,0) + alpha * bdatxlo_ph_np1(ii_cc,jj_cc,kk,0);
                    return (ph_t / xmu_f + phb_arr(ii_cc,jj_cc,kk,0)) / CONST_GRAV;
                };
                if (ksrc_max > 0) {
                    Real z0 = Real(0.5) * (z_target(0) + z_target(1));
                    Real q0 = oma * bdatxlo_n(ii,jj,0) + alpha * bdatxlo_np1(ii,jj,0);
                    if (z_state <= z0) {
                        qv_t = q0;
                    } else {
                        bool found = false;
                        for (int kk = 0; kk < ksrc_max; ++kk) {
                            Real zl = Real(0.5) * (z_target(kk) + z_target(kk+1));
                            Real zh = Real(0.5) * (z_target(kk+1) + z_target(kk+2));
                            if (z_state <= zh || kk == ksrc_max-1) {
                                Real ql = oma * bdatxlo_n(ii,jj,kk) + alpha * bdatxlo_np1(ii,jj,kk);
                                Real qh = oma * bdatxlo_n(ii,jj,kk+1) + alpha * bdatxlo_np1(ii,jj,kk+1);
                                Real dz = amrex::max(zh-zl, Real(1.e-12));
                                Real lam = (z_state - zl) / dz;
                                lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                qv_t = (Real(1.0)-lam)*ql + lam*qh;
                                found = true;
                                break;
                            }
                        }
                        if (!found) { qv_t = oma * bdatxlo_n(ii,jj,ksrc_max) + alpha * bdatxlo_np1(ii,jj,ksrc_max); }
                    }
                }
            }
        }
        arr_xlo(i,j,k) = rho * qv_t;
        u_xlo(i,j,k) = ( oma * bdatxlo_n_u(ii,jj,k) + alpha * bdatxlo_np1_u(ii,jj,k) );
        v_xlo(i,j,k) = ( oma * bdatxlo_n_v(ii,jj,k) + alpha * bdatxlo_np1_v(ii,jj,k) );
        if (j == dom_hi.y) {
            v_xlo(i,dom_hi.y+1,k) = ( oma   * bdatxlo_n_v  (ii,dom_hi.y+1,k)
                                    + alpha * bdatxlo_np1_v(ii,dom_hi.y+1,k) );
        }
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        int ii = std::min(std::max(i , dom_hi.x-offset), dom_hi.x);
        int jj = std::min(std::max(j , dom_lo.y       ), dom_hi.y);
        Real rho = new_cons(i,j,k,Rho_comp);
        Real qv_t;
        if (bdatxhi) {
            qv_t = bdatxhi(ii,jj,k,bdy_comp);
        } else {
            Real qv_base = oma * bdatxhi_n(ii,jj,k) + alpha * bdatxhi_np1(ii,jj,k);
            qv_t = qv_base;
            if (use_qv_vertical_remap) {
                // qv remap at cell centers. Modes 1/2 currently share the same operator.
                const int ksrc_max = amrex::min(kmax_qv_xhi, kmax_ph_xhi-1);
                const int ii_cc = amrex::min(amrex::max(ii, dom_lo.x), dom_hi.x);
                const int jj_cc = amrex::min(amrex::max(jj, dom_lo.y), dom_hi.y);
                const Real z_state = zcc_arr(i,j,k,0);
                auto z_target = [&](int kk) AMREX_GPU_DEVICE noexcept -> Real {
                    Real mu_t = oma * bdatxhi_mu_n(ii_cc,jj_cc,0,0) + alpha * bdatxhi_mu_np1(ii_cc,jj_cc,0,0) + mub_arr(ii_cc,jj_cc,0);
                    Real xmu_f = c1f_arr(0,0,kk,0) * mu_t + c2f_arr(0,0,kk,0);
                    Real ph_t = oma * bdatxhi_ph_n(ii_cc,jj_cc,kk,0) + alpha * bdatxhi_ph_np1(ii_cc,jj_cc,kk,0);
                    return (ph_t / xmu_f + phb_arr(ii_cc,jj_cc,kk,0)) / CONST_GRAV;
                };
                if (ksrc_max > 0) {
                    Real z0 = Real(0.5) * (z_target(0) + z_target(1));
                    Real q0 = oma * bdatxhi_n(ii,jj,0) + alpha * bdatxhi_np1(ii,jj,0);
                    if (z_state <= z0) {
                        qv_t = q0;
                    } else {
                        bool found = false;
                        for (int kk = 0; kk < ksrc_max; ++kk) {
                            Real zl = Real(0.5) * (z_target(kk) + z_target(kk+1));
                            Real zh = Real(0.5) * (z_target(kk+1) + z_target(kk+2));
                            if (z_state <= zh || kk == ksrc_max-1) {
                                Real ql = oma * bdatxhi_n(ii,jj,kk) + alpha * bdatxhi_np1(ii,jj,kk);
                                Real qh = oma * bdatxhi_n(ii,jj,kk+1) + alpha * bdatxhi_np1(ii,jj,kk+1);
                                Real dz = amrex::max(zh-zl, Real(1.e-12));
                                Real lam = (z_state - zl) / dz;
                                lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                qv_t = (Real(1.0)-lam)*ql + lam*qh;
                                found = true;
                                break;
                            }
                        }
                        if (!found) { qv_t = oma * bdatxhi_n(ii,jj,ksrc_max) + alpha * bdatxhi_np1(ii,jj,ksrc_max); }
                    }
                }
            }
        }
        arr_xhi(i,j,k) = rho * qv_t;
        // NOTE: correct for idx type mismatch with u bdy data
        u_xhi(i+1,j,k) = ( oma * bdatxhi_n_u(ii+1,jj,k) + alpha * bdatxhi_np1_u(ii+1,jj,k) );
        v_xhi(i  ,j,k) = ( oma * bdatxhi_n_v(ii  ,jj,k) + alpha * bdatxhi_np1_v(ii,jj,k) );
        if (j == dom_hi.y) {
            v_xhi(i,dom_hi.y+1,k) = ( oma   * bdatxhi_n_v  (ii,dom_hi.y+1,k)
                                    + alpha * bdatxhi_np1_v(ii,dom_hi.y+1,k) );
        }
    });

    ParallelFor(tbx_ylo, tbx_yhi,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        int ii = std::min(std::max(i , dom_lo.x), dom_hi.x       );
        int jj = std::min(std::max(j , dom_lo.y), dom_lo.y+offset);
        Real rho = new_cons(i,j,k,Rho_comp);
        Real qv_t;
        if (bdatylo) {
            qv_t = bdatylo(ii,jj,k,bdy_comp);
        } else {
            Real qv_base = oma * bdatylo_n(ii,jj,k) + alpha * bdatylo_np1(ii,jj,k);
            qv_t = qv_base;
            if (use_qv_vertical_remap) {
                // qv remap at cell centers. Modes 1/2 currently share the same operator.
                const int ksrc_max = amrex::min(kmax_qv_ylo, kmax_ph_ylo-1);
                const int ii_cc = amrex::min(amrex::max(ii, dom_lo.x), dom_hi.x);
                const int jj_cc = amrex::min(amrex::max(jj, dom_lo.y), dom_hi.y);
                const Real z_state = zcc_arr(i,j,k,0);
                auto z_target = [&](int kk) AMREX_GPU_DEVICE noexcept -> Real {
                    Real mu_t = oma * bdatylo_mu_n(ii_cc,jj_cc,0,0) + alpha * bdatylo_mu_np1(ii_cc,jj_cc,0,0) + mub_arr(ii_cc,jj_cc,0);
                    Real xmu_f = c1f_arr(0,0,kk,0) * mu_t + c2f_arr(0,0,kk,0);
                    Real ph_t = oma * bdatylo_ph_n(ii_cc,jj_cc,kk,0) + alpha * bdatylo_ph_np1(ii_cc,jj_cc,kk,0);
                    return (ph_t / xmu_f + phb_arr(ii_cc,jj_cc,kk,0)) / CONST_GRAV;
                };
                if (ksrc_max > 0) {
                    Real z0 = Real(0.5) * (z_target(0) + z_target(1));
                    Real q0 = oma * bdatylo_n(ii,jj,0) + alpha * bdatylo_np1(ii,jj,0);
                    if (z_state <= z0) {
                        qv_t = q0;
                    } else {
                        bool found = false;
                        for (int kk = 0; kk < ksrc_max; ++kk) {
                            Real zl = Real(0.5) * (z_target(kk) + z_target(kk+1));
                            Real zh = Real(0.5) * (z_target(kk+1) + z_target(kk+2));
                            if (z_state <= zh || kk == ksrc_max-1) {
                                Real ql = oma * bdatylo_n(ii,jj,kk) + alpha * bdatylo_np1(ii,jj,kk);
                                Real qh = oma * bdatylo_n(ii,jj,kk+1) + alpha * bdatylo_np1(ii,jj,kk+1);
                                Real dz = amrex::max(zh-zl, Real(1.e-12));
                                Real lam = (z_state - zl) / dz;
                                lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                qv_t = (Real(1.0)-lam)*ql + lam*qh;
                                found = true;
                                break;
                            }
                        }
                        if (!found) { qv_t = oma * bdatylo_n(ii,jj,ksrc_max) + alpha * bdatylo_np1(ii,jj,ksrc_max); }
                    }
                }
            }
        }
        arr_ylo(i,j,k) = rho * qv_t;
        v_ylo(i,j,k) = ( oma * bdatylo_n_v(ii,jj,k) + alpha * bdatylo_np1_v(ii,jj,k) );
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        int ii = std::min(std::max(i , dom_lo.x       ), dom_hi.x);
        int jj = std::min(std::max(j , dom_hi.y-offset), dom_hi.y);
            jj = std::min(jj, dom_hi.y);
        Real rho = new_cons(i,j,k,Rho_comp);
        Real qv_t;
        if (bdatyhi) {
            qv_t = bdatyhi(ii,jj,k,bdy_comp);
        } else {
            Real qv_base = oma * bdatyhi_n(ii,jj,k) + alpha * bdatyhi_np1(ii,jj,k);
            qv_t = qv_base;
            if (use_qv_vertical_remap) {
                // qv remap at cell centers. Modes 1/2 currently share the same operator.
                const int ksrc_max = amrex::min(kmax_qv_yhi, kmax_ph_yhi-1);
                const int ii_cc = amrex::min(amrex::max(ii, dom_lo.x), dom_hi.x);
                const int jj_cc = amrex::min(amrex::max(jj, dom_lo.y), dom_hi.y);
                const Real z_state = zcc_arr(i,j,k,0);
                auto z_target = [&](int kk) AMREX_GPU_DEVICE noexcept -> Real {
                    Real mu_t = oma * bdatyhi_mu_n(ii_cc,jj_cc,0,0) + alpha * bdatyhi_mu_np1(ii_cc,jj_cc,0,0) + mub_arr(ii_cc,jj_cc,0);
                    Real xmu_f = c1f_arr(0,0,kk,0) * mu_t + c2f_arr(0,0,kk,0);
                    Real ph_t = oma * bdatyhi_ph_n(ii_cc,jj_cc,kk,0) + alpha * bdatyhi_ph_np1(ii_cc,jj_cc,kk,0);
                    return (ph_t / xmu_f + phb_arr(ii_cc,jj_cc,kk,0)) / CONST_GRAV;
                };
                if (ksrc_max > 0) {
                    Real z0 = Real(0.5) * (z_target(0) + z_target(1));
                    Real q0 = oma * bdatyhi_n(ii,jj,0) + alpha * bdatyhi_np1(ii,jj,0);
                    if (z_state <= z0) {
                        qv_t = q0;
                    } else {
                        bool found = false;
                        for (int kk = 0; kk < ksrc_max; ++kk) {
                            Real zl = Real(0.5) * (z_target(kk) + z_target(kk+1));
                            Real zh = Real(0.5) * (z_target(kk+1) + z_target(kk+2));
                            if (z_state <= zh || kk == ksrc_max-1) {
                                Real ql = oma * bdatyhi_n(ii,jj,kk) + alpha * bdatyhi_np1(ii,jj,kk);
                                Real qh = oma * bdatyhi_n(ii,jj,kk+1) + alpha * bdatyhi_np1(ii,jj,kk+1);
                                Real dz = amrex::max(zh-zl, Real(1.e-12));
                                Real lam = (z_state - zl) / dz;
                                lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                qv_t = (Real(1.0)-lam)*ql + lam*qh;
                                found = true;
                                break;
                            }
                        }
                        if (!found) { qv_t = oma * bdatyhi_n(ii,jj,ksrc_max) + alpha * bdatyhi_np1(ii,jj,ksrc_max); }
                    }
                }
            }
        }
        arr_yhi(i,j,k) = rho * qv_t;
        // NOTE: correct for idx type mismatch with v bdy data
        v_yhi(i,j+1,k) = ( oma * bdatyhi_n_v(ii,jj+1,k) + alpha * bdatyhi_np1_v(ii,jj+1,k) );
    });


    // Compute RHS in relaxation region
    //==========================================================
    realbdy_interior_bxs_xy(tbx, domain, width,
                            tbx_xlo, tbx_xhi,
                            tbx_ylo, tbx_yhi,
                            ng_vect, false,
                            y_face_owns_corners);

    if (dbg_nudge_print_stats) {
        auto print_stats_for_side = [&](const Box& bx, const Array4<Real>& targ, const char* sname)
        {
            if (!bx.ok()) return;
            ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpMax, ReduceOpMax, ReduceOpMin, ReduceOpMax, ReduceOpSum> reduce_op;
            ReduceData<Real, Real, Real, Real, Real, Real, Long> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
            {
                Real dc  = targ(i,j,k,0) - new_cons(i,j,k,RhoQ1_comp);
                Real rho = amrex::max(new_cons(i,j,k,Rho_comp), Real(1.e-16));
                Real dp  = dc / rho;
                return {dc, dp, std::abs(dc), std::abs(dp), rho, rho, Long(1)};
            });
            auto hv = reduce_data.value();

            Real sum_dc    = amrex::get<0>(hv);
            Real sum_dp    = amrex::get<1>(hv);
            Real maxabs_dc = amrex::get<2>(hv);
            Real maxabs_dp = amrex::get<3>(hv);
            Real rho_min   = amrex::get<4>(hv);
            Real rho_max   = amrex::get<5>(hv);
            Long n         = amrex::get<6>(hv);

            ParallelDescriptor::ReduceRealSum(sum_dc);
            ParallelDescriptor::ReduceRealSum(sum_dp);
            ParallelDescriptor::ReduceRealMax(maxabs_dc);
            ParallelDescriptor::ReduceRealMax(maxabs_dp);
            ParallelDescriptor::ReduceRealMin(rho_min);
            ParallelDescriptor::ReduceRealMax(rho_max);
            ParallelDescriptor::ReduceLongSum(n);

            if (ParallelDescriptor::IOProcessor() && n > 0) {
                Print() << "[DBG_NUDGE qv]"
                        << " t=" << time
                        << " t_elapsed=" << (time-start_bdy_time)
                        << " bdy_idx=" << n_time
                        << " side=" << sname
                        << " mean_dc=" << (sum_dc/Real(n))
                        << " maxabs_dc=" << maxabs_dc
                        << " mean_dp=" << (sum_dp/Real(n))
                        << " maxabs_dp=" << maxabs_dp
                        << " rho_min=" << rho_min
                        << " rho_max=" << rho_max
                        << " n=" << n
                        << "\n";
            }
        };

        print_stats_for_side(tbx_xlo, arr_xlo, "xlo");
        print_stats_for_side(tbx_xhi, arr_xhi, "xhi");
        print_stats_for_side(tbx_ylo, arr_ylo, "ylo");
        print_stats_for_side(tbx_yhi, arr_yhi, "yhi");

        auto print_qv_band = [&](int ksel, const char* tag)
        {
            Box pbx = tbx;
            pbx.setSmall(2, ksel);
            pbx.setBig  (2, ksel);
            if (!pbx.ok()) return;

            ReduceOps<ReduceOpSum,ReduceOpMin,ReduceOpMax,ReduceOpSum,
                      ReduceOpSum,ReduceOpMin,ReduceOpMax,ReduceOpSum> rop;
            ReduceData<Real,Real,Real,Long,Real,Real,Real,Long> rdata(rop);
            using RT = typename decltype(rdata)::Type;
            rop.eval(pbx, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> RT
            {
                Real rho = amrex::max(new_cons(i,j,k,Rho_comp), Real(1.e-16));
                Real qvp = new_cons(i,j,k,RhoQ1_comp) / rho;
                bool is_bnd = (i < dom_lo.x + width) || (i > dom_hi.x - width) ||
                              (j < dom_lo.y + width) || (j > dom_hi.y - width);
                if (is_bnd) {
                    return {qvp, qvp, qvp, Long(1),
                            Real(0.0), std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max(), Long(0)};
                } else {
                    return {Real(0.0), std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max(), Long(0),
                            qvp, qvp, qvp, Long(1)};
                }
            });
            auto hv = rdata.value();
            Real sum_b = amrex::get<0>(hv);
            Real min_b = amrex::get<1>(hv);
            Real max_b = amrex::get<2>(hv);
            Long n_b   = amrex::get<3>(hv);
            Real sum_i = amrex::get<4>(hv);
            Real min_i = amrex::get<5>(hv);
            Real max_i = amrex::get<6>(hv);
            Long n_i   = amrex::get<7>(hv);

            ParallelDescriptor::ReduceRealSum(sum_b);
            ParallelDescriptor::ReduceRealMin(min_b);
            ParallelDescriptor::ReduceRealMax(max_b);
            ParallelDescriptor::ReduceLongSum(n_b);
            ParallelDescriptor::ReduceRealSum(sum_i);
            ParallelDescriptor::ReduceRealMin(min_i);
            ParallelDescriptor::ReduceRealMax(max_i);
            ParallelDescriptor::ReduceLongSum(n_i);

            if (ParallelDescriptor::IOProcessor()) {
                Print() << "[DBG_NUDGE " << tag << "]"
                        << " t=" << time
                        << " t_elapsed=" << (time-start_bdy_time)
                        << " bdy_idx=" << n_time
                        << " k=" << ksel
                        << " bnd_mean=" << (n_b > 0 ? sum_b/Real(n_b) : Real(0.0))
                        << " bnd_min=" << (n_b > 0 ? min_b : Real(0.0))
                        << " bnd_max=" << (n_b > 0 ? max_b : Real(0.0))
                        << " bnd_n=" << n_b
                        << " int_mean=" << (n_i > 0 ? sum_i/Real(n_i) : Real(0.0))
                        << " int_min=" << (n_i > 0 ? min_i : Real(0.0))
                        << " int_max=" << (n_i > 0 ? max_i : Real(0.0))
                        << " int_n=" << n_i
                        << "\n";
            }
        };
        print_qv_band(dom_lo.z, "qv_bot_band");
        print_qv_band(dom_hi.z, "qv_top_band");
    }

    realbdy_compute_relaxation(RhoQ1_comp, 1,
                               width, dx, ProbLo, ProbHi, F1, domain,
                               tbx_xlo , tbx_xhi , tbx_ylo , tbx_yhi ,
                               arr_xlo , arr_xhi , arr_ylo , arr_yhi ,
                               u_xlo, u_xhi, v_xlo, v_xhi, v_ylo, v_yhi,
                               new_cons, cell_rhs, nudge_scale, dbg_nudge_x, dbg_nudge_y,
                               dbg_nudge_exclude_x_corners, dbg_nudge_const_factor, do_upwind,
                               y_face_owns_corners, dbg_realbdy_yface_corner_use_max_metric,
                               dbg_realbdy_weight_profile_id, dbg_realbdy_weight_tanh_beta);

    /*
    // UNIT TEST DEBUG
    realbdy_interior_bxs_xy(tbx, domain, width,
                            tbx_xlo, tbx_xhi,
                            tbx_ylo, tbx_yhi);
    ParallelFor(tbx_xlo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      if (std::fabs(arr_xlo(i,j,k) - new_cons(i,j,k,RhoQ1_comp)) > Real(1.0e-7)) {
            Print() << "ERROR XLO: " <<  RhoQ1_comp << ' ' << IntVect(i,j,k) << "\n";
            exit(0);
        }
    });
    ParallelFor(tbx_xhi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      if (std::fabs(arr_xhi(i,j,k) - new_cons(i,j,k,RhoQ1_comp)) > Real(1.0e-7)) {
            Print() << "ERROR XHI: " << RhoQ1_comp<< ' ' << IntVect(i,j,k) << "\n";
            exit(0);
        }
    });
    ParallelFor(tbx_ylo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      if (std::fabs(arr_ylo(i,j,k) - new_cons(i,j,k,RhoQ1_comp))> Real(1.0e-7)) {
            Print() << "ERROR YLO: " << RhoQ1_comp << ' ' << IntVect(i,j,k) << "\n";
            exit(0);
        }
    });
    ParallelFor(tbx_yhi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      if (std::fabs(arr_yhi(i,j,k) - new_cons(i,j,k,RhoQ1_comp))> Real(1.0e-7)) {
            Print() << "ERROR YHI: " << RhoQ1_comp << ' ' << IntVect(i,j,k) << "\n";
            exit(0);
        }
    });
    exit(0);
    */
} // moist_set_rhs
#endif
