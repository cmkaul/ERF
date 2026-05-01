#include <ERF_Utils.H>

using namespace amrex;

PhysBCFunctNoOp void_bc;

/**
 * Get the boxes for looping over interior/exterior ghost cells
 * for use by fillpatch, erf_slow_rhs_pre, and erf_slow_rhs_post.
 *
 * @param[in] bx box to intersect with 4 halo regions
 * @param[in] domain box of the whole domain
 * @param[in] width number of cells in (relaxation+specified) zone
 * @param[in] set_width number of cells in (specified) zone
 * @param[out] bx_xlo halo box at x_lo boundary
 * @param[out] bx_xhi halo box at x_hi boundary
 * @param[out] bx_ylo halo box at y_lo boundary
 * @param[out] bx_yhi halo box at y_hi boundary
 * @param[in] ng_vect number of ghost cells in each direction
 * @param[in] get_int_ng flag to get ghost cells inside the domain
 */
void
realbdy_interior_bxs_xy (const Box& bx,
                         const Box& domain,
                         const int& width,
                         Box& bx_xlo,
                         Box& bx_xhi,
                         Box& bx_ylo,
                         Box& bx_yhi,
                         const IntVect& ng_vect,
                         const bool get_int_ng,
                         const bool y_face_owns_corners)
{
    AMREX_ALWAYS_ASSERT(bx.ixType() == domain.ixType());

    //==================================================================
    // NOTE: Ownership of overlapping corner region is configurable.
    //       Legacy/default behavior is x-face ownership.
    //==================================================================

    // Domain bounds without ghost cells
    const auto& dom_lo = lbound(domain);
    const auto& dom_hi = ubound(domain);

    // Four boxes matching the domain
    Box gdom_xlo(domain); Box gdom_xhi(domain);
    Box gdom_ylo(domain); Box gdom_yhi(domain);

    // Trim the boxes to only include internal ghost cells
    gdom_xlo.setBig(0,dom_lo.x+width-1); gdom_xhi.setSmall(0,dom_hi.x-width+1);
    gdom_ylo.setBig(1,dom_lo.y+width-1); gdom_yhi.setSmall(1,dom_hi.y-width+1);

    // Remove overlapping corners using selected ownership.
    if (!y_face_owns_corners) {
        // Legacy/default: x-face boxes own corners
        gdom_ylo.setSmall(0,gdom_xlo.bigEnd(0)+1); gdom_ylo.setBig(0,gdom_xhi.smallEnd(0)-1);
        gdom_yhi.setSmall(0,gdom_xlo.bigEnd(0)+1); gdom_yhi.setBig(0,gdom_xhi.smallEnd(0)-1);
    } else {
        // Debug option: y-face boxes own corners
        gdom_xlo.setSmall(1,gdom_ylo.bigEnd(1)+1); gdom_xlo.setBig(1,gdom_yhi.smallEnd(1)-1);
        gdom_xhi.setSmall(1,gdom_ylo.bigEnd(1)+1); gdom_xhi.setBig(1,gdom_yhi.smallEnd(1)-1);
    }

    // Grow boxes to get external ghost cells only
    gdom_xlo.growLo(0,ng_vect[0]); gdom_xhi.growHi(0,ng_vect[0]);
    gdom_xlo.grow  (1,ng_vect[1]); gdom_xhi.grow  (1,ng_vect[1]);
    gdom_ylo.growLo(1,ng_vect[1]); gdom_yhi.growHi(1,ng_vect[1]);

    // Grow boxes to get internal ghost cells
    if (get_int_ng) {
        gdom_xlo.growHi(0,ng_vect[0]); gdom_xhi.growLo(0,ng_vect[0]);
        gdom_ylo.grow  (0,ng_vect[0]); gdom_yhi.grow  (0,ng_vect[0]);
        gdom_ylo.growHi(1,ng_vect[1]); gdom_yhi.growLo(1,ng_vect[1]);
    }

    // Populate everything
    bx_xlo = (bx & gdom_xlo);
    bx_xhi = (bx & gdom_xhi);
    bx_ylo = (bx & gdom_ylo);
    bx_yhi = (bx & gdom_yhi);
}


/**
 * Compute the RHS in the relaxation zone
 *
 * @param[in] time              current (total) time
 * @param[in] delta_t           timestep
 * @param[in] start_bdy_time    full time of the first time slice of boundary data
 * @param[in] final_bdy_time    full time of the  last time slice of boundary data
 * @param[in] bdy_time_interval time interval between boundary condition time stamps
 * @param[in] width             number of cells in (relaxation+specified) zone
 * @param[in] set_width         number of cells in (specified) zone
 * @param[in] geom              container for geometric information
 * @param[out] S_rhs            RHS to be computed here
 * @param[in] S_data            current value of the solution
 * @param[in] bdy_data_xlo boundary data on interior of low x-face
 * @param[in] bdy_data_xhi boundary data on interior of high x-face
 * @param[in] bdy_data_ylo boundary data on interior of low y-face
 * @param[in] bdy_data_yhi boundary data on interior of high y-face
 */
void
realbdy_compute_interior_ghost_rhs (const Real& time,
                                    const Real& delta_t,
                                    const Real& start_bdy_time,
                                    const Real& final_bdy_time,
                                    const Real& bdy_time_interval,
                                    const Real& nudge_factor,
                                    int  width,
                                    bool do_upwind,
                                    const Geometry& geom,
                                    Vector<MultiFab>& S_rhs,
                                    Vector<MultiFab>& S_cur_data,
                                    Vector<Vector<FArrayBox>>& bdy_data_xlo,
                                    Vector<Vector<FArrayBox>>& bdy_data_xhi,
                                    Vector<Vector<FArrayBox>>& bdy_data_ylo,
                                    Vector<Vector<FArrayBox>>& bdy_data_yhi,
                                    std::unique_ptr<ReadBndryPlanes>& m_r2d)
{
    BL_PROFILE_REGION("realbdy_compute_interior_ghost_RHS()");

    // Debug controls for isolating real-boundary nudging behavior.
    static bool dbg_flags_init = false;
    static bool dbg_nudge_uv = true;
    static bool dbg_nudge_tq = true;
    static bool dbg_nudge_x = true;
    static bool dbg_nudge_y = true;
    static bool dbg_nudge_print_stats = false;
    static bool dbg_nudge_print_v_only = false;
    static bool dbg_tend_v_print = false;
    static bool dbg_nudge_freeze_alpha = false;
    static bool dbg_nudge_exclude_x_corners = false;
    static bool dbg_realbdy_yface_corner_owner = false;
    static bool dbg_realbdy_component_corner_owner = false;
    static bool dbg_realbdy_yface_corner_use_max_metric = true;
    static bool dbg_realbdy_use_primitive_delta = false;
    static std::string dbg_realbdy_weight_profile = "quadratic";
    static int dbg_realbdy_weight_profile_id = 0; // 0=quadratic,1=cosine,2=tanh
    static Real dbg_realbdy_weight_tanh_beta = Real(2.5);
    static bool dbg_corner_owner_print_once = false;
    static Real dbg_nudge_const_factor = Real(-1.0);
    static Real dbg_nudge_factor_uv = Real(-1.0);
    static Real dbg_nudge_factor_tq = Real(-1.0);
    if (!dbg_flags_init) {
        ParmParse pp("erf");
        pp.query("dbg_nudge_uv", dbg_nudge_uv);
        pp.query("dbg_nudge_tq", dbg_nudge_tq);
        pp.query("dbg_nudge_x",  dbg_nudge_x);
        pp.query("dbg_nudge_y",  dbg_nudge_y);
        pp.query("dbg_nudge_print_stats", dbg_nudge_print_stats);
        pp.query("dbg_nudge_print_v_only", dbg_nudge_print_v_only);
        pp.query("dbg_tend_v_print", dbg_tend_v_print);
        pp.query("dbg_nudge_freeze_alpha", dbg_nudge_freeze_alpha);
        pp.query("dbg_nudge_exclude_x_corners", dbg_nudge_exclude_x_corners);
        pp.query("dbg_realbdy_yface_corner_owner", dbg_realbdy_yface_corner_owner);
        pp.query("dbg_realbdy_component_corner_owner", dbg_realbdy_component_corner_owner);
        pp.query("dbg_realbdy_yface_corner_use_max_metric", dbg_realbdy_yface_corner_use_max_metric);
        pp.query("dbg_realbdy_use_primitive_delta", dbg_realbdy_use_primitive_delta);
        pp.query("dbg_realbdy_weight_profile", dbg_realbdy_weight_profile);
        pp.query("dbg_realbdy_weight_tanh_beta", dbg_realbdy_weight_tanh_beta);
        pp.query("dbg_nudge_const_factor", dbg_nudge_const_factor);
        pp.query("dbg_nudge_factor_uv", dbg_nudge_factor_uv);
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
        dbg_flags_init = true;
    }

    if (!dbg_corner_owner_print_once && ParallelDescriptor::IOProcessor()) {
        if (dbg_realbdy_component_corner_owner) {
            Print() << "[DBG_CORNER_OWNER] mode=component-aware "
                    << "U=x-owner V=y-owner T=x-owner "
                    << "(dbg_realbdy_component_corner_owner=1, "
                    << "dbg_realbdy_yface_corner_owner=" << dbg_realbdy_yface_corner_owner << ", "
                    << "dbg_realbdy_yface_corner_use_max_metric=" << dbg_realbdy_yface_corner_use_max_metric << ", "
                    << "dbg_realbdy_weight_profile='" << dbg_realbdy_weight_profile << "', "
                    << "dbg_realbdy_weight_tanh_beta=" << dbg_realbdy_weight_tanh_beta
                    << ")\n";
        } else {
            Print() << "[DBG_CORNER_OWNER] mode=global "
                    << (dbg_realbdy_yface_corner_owner ? "all=y-owner" : "all=x-owner")
                    << " (dbg_realbdy_component_corner_owner=0, "
                    << "dbg_realbdy_yface_corner_use_max_metric=" << dbg_realbdy_yface_corner_use_max_metric << ", "
                    << "dbg_realbdy_weight_profile='" << dbg_realbdy_weight_profile << "', "
                    << "dbg_realbdy_weight_tanh_beta=" << dbg_realbdy_weight_tanh_beta
                    << ")\n";
        }
        dbg_corner_owner_print_once = true;
    }

    // HACK HACK HACK
    // Get bndry data
    Vector<int> ind_map2 = {BCVars::xvel_bc, BCVars::yvel_bc, BCVars::RhoTheta_bc_comp};
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
    Real F1 = one/(nudge_factor*delta_t);

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

    /*
    // UNIT TEST DEBUG
    oma = one; alpha = zero;
    */

    // Temporary FABs for storage (owned/filled on all ranks)
    FArrayBox U_xlo, U_xhi, U_ylo, U_yhi;
    FArrayBox V_xlo, V_xhi, V_ylo, V_yhi;
    FArrayBox T_xlo, T_xhi, T_ylo, T_yhi;

    // Variable index map (WRFBdyVars -> Vars)
    Vector<int> var_map  = {Vars::xvel,    Vars::yvel,    Vars::cons,    Vars::cons};
    Vector<int> ivar_map = {IntVars::xmom, IntVars::ymom, IntVars::cons, IntVars::cons};

    // Variable icomp map
    Vector<int> comp_map = {0, 0, RhoTheta_comp};

    // Indices
    int  ivarU = RealBdyVars::U;
    int  ivarV = RealBdyVars::V;
    int  ivarT = RealBdyVars::T;
    int BdyEnd = RealBdyVars::NumTypes-1;


    // NOTE: The sizing of the temporary BDY FABS is
    //       GLOBAL and occurs over the entire BDY region.

    // Size the FABs
    //==========================================================
    for (int ivar(ivarU); ivar < BdyEnd; ivar++) {
        bool y_face_owns_corners = dbg_realbdy_yface_corner_owner;
        if (dbg_realbdy_component_corner_owner) {
            // Component-aware corner ownership:
            // u -> x-face ownership, v -> y-face ownership, scalars -> x-face ownership.
            y_face_owns_corners = (ivar == ivarV);
        }

        int ivar_idx = var_map[ivar];
        Box domain   = geom.Domain();
        auto ixtype  = S_cur_data[ivar_idx].boxArray().ixType();
        domain.convert(ixtype);

        // NOTE: Ghost cells needed for idx type mismatch between mask and data (do_upwind)
        IntVect ng_vect(0);
        //IntVect ng_vect(1,1,0);
        Box gdom(domain); gdom.grow(ng_vect);
        Box bx_xlo, bx_xhi, bx_ylo, bx_yhi;
        realbdy_interior_bxs_xy(gdom, domain, width,
                                bx_xlo, bx_xhi,
                                bx_ylo, bx_yhi,
                                ng_vect, true,
                                y_face_owns_corners);

        // Size the FABs
        if (ivar  == ivarU) {
            U_xlo.resize(bx_xlo,1,The_Async_Arena()); U_xhi.resize(bx_xhi,1,The_Async_Arena());
            U_ylo.resize(bx_ylo,1,The_Async_Arena()); U_yhi.resize(bx_yhi,1,The_Async_Arena());
        } else if (ivar  == ivarV) {
            V_xlo.resize(bx_xlo,1,The_Async_Arena()); V_xhi.resize(bx_xhi,1,The_Async_Arena());
            V_ylo.resize(bx_ylo,1,The_Async_Arena()); V_yhi.resize(bx_yhi,1,The_Async_Arena());
        } else if (ivar  == ivarT){
            T_xlo.resize(bx_xlo,1,The_Async_Arena()); T_xhi.resize(bx_xhi,1,The_Async_Arena());
            T_ylo.resize(bx_ylo,1,The_Async_Arena()); T_yhi.resize(bx_yhi,1,The_Async_Arena());
        } else {
            continue;
        }
    } // ivar


    // NOTE: These operations use the BDY FABS and RHO. The
    //       use of RHO to go from PRIM -> CONS requires that
    //       these operations be LOCAL. So we have allocated
    //       enough space to do global operations (1 rank) but
    //       will fill a subset of that data that the rank owns.

    // Populate FABs from bdy interpolation (primitive vars)
    //==========================================================
    for (int ivar(ivarU); ivar < BdyEnd; ivar++) {
        bool y_face_owns_corners = dbg_realbdy_yface_corner_owner;
        if (dbg_realbdy_component_corner_owner) {
            y_face_owns_corners = (ivar == ivarV);
        }

        int ivar_idx = var_map[ivar];
        Box domain   = geom.Domain();
        auto ixtype  = S_cur_data[ivar_idx].boxArray().ixType();
        domain.convert(ixtype);
        const auto& dom_lo = lbound(domain);
        const auto& dom_hi = ubound(domain);

        // BndryReg idx and limiting
        int bdy_comp = ind_map2[ivar];
        const auto& dom_cc_lo = lbound(geom.Domain());
        const auto& dom_cc_hi = ubound(geom.Domain());

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(S_cur_data[ivar_idx],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            // NOTE: Ghost cells needed for idx type mismatch between mask and data (do_upwind)
            IntVect ng_vect(0);
            //IntVect ng_vect(1,1,0);
            Box gtbx = grow(mfi.tilebox(ixtype.toIntVect()),ng_vect);
            Box tbx_xlo, tbx_xhi, tbx_ylo, tbx_yhi;
            realbdy_interior_bxs_xy(gtbx, domain, width,
                                    tbx_xlo, tbx_xhi,
                                    tbx_ylo, tbx_yhi,
                                    ng_vect, true,
                                    y_face_owns_corners);

            Array4<Real> arr_xlo;  Array4<Real> arr_xhi;
            Array4<Real> arr_ylo;  Array4<Real> arr_yhi;
            if (ivar  == ivarU) {
                arr_xlo = U_xlo.array(); arr_xhi = U_xhi.array();
                arr_ylo = U_ylo.array(); arr_yhi = U_yhi.array();
            } else if (ivar  == ivarV) {
                arr_xlo = V_xlo.array(); arr_xhi = V_xhi.array();
                arr_ylo = V_ylo.array(); arr_yhi = V_yhi.array();
            } else if (ivar  == ivarT){
                arr_xlo = T_xlo.array(); arr_xhi = T_xhi.array();
                arr_ylo = T_ylo.array(); arr_yhi = T_yhi.array();
            } else {
                continue;
            }

            // Boundary data at fixed time intervals
            const auto& bdatxlo_n   = bdy_data_xlo[n_time   ][ivar].const_array();
            const auto& bdatxlo_np1 = bdy_data_xlo[n_time_p1][ivar].const_array();
            const auto& bdatxhi_n   = bdy_data_xhi[n_time   ][ivar].const_array();
            const auto& bdatxhi_np1 = bdy_data_xhi[n_time_p1][ivar].const_array();
            const auto& bdatylo_n   = bdy_data_ylo[n_time   ][ivar].const_array();
            const auto& bdatylo_np1 = bdy_data_ylo[n_time_p1][ivar].const_array();
            const auto& bdatyhi_n   = bdy_data_yhi[n_time   ][ivar].const_array();
            const auto& bdatyhi_np1 = bdy_data_yhi[n_time_p1][ivar].const_array();

            // Current density to convert to conserved vars
            Array4<Real> r_arr = S_cur_data[IntVars::cons].array(mfi);

            // Limiting offset
            int offset = width - 1;

            // Populate with interpolation (protect from ghost cells)
            ParallelFor(tbx_xlo, tbx_xhi,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int ii = std::max(i , dom_lo.x);
                    ii = std::min(ii, dom_lo.x+offset);
                int jj = std::max(j , dom_lo.y);
                    jj = std::min(jj, dom_hi.y);

                Real rho_interp;
                if (ivar==ivarU) {
                    rho_interp = myhalf * ( r_arr(i-1,j  ,k) + r_arr(i,j,k) );
                } else if (ivar==ivarV) {
                    rho_interp = myhalf * ( r_arr(i  ,j-1,k) + r_arr(i,j,k) );
                } else {
                    rho_interp = r_arr(i,j,k);
                }

                if (bdatxlo) {
                    int ii2 = std::min(std::max(i , dom_cc_lo.x), dom_cc_hi.x);
                    int jj2 = std::min(std::max(j , dom_cc_lo.y), dom_cc_hi.y);
                    arr_xlo(i,j,k) = rho_interp * bdatxlo(ii2,jj2,k,bdy_comp);
                } else {
                    arr_xlo(i,j,k) = rho_interp * ( oma   * bdatxlo_n  (ii,jj,k,0)
                                                  + alpha * bdatxlo_np1(ii,jj,k,0) );
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int ii = std::max(i , dom_hi.x-offset);
                    ii = std::min(ii, dom_hi.x);
                int jj = std::max(j , dom_lo.y);
                    jj = std::min(jj, dom_hi.y);

                Real rho_interp;
                if (ivar==ivarU) {
                    rho_interp = myhalf * ( r_arr(i-1,j  ,k) + r_arr(i,j,k) );
                } else if (ivar==ivarV) {
                    rho_interp = myhalf * ( r_arr(i  ,j-1,k) + r_arr(i,j,k) );
                } else {
                    rho_interp = r_arr(i,j,k);
                }

                if (bdatxhi) {
                    int ii2 = std::min(std::max(i , dom_cc_lo.x), dom_cc_hi.x);
                    int jj2 = std::min(std::max(j , dom_cc_lo.y), dom_cc_hi.y);
                    arr_xhi(i,j,k) = rho_interp * bdatxhi(ii2,jj2,k,bdy_comp);
                } else {
                    arr_xhi(i,j,k) = rho_interp * ( oma   * bdatxhi_n  (ii,jj,k,0)
                                                  + alpha * bdatxhi_np1(ii,jj,k,0) );
                }
            });

            ParallelFor(tbx_ylo, tbx_yhi,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int ii = std::max(i , dom_lo.x);
                    ii = std::min(ii, dom_hi.x);
                int jj = std::max(j , dom_lo.y);
                    jj = std::min(jj, dom_lo.y+offset);

                Real rho_interp;
                if (ivar==ivarU) {
                    rho_interp = myhalf * ( r_arr(i-1,j  ,k) + r_arr(i,j,k) );
                } else if (ivar==ivarV) {
                    rho_interp = myhalf * ( r_arr(i  ,j-1,k) + r_arr(i,j,k) );
                } else {
                    rho_interp = r_arr(i,j,k);
                }

                if (bdatylo) {
                    int ii2 = std::min(std::max(i , dom_cc_lo.x), dom_cc_hi.x);
                    int jj2 = std::min(std::max(j , dom_cc_lo.y), dom_cc_hi.y);
                    arr_ylo(i,j,k) = rho_interp * bdatylo(ii2,jj2,k,bdy_comp);
                } else {
                    arr_ylo(i,j,k) = rho_interp * ( oma  * bdatylo_n  (ii,jj,k,0)
                                                  + alpha * bdatylo_np1(ii,jj,k,0) );
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int ii = std::max(i , dom_lo.x);
                    ii = std::min(ii, dom_hi.x);
                int jj = std::max(j , dom_hi.y-offset);
                    jj = std::min(jj, dom_hi.y);

                Real rho_interp;
                if (ivar==ivarU) {
                    rho_interp = myhalf * ( r_arr(i-1,j  ,k) + r_arr(i,j,k) );
                } else if (ivar==ivarV) {
                    rho_interp = myhalf * ( r_arr(i  ,j-1,k) + r_arr(i,j,k) );
                } else {
                    rho_interp = r_arr(i,j,k);
                }

                if (bdatyhi) {
                    int ii2 = std::min(std::max(i , dom_cc_lo.x), dom_cc_hi.x);
                    int jj2 = std::min(std::max(j , dom_cc_lo.y), dom_cc_hi.y);
                    arr_yhi(i,j,k) = rho_interp * bdatyhi(ii2,jj2,k,bdy_comp);
                } else {
                    arr_yhi(i,j,k) = rho_interp * ( oma   * bdatyhi_n  (ii,jj,k,0)
                                                  + alpha * bdatyhi_np1(ii,jj,k,0) );
                }
            });
        } // mfi
    } // ivar


    // Compute RHS in relaxation region
    //==========================================================
    auto dx = geom.CellSizeArray();
    auto ProbLo = geom.ProbLoArray();
    auto ProbHi = geom.ProbHiArray();
    for (int ivar(ivarU); ivar < BdyEnd; ivar++) {
        bool y_face_owns_corners = dbg_realbdy_yface_corner_owner;
        if (dbg_realbdy_component_corner_owner) {
            y_face_owns_corners = (ivar == ivarV);
        }

        int ivar_idx = ivar_map[ivar];
        int icomp    = comp_map[ivar];
        Real nudge_scale = Real(1.0);
        if ((ivar == ivarU || ivar == ivarV) && !dbg_nudge_uv) { nudge_scale = Real(0.0); }
        if (ivar == ivarT && !dbg_nudge_tq) { nudge_scale = Real(0.0); }
        constexpr int NSIDES = 4;
        Real best_absdp[NSIDES] = {-1.0, -1.0, -1.0, -1.0};
        int best_i[NSIDES] = {0,0,0,0};
        int best_j[NSIDES] = {0,0,0,0};
        int best_k[NSIDES] = {0,0,0,0};
        Real best_dc[NSIDES] = {0.0,0.0,0.0,0.0};
        Real best_dp[NSIDES] = {0.0,0.0,0.0,0.0};
        Real best_rho[NSIDES] = {0.0,0.0,0.0,0.0};
        Real best_statep[NSIDES] = {0.0,0.0,0.0,0.0};
        Real best_targp[NSIDES] = {0.0,0.0,0.0,0.0};
        Real sum_top_bnd = 0.0, min_top_bnd = std::numeric_limits<Real>::max(), max_top_bnd = -std::numeric_limits<Real>::max();
        Real sum_top_int = 0.0, min_top_int = std::numeric_limits<Real>::max(), max_top_int = -std::numeric_limits<Real>::max();
        Long n_top_bnd = 0, n_top_int = 0;
        Real sum_bot_bnd = 0.0, min_bot_bnd = std::numeric_limits<Real>::max(), max_bot_bnd = -std::numeric_limits<Real>::max();
        Real sum_bot_int = 0.0, min_bot_int = std::numeric_limits<Real>::max(), max_bot_int = -std::numeric_limits<Real>::max();
        Long n_bot_bnd = 0, n_bot_int = 0;

        Box domain = geom.Domain();
        domain.convert(S_cur_data[ivar_idx].boxArray().ixType());
        IntVect ng_vect(0);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(S_cur_data[ivar_idx],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box tbx = mfi.tilebox();
            Box tbx_xlo, tbx_xhi, tbx_ylo, tbx_yhi;
            realbdy_interior_bxs_xy(tbx, domain, width,
                                    tbx_xlo, tbx_xhi,
                                    tbx_ylo, tbx_yhi,
                                    ng_vect, false,
                                    y_face_owns_corners);

            Array4<Real> rhs_arr; Array4<Real> data_arr;
            Array4<Real> rho_cc_arr;
            Array4<Real> arr_xlo;  Array4<Real> arr_xhi;
            Array4<Real> arr_ylo;  Array4<Real> arr_yhi;
            if (ivar  == ivarU) {
                arr_xlo  = U_xlo.array(); arr_xhi = U_xhi.array();
                arr_ylo  = U_ylo.array(); arr_yhi = U_yhi.array();
                rhs_arr  = S_rhs[IntVars::xmom].array(mfi);
                data_arr = S_cur_data[IntVars::xmom].array(mfi);
                rho_cc_arr = S_cur_data[IntVars::cons].array(mfi);
            } else if (ivar  == ivarV) {
                arr_xlo  = V_xlo.array(); arr_xhi = V_xhi.array();
                arr_ylo  = V_ylo.array(); arr_yhi = V_yhi.array();
                rhs_arr  = S_rhs[IntVars::ymom].array(mfi);
                data_arr = S_cur_data[IntVars::ymom].array(mfi);
                rho_cc_arr = S_cur_data[IntVars::cons].array(mfi);
            } else if (ivar  == ivarT){
                arr_xlo  = T_xlo.array(); arr_xhi = T_xhi.array();
                arr_ylo  = T_ylo.array(); arr_yhi = T_yhi.array();
                rhs_arr  = S_rhs[IntVars::cons].array(mfi);
                data_arr = S_cur_data[IntVars::cons].array(mfi);
                rho_cc_arr = S_cur_data[IntVars::cons].array(mfi);
            } else {
                continue;
            }

            Array4<Real> u_xlo = U_xlo.array(); Array4<Real> u_xhi = U_xhi.array();
            Array4<Real> v_xlo = V_xlo.array(); Array4<Real> v_xhi = V_xhi.array();
            Array4<Real> v_ylo = V_ylo.array(); Array4<Real> v_yhi = V_yhi.array();

            if ((dbg_nudge_print_stats &&
                 (ivar == ivarV || (ivar == ivarT && !dbg_nudge_print_v_only))) ||
                (dbg_tend_v_print && ivar == ivarV)) {
                const auto& rho_arr = S_cur_data[IntVars::cons].const_array(mfi);
                const auto dom3 = lbound(geom.Domain());
                const auto domh = ubound(geom.Domain());
                const int ktop = domh.z;
                const char* dbg_var = (ivar == ivarT) ? "theta" : "v";

                auto scan_box_for_location = [&](const Box& bx, const Array4<Real>& targ, int iside)
                {
                    if (!bx.ok()) return;
                    const auto lo = lbound(bx);
                    const auto hi = ubound(bx);
                    for (int k = lo.z; k <= hi.z; ++k) {
                        for (int j = lo.y; j <= hi.y; ++j) {
                            for (int i = lo.x; i <= hi.x; ++i) {
                                int ic = std::min(std::max(i, dom3.x), domh.x);
                                int jc = std::min(std::max(j, dom3.y), domh.y);
                                Real rho;
                                if (ivar == ivarV) {
                                    int jcm1 = std::min(std::max(j-1, dom3.y), domh.y);
                                    rho = amrex::max(myhalf * (rho_arr(ic,jcm1,k,Rho_comp) + rho_arr(ic,jc,k,Rho_comp)), Real(1.e-16));
                                } else {
                                    rho = amrex::max(rho_arr(ic,jc,k,Rho_comp), Real(1.e-16));
                                }
                                Real dc  = targ(i,j,k,0) - data_arr(i,j,k,icomp);
                                Real dp  = dc / rho;
                                Real adp = std::abs(dp);
                                if (adp > best_absdp[iside]) {
                                    best_absdp[iside] = adp;
                                    best_i[iside] = i;
                                    best_j[iside] = j;
                                    best_k[iside] = k;
                                    best_dc[iside] = dc;
                                    best_dp[iside] = dp;
                                    best_rho[iside] = rho;
                                    best_statep[iside] = data_arr(i,j,k,icomp) / rho;
                                    best_targp[iside] = targ(i,j,k,0) / rho;
                                }
                            }
                        }
                    }
                };
                auto print_stats_for_side = [&](const Box& bx, const Array4<Real>& targ, const char* sname)
                {
                    if (!bx.ok()) return;
                    ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpMax, ReduceOpMax,
                              ReduceOpMin, ReduceOpMax, ReduceOpSum,
                              ReduceOpSum, ReduceOpSum, ReduceOpMax, ReduceOpMax> reduce_op;
                    ReduceData<Real, Real, Real, Real, Real, Real, Long,
                               Real, Real, Real, Real> reduce_data(reduce_op);
                    using ReduceTuple = typename decltype(reduce_data)::Type;
                    reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
                    {
                        Real dc  = targ(i,j,k,0) - data_arr(i,j,k,icomp);
                        int ic = amrex::min(amrex::max(i, dom3.x), domh.x);
                        int jc = amrex::min(amrex::max(j, dom3.y), domh.y);
                        Real rho;
                        if (ivar == ivarV) {
                            int jcm1 = amrex::min(amrex::max(j-1, dom3.y), domh.y);
                            rho = amrex::max(myhalf * (rho_arr(ic,jcm1,k,Rho_comp) + rho_arr(ic,jc,k,Rho_comp)), Real(1.e-16));
                        } else {
                            rho = amrex::max(rho_arr(ic,jc,k,Rho_comp), Real(1.e-16));
                        }
                        Real sp  = data_arr(i,j,k,icomp) / rho;
                        Real tp  = targ(i,j,k,0) / rho;
                        Real dp  = dc / rho;
                        return {dc, dp, std::abs(dc), std::abs(dp), rho, rho, Long(1),
                                sp, tp, std::abs(sp), std::abs(tp)};
                    });
                    auto hv = reduce_data.value();

                    Real sum_dc     = amrex::get<0>(hv);
                    Real sum_dp     = amrex::get<1>(hv);
                    Real maxabs_dc  = amrex::get<2>(hv);
                    Real maxabs_dp  = amrex::get<3>(hv);
                    Real rho_min    = amrex::get<4>(hv);
                    Real rho_max    = amrex::get<5>(hv);
                    Long n          = amrex::get<6>(hv);
                    Real sum_statep = amrex::get<7>(hv);
                    Real sum_targp  = amrex::get<8>(hv);
                    Real maxabs_sp  = amrex::get<9>(hv);
                    Real maxabs_tp  = amrex::get<10>(hv);

                    if (ivar == ivarV) {
                        const int my_rank = ParallelDescriptor::MyProc();
                        if (n > 0) {
                            amrex::AllPrint() << "[DBG_NUDGE " << dbg_var << "]"
                                              << " rank=" << my_rank
                                              << " t=" << time
                                              << " t_elapsed=" << (time-start_bdy_time)
                                              << " bdy_idx=" << n_time
                                              << " side=" << sname
                                              << " mean_statep=" << (sum_statep/Real(n))
                                              << " mean_targp=" << (sum_targp/Real(n))
                                              << " maxabs_statep=" << maxabs_sp
                                              << " maxabs_targp=" << maxabs_tp
                                              << " mean_dc=" << (sum_dc/Real(n))
                                              << " maxabs_dc=" << maxabs_dc
                                              << " mean_dp=" << (sum_dp/Real(n))
                                              << " maxabs_dp=" << maxabs_dp
                                              << " rho_min=" << rho_min
                                              << " rho_max=" << rho_max
                                              << " n=" << n
                                              << "\n";
                        }
                        return;
                    }

                    ParallelDescriptor::ReduceRealSum(sum_dc);
                    ParallelDescriptor::ReduceRealSum(sum_dp);
                    ParallelDescriptor::ReduceRealMax(maxabs_dc);
                    ParallelDescriptor::ReduceRealMax(maxabs_dp);
                    ParallelDescriptor::ReduceRealMin(rho_min);
                    ParallelDescriptor::ReduceRealMax(rho_max);
                    ParallelDescriptor::ReduceLongSum(n);
                    ParallelDescriptor::ReduceRealSum(sum_statep);
                    ParallelDescriptor::ReduceRealSum(sum_targp);
                    ParallelDescriptor::ReduceRealMax(maxabs_sp);
                    ParallelDescriptor::ReduceRealMax(maxabs_tp);

                    if (ParallelDescriptor::IOProcessor() && n > 0) {
                        Print() << "[DBG_NUDGE " << dbg_var << "]"
                                << " t=" << time
                                << " t_elapsed=" << (time-start_bdy_time)
                                << " bdy_idx=" << n_time
                                << " side=" << sname
                                << " mean_statep=" << (sum_statep/Real(n))
                                << " mean_targp=" << (sum_targp/Real(n))
                                << " maxabs_statep=" << maxabs_sp
                                << " maxabs_targp=" << maxabs_tp
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

                auto print_xside_corner_split = [&](const Box& bx, const Array4<Real>& targ, const char* side_tag)
                {
                    if (!bx.ok()) return;
                    ReduceOps<ReduceOpSum, ReduceOpMax, ReduceOpSum,
                              ReduceOpSum, ReduceOpMax, ReduceOpSum> rop;
                    ReduceData<Real, Real, Long, Real, Real, Long> rdata(rop);
                    using RT = typename decltype(rdata)::Type;
                    rop.eval(bx, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> RT
                    {
                        int ic = amrex::min(amrex::max(i, dom3.x), domh.x);
                        int jc = amrex::min(amrex::max(j, dom3.y), domh.y);
                        Real rho;
                        if (ivar == ivarV) {
                            int jcm1 = amrex::min(amrex::max(j-1, dom3.y), domh.y);
                            rho = amrex::max(myhalf * (rho_arr(ic,jcm1,k,Rho_comp) + rho_arr(ic,jc,k,Rho_comp)), Real(1.e-16));
                        } else {
                            rho = amrex::max(rho_arr(ic,jc,k,Rho_comp), Real(1.e-16));
                        }
                        Real dp  = (targ(i,j,k,0) - data_arr(i,j,k,icomp)) / rho;
                        bool is_corner = (j < dom3.y + width) || (j > domh.y - width);
                        if (is_corner) {
                            return {dp, std::abs(dp), Long(1), Real(0.0), Real(0.0), Long(0)};
                        } else {
                            return {Real(0.0), Real(0.0), Long(0), dp, std::abs(dp), Long(1)};
                        }
                    });
                    auto hv = rdata.value();
                    Real sum_dp_c    = amrex::get<0>(hv);
                    Real maxabs_dp_c = amrex::get<1>(hv);
                    Long n_c         = amrex::get<2>(hv);
                    Real sum_dp_e    = amrex::get<3>(hv);
                    Real maxabs_dp_e = amrex::get<4>(hv);
                    Long n_e         = amrex::get<5>(hv);
                    if (n_c > 0 || n_e > 0) {
                        const int my_rank = ParallelDescriptor::MyProc();
                        amrex::AllPrint() << "[DBG_NUDGE " << dbg_var << "_" << side_tag << "_corner_split]"
                                          << " rank=" << my_rank
                                          << " t=" << time
                                          << " t_elapsed=" << (time-start_bdy_time)
                                          << " bdy_idx=" << n_time
                                          << " width=" << width
                                          << " corner_mean_dp=" << (n_c > 0 ? sum_dp_c/Real(n_c) : Real(0.0))
                                          << " corner_maxabs_dp=" << (n_c > 0 ? maxabs_dp_c : Real(0.0))
                                          << " corner_n=" << n_c
                                          << " edge_mean_dp=" << (n_e > 0 ? sum_dp_e/Real(n_e) : Real(0.0))
                                          << " edge_maxabs_dp=" << (n_e > 0 ? maxabs_dp_e : Real(0.0))
                                          << " edge_n=" << n_e
                                          << "\n";
                    }
                };

                auto print_xside_setrelax_split = [&](const Box& bx, const Array4<Real>& targ, const bool is_hi, const char* side_tag)
                {
                    if (!bx.ok()) return;
                    ReduceOps<ReduceOpSum, ReduceOpMax, ReduceOpSum,
                              ReduceOpSum, ReduceOpMax, ReduceOpSum> rop;
                    ReduceData<Real, Real, Long, Real, Real, Long> rdata(rop);
                    using RT = typename decltype(rdata)::Type;
                    rop.eval(bx, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> RT
                    {
                        int ic = amrex::min(amrex::max(i, dom3.x), domh.x);
                        int jc = amrex::min(amrex::max(j, dom3.y), domh.y);
                        Real rho;
                        if (ivar == ivarV) {
                            int jcm1 = amrex::min(amrex::max(j-1, dom3.y), domh.y);
                            rho = amrex::max(myhalf * (rho_arr(ic,jcm1,k,Rho_comp) + rho_arr(ic,jc,k,Rho_comp)), Real(1.e-16));
                        } else {
                            rho = amrex::max(rho_arr(ic,jc,k,Rho_comp), Real(1.e-16));
                        }
                        Real dp = (targ(i,j,k,0) - data_arr(i,j,k,icomp)) / rho;
                        int dist = is_hi ? (domh.x - i) : (i - dom3.x);
                        bool is_set = (dist == 0);
                        if (is_set) {
                            return {dp, std::abs(dp), Long(1), Real(0.0), Real(0.0), Long(0)};
                        } else {
                            return {Real(0.0), Real(0.0), Long(0), dp, std::abs(dp), Long(1)};
                        }
                    });
                    auto hv = rdata.value();
                    Real sum_dp_set    = amrex::get<0>(hv);
                    Real maxabs_dp_set = amrex::get<1>(hv);
                    Long n_set         = amrex::get<2>(hv);
                    Real sum_dp_relax    = amrex::get<3>(hv);
                    Real maxabs_dp_relax = amrex::get<4>(hv);
                    Long n_relax         = amrex::get<5>(hv);
                    if (n_set > 0 || n_relax > 0) {
                        const int my_rank = ParallelDescriptor::MyProc();
                        amrex::AllPrint() << "[DBG_NUDGE " << dbg_var << "_" << side_tag << "_setrelax_split]"
                                          << " rank=" << my_rank
                                          << " t=" << time
                                          << " t_elapsed=" << (time-start_bdy_time)
                                          << " bdy_idx=" << n_time
                                          << " width=" << width
                                          << " set_mean_dp=" << (n_set > 0 ? sum_dp_set/Real(n_set) : Real(0.0))
                                          << " set_maxabs_dp=" << (n_set > 0 ? maxabs_dp_set : Real(0.0))
                                          << " set_n=" << n_set
                                          << " relax_mean_dp=" << (n_relax > 0 ? sum_dp_relax/Real(n_relax) : Real(0.0))
                                          << " relax_maxabs_dp=" << (n_relax > 0 ? maxabs_dp_relax : Real(0.0))
                                          << " relax_n=" << n_relax
                                          << "\n";
                    }
                };

                auto print_tend_for_side = [&](const Box& bx, const Array4<Real>& targ, const char* sname)
                {
                    if (!bx.ok()) return;
                    const bool dbg_nudge_exclude_x_corners_l = dbg_nudge_exclude_x_corners;
                    const Real dbg_nudge_const_factor_l = dbg_nudge_const_factor;
                    const auto iv = bx.type();
                    const Real ioff = (iv[0] == 1) ? zero : myhalf;
                    const Real joff = (iv[1] == 1) ? zero : myhalf;
                    const auto dom_cc_lo = lbound(geom.Domain());
                    const auto dom_cc_hi = ubound(geom.Domain());

                    ReduceOps<ReduceOpSum, ReduceOpMax, ReduceOpSum,
                              ReduceOpSum, ReduceOpMax, ReduceOpSum> rop;
                    ReduceData<Real, Real, Long, Real, Real, Long> rdata(rop);
                    using RT = typename decltype(rdata)::Type;
                    rop.eval(bx, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> RT
                    {
                        int ic = amrex::min(amrex::max(i, dom3.x), domh.x);
                        int jc = amrex::min(amrex::max(j, dom3.y), domh.y);
                        Real rho;
                        if (ivar == ivarV) {
                            int jcm1 = amrex::min(amrex::max(j-1, dom3.y), domh.y);
                            rho = amrex::max(myhalf * (rho_arr(ic,jcm1,k,Rho_comp) + rho_arr(ic,jc,k,Rho_comp)), Real(1.e-16));
                        } else {
                            rho = amrex::max(rho_arr(ic,jc,k,Rho_comp), Real(1.e-16));
                        }

                        Real delta = targ(i,j,k,0) - data_arr(i,j,k,icomp);
                        Real Factor = Real(0.0);
                        bool apply = true;

                        if (sname[0] == 'x') {
                            Real x = ProbLo[0] + (i + ioff) * dx[0];
                            Real y = ProbLo[1] + (j + joff) * dx[1];
                            Real y_end  = ProbLo[1] + width * dx[1];
                            Real y_strt = ProbHi[1] - width * dx[1];
                            Real eta_lo = (y < y_end ) ? (y_end  - y) / (y_end  - ProbLo[1]) : zero;
                            Real eta_hi = (y > y_strt) ? (y - y_strt) / (ProbHi[1] - y_strt) : zero;
                            Real eta    = amrex::max(eta_lo,eta_hi);
                            if (dbg_nudge_exclude_x_corners_l && eta > zero) { apply = false; }

                            if (sname[2] == 'o') { // xlo
                                Real x_end = ProbLo[0] + width * dx[0];
                                Real xi = (x_end - x) / (x_end - ProbLo[0]);
                                Factor = amrex::max(xi*xi, eta*eta);
                                if (dbg_nudge_const_factor_l >= zero) { Factor = dbg_nudge_const_factor_l; }
                                if (do_upwind) {
                                    int jju = amrex::min(amrex::max(j,dom_cc_lo.y),dom_cc_hi.y);
                                    int iiv = amrex::min(amrex::max(i,dom_cc_lo.x),dom_cc_hi.x);
                                    bool up_ok =
                                        (u_xlo(dom_cc_lo.x,jju,k) >= zero) ||
                                        ((j == dom_cc_lo.y      ) && (v_xlo(iiv,dom_cc_lo.y  ,k) >= zero)) ||
                                        ((j == dom_cc_hi.y+iv[1]) && (v_xlo(iiv,dom_cc_hi.y+1,k) <= zero));
                                    apply = apply && up_ok;
                                }
                            } else { // xhi
                                Real x_strt = ProbHi[0] - width * dx[0];
                                Real xi = (x - x_strt) / (ProbHi[0] - x_strt);
                                Factor = amrex::max(xi*xi, eta*eta);
                                if (dbg_nudge_const_factor_l >= zero) { Factor = dbg_nudge_const_factor_l; }
                                if (do_upwind) {
                                    int jju = amrex::min(amrex::max(j,dom_cc_lo.y),dom_cc_hi.y);
                                    int iiv = amrex::min(amrex::max(i,dom_cc_lo.x),dom_cc_hi.x);
                                    bool up_ok =
                                        (u_xhi(dom_cc_hi.x+1,jju,k) <= zero) ||
                                        ((j == dom_cc_lo.y      ) && (v_xhi(iiv,dom_cc_lo.y  ,k) >= zero)) ||
                                        ((j == dom_cc_hi.y+iv[1]) && (v_xhi(iiv,dom_cc_hi.y+1,k) <= zero));
                                    apply = apply && up_ok;
                                }
                            }
                        } else { // y-side
                            Real y = ProbLo[1] + (j + joff) * dx[1];
                            if (sname[2] == 'o') { // ylo
                                Real y_end = ProbLo[1] + width * dx[1];
                                Real eta = (y_end - y) / (y_end - ProbLo[1]);
                                Factor = eta*eta;
                                if (dbg_nudge_const_factor_l >= zero) { Factor = dbg_nudge_const_factor_l; }
                                if (do_upwind) {
                                    int iiv = amrex::min(amrex::max(i,dom_cc_lo.x+width),dom_cc_hi.x-width);
                                    apply = (v_ylo(iiv,dom_cc_lo.y,k) >= zero);
                                }
                            } else { // yhi
                                Real y_strt = ProbHi[1] - width * dx[1];
                                Real eta = (y - y_strt) / (ProbHi[1] - y_strt);
                                Factor = eta*eta;
                                if (dbg_nudge_const_factor_l >= zero) { Factor = dbg_nudge_const_factor_l; }
                                if (do_upwind) {
                                    int iiv = amrex::min(amrex::max(i,dom_cc_lo.x+width),dom_cc_hi.x-width);
                                    apply = (v_yhi(iiv,dom_cc_hi.y+1,k) >= zero);
                                }
                            }
                        }

                        if (!apply) {
                            return {Real(0.0), Real(0.0), Long(0),
                                    Real(0.0), Real(0.0), Long(1)};
                        }
                        Real temp  = nudge_scale * Factor * F1 * delta;
                        Real tdp   = temp / rho;
                        return {temp, std::abs(temp), Long(1),
                                tdp, std::abs(tdp), Long(1)};
                    });
                    auto hv = rdata.value();
                    Real sum_temp    = amrex::get<0>(hv);
                    Real maxabs_temp = amrex::get<1>(hv);
                    Long n_apply     = amrex::get<2>(hv);
                    Real sum_tdp     = amrex::get<3>(hv);
                    Real maxabs_tdp  = amrex::get<4>(hv);
                    Long n_eval      = amrex::get<5>(hv);
                    if (n_eval > 0) {
                        const int my_rank = ParallelDescriptor::MyProc();
                        amrex::AllPrint() << "[DBG_TEND " << dbg_var << "]"
                                          << " rank=" << my_rank
                                          << " t=" << time
                                          << " t_elapsed=" << (time-start_bdy_time)
                                          << " bdy_idx=" << n_time
                                          << " side=" << sname
                                          << " width=" << width
                                          << " n_eval=" << n_eval
                                          << " n_apply=" << n_apply
                                          << " mean_temp=" << (n_apply > 0 ? sum_temp/Real(n_apply) : Real(0.0))
                                          << " maxabs_temp=" << (n_apply > 0 ? maxabs_temp : Real(0.0))
                                          << " mean_tdp=" << (n_apply > 0 ? sum_tdp/Real(n_apply) : Real(0.0))
                                          << " maxabs_tdp=" << (n_apply > 0 ? maxabs_tdp : Real(0.0))
                                          << "\n";
                    }
                };

                print_stats_for_side(tbx_xlo, arr_xlo, "xlo");
                print_stats_for_side(tbx_xhi, arr_xhi, "xhi");
                print_stats_for_side(tbx_ylo, arr_ylo, "ylo");
                print_stats_for_side(tbx_yhi, arr_yhi, "yhi");
                if (ivar == ivarV) {
                    print_xside_corner_split(tbx_xlo, arr_xlo, "xlo");
                    print_xside_corner_split(tbx_xhi, arr_xhi, "xhi");
                    print_xside_setrelax_split(tbx_xlo, arr_xlo, false, "xlo");
                    print_xside_setrelax_split(tbx_xhi, arr_xhi, true, "xhi");
                    print_tend_for_side(tbx_xlo, arr_xlo, "xlo");
                    print_tend_for_side(tbx_xhi, arr_xhi, "xhi");
                    print_tend_for_side(tbx_ylo, arr_ylo, "ylo");
                    print_tend_for_side(tbx_yhi, arr_yhi, "yhi");
                }

                scan_box_for_location(tbx_xlo, arr_xlo, 0);
                scan_box_for_location(tbx_xhi, arr_xhi, 1);
                scan_box_for_location(tbx_ylo, arr_ylo, 2);
                scan_box_for_location(tbx_yhi, arr_yhi, 3);

                // Compare theta in boundary band vs interior at top and bottom levels.
                auto reduce_band_stats_at_k = [&](int ksel,
                                                  Real& sum_bnd, Real& min_bnd, Real& max_bnd, Long& n_bnd,
                                                  Real& sum_int, Real& min_int, Real& max_int, Long& n_int)
                {
                    Box pbx = mfi.validbox();
                    pbx.setSmall(2, ksel);
                    pbx.setBig  (2, ksel);
                    if (!pbx.ok()) { return; }
                    ReduceOps<ReduceOpSum,ReduceOpMin,ReduceOpMax,ReduceOpSum,
                              ReduceOpSum,ReduceOpMin,ReduceOpMax,ReduceOpSum> rop;
                    ReduceData<Real,Real,Real,Long,Real,Real,Real,Long> rdata(rop);
                    using RT = typename decltype(rdata)::Type;
                    rop.eval(pbx, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> RT
                    {
                        Real rho = amrex::max(rho_arr(i,j,k,Rho_comp), Real(1.e-16));
                        Real thp = data_arr(i,j,k,icomp) / rho;
                        bool is_bnd = (i < dom3.x + width) || (i > domh.x - width) ||
                                      (j < dom3.y + width) || (j > domh.y - width);
                        if (is_bnd) {
                            return {thp, thp, thp, Long(1),
                                    Real(0.0), std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max(), Long(0)};
                        } else {
                            return {Real(0.0), std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max(), Long(0),
                                    thp, thp, thp, Long(1)};
                        }
                    });
                    auto hv = rdata.value();
                    sum_bnd += amrex::get<0>(hv);
                    min_bnd  = std::min(min_bnd, amrex::get<1>(hv));
                    max_bnd  = std::max(max_bnd, amrex::get<2>(hv));
                    n_bnd   += amrex::get<3>(hv);
                    sum_int += amrex::get<4>(hv);
                    min_int  = std::min(min_int, amrex::get<5>(hv));
                    max_int  = std::max(max_int, amrex::get<6>(hv));
                    n_int   += amrex::get<7>(hv);
                };
                reduce_band_stats_at_k(ktop,
                                       sum_top_bnd, min_top_bnd, max_top_bnd, n_top_bnd,
                                       sum_top_int, min_top_int, max_top_int, n_top_int);
                reduce_band_stats_at_k(dom3.z,
                                       sum_bot_bnd, min_bot_bnd, max_bot_bnd, n_bot_bnd,
                                       sum_bot_int, min_bot_int, max_bot_int, n_bot_int);
            }

            Real nudge_factor_local = nudge_factor;
            if (ivar == ivarU || ivar == ivarV) {
                if (dbg_nudge_factor_uv > Real(0.0)) { nudge_factor_local = dbg_nudge_factor_uv; }
            } else if (ivar == ivarT) {
                if (dbg_nudge_factor_tq > Real(0.0)) { nudge_factor_local = dbg_nudge_factor_tq; }
            }
            const Real F1_local = one / (nudge_factor_local * delta_t);

            realbdy_compute_relaxation(icomp, 1,
                                       width, dx, ProbLo, ProbHi, F1_local, geom.Domain(),
                                       tbx_xlo , tbx_xhi , tbx_ylo , tbx_yhi ,
                                       arr_xlo , arr_xhi , arr_ylo , arr_yhi ,
                                       u_xlo, u_xhi, v_xlo, v_xhi, v_ylo, v_yhi,
                                       rho_cc_arr,
                                       data_arr, rhs_arr, nudge_scale, dbg_nudge_x, dbg_nudge_y,
                                       dbg_nudge_exclude_x_corners, dbg_nudge_const_factor, do_upwind,
                                       y_face_owns_corners, dbg_realbdy_yface_corner_use_max_metric,
                                       dbg_realbdy_use_primitive_delta,
                                       dbg_realbdy_weight_profile_id, dbg_realbdy_weight_tanh_beta);

            /*
            // UNIT TEST DEBUG
            realbdy_interior_bxs_xy(tbx, domain, width,
            tbx_xlo, tbx_xhi,
            tbx_ylo, tbx_yhi);
            ParallelFor(tbx_xlo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
              if (std::fabs(arr_xlo(i,j,k) - data_arr(i,j,k,icomp)) > Real(0.01)) {
                Print() << "ERROR XLO: " << ivar << ' ' << icomp << ' ' << IntVect(i,j,k) << "\n";
                Print() << "DATA: " << data_arr(i,j,k,icomp) << ' ' << arr_xlo(i,j,k) << "\n";
                exit(0);
              }
            });
            ParallelFor(tbx_xhi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
              if (std::fabs(arr_xhi(i,j,k) - data_arr(i,j,k,icomp)) > Real(0.01)) {
                Print() << "ERROR XHI: " << ivar << ' ' << icomp << ' ' << IntVect(i,j,k) << "\n";
                Print() << "DATA: " << data_arr(i,j,k,icomp) << ' ' << arr_xhi(i,j,k) << "\n";
                exit(0);
              }
            });
            ParallelFor(tbx_ylo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
              if (std::fabs(arr_ylo(i,j,k) - data_arr(i,j,k,icomp)) > Real(0.01)) {
                Print() << "ERROR YLO: " << ivar << ' ' << icomp << ' ' << IntVect(i,j,k) << "\n";
                Print() << "DATA: " << data_arr(i,j,k,icomp) << ' ' << arr_ylo(i,j,k) << "\n";
                exit(0);
              }
            });
            ParallelFor(tbx_yhi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
              if (std::fabs(arr_yhi(i,j,k)-data_arr(i,j,k,icomp)) > Real(0.01)) {
                Print() << "ERROR YHI: " << ivar << ' ' << icomp << ' ' << IntVect(i,j,k) << "\n";
                Print() << "DATA: " << data_arr(i,j,k,icomp) << ' ' << arr_yhi(i,j,k) << "\n";
                exit(0);
              }
            });
            */
        } // mfi

        if ((dbg_nudge_print_stats &&
             (ivar == ivarV || (ivar == ivarT && !dbg_nudge_print_v_only)) &&
             ((ivar == ivarV) || ParallelDescriptor::IOProcessor())) ||
            (dbg_tend_v_print && ivar == ivarV) ) {
            const char* dbg_var = (ivar == ivarT) ? "theta" : "v";
            static const char* sname[NSIDES] = {"xlo","xhi","ylo","yhi"};
            const int my_rank = ParallelDescriptor::MyProc();
            for (int s=0; s<NSIDES; ++s) {
                if (best_absdp[s] >= 0.0) {
                    if (ivar == ivarV) {
                        amrex::AllPrint() << "[DBG_NUDGE " << dbg_var << "_loc]"
                                          << " rank=" << my_rank
                                          << " t=" << time
                                          << " t_elapsed=" << (time-start_bdy_time)
                                          << " bdy_idx=" << n_time
                                          << " side=" << sname[s]
                                          << " i=" << best_i[s]
                                          << " j=" << best_j[s]
                                          << " k=" << best_k[s]
                                          << " dc=" << best_dc[s]
                                          << " dp=" << best_dp[s]
                                          << " absdp=" << best_absdp[s]
                                          << " state_p=" << best_statep[s]
                                          << " targ_p=" << best_targp[s]
                                          << " rho=" << best_rho[s]
                                          << "\n";
                    } else {
                        Print() << "[DBG_NUDGE " << dbg_var << "_loc]"
                                << " rank=" << my_rank
                                << " t=" << time
                                << " t_elapsed=" << (time-start_bdy_time)
                                << " bdy_idx=" << n_time
                                << " side=" << sname[s]
                                << " i=" << best_i[s]
                                << " j=" << best_j[s]
                                << " k=" << best_k[s]
                                << " dc=" << best_dc[s]
                                << " dp=" << best_dp[s]
                                << " absdp=" << best_absdp[s]
                                << " state_p=" << best_statep[s]
                                << " targ_p=" << best_targp[s]
                                << " rho=" << best_rho[s]
                                << "\n";
                    }
                }
            }
        }

        if (dbg_nudge_print_stats && ivar == ivarT && !dbg_nudge_print_v_only) {
            ParallelDescriptor::ReduceRealSum(sum_top_bnd);
            ParallelDescriptor::ReduceRealMin(min_top_bnd);
            ParallelDescriptor::ReduceRealMax(max_top_bnd);
            ParallelDescriptor::ReduceLongSum(n_top_bnd);
            ParallelDescriptor::ReduceRealSum(sum_top_int);
            ParallelDescriptor::ReduceRealMin(min_top_int);
            ParallelDescriptor::ReduceRealMax(max_top_int);
            ParallelDescriptor::ReduceLongSum(n_top_int);
            ParallelDescriptor::ReduceRealSum(sum_bot_bnd);
            ParallelDescriptor::ReduceRealMin(min_bot_bnd);
            ParallelDescriptor::ReduceRealMax(max_bot_bnd);
            ParallelDescriptor::ReduceLongSum(n_bot_bnd);
            ParallelDescriptor::ReduceRealSum(sum_bot_int);
            ParallelDescriptor::ReduceRealMin(min_bot_int);
            ParallelDescriptor::ReduceRealMax(max_bot_int);
            ParallelDescriptor::ReduceLongSum(n_bot_int);

            if (ParallelDescriptor::IOProcessor()) {
                Print() << "[DBG_NUDGE theta_top_band]"
                        << " t=" << time
                        << " t_elapsed=" << (time-start_bdy_time)
                        << " bdy_idx=" << n_time
                        << " ktop=" << ubound(geom.Domain()).z
                        << " bnd_mean=" << (n_top_bnd > 0 ? sum_top_bnd/Real(n_top_bnd) : Real(0.0))
                        << " bnd_min=" << (n_top_bnd > 0 ? min_top_bnd : Real(0.0))
                        << " bnd_max=" << (n_top_bnd > 0 ? max_top_bnd : Real(0.0))
                        << " bnd_n=" << n_top_bnd
                        << " int_mean=" << (n_top_int > 0 ? sum_top_int/Real(n_top_int) : Real(0.0))
                        << " int_min=" << (n_top_int > 0 ? min_top_int : Real(0.0))
                        << " int_max=" << (n_top_int > 0 ? max_top_int : Real(0.0))
                        << " int_n=" << n_top_int
                        << "\n";
                Print() << "[DBG_NUDGE theta_bot_band]"
                        << " t=" << time
                        << " t_elapsed=" << (time-start_bdy_time)
                        << " bdy_idx=" << n_time
                        << " kbot=" << lbound(geom.Domain()).z
                        << " bnd_mean=" << (n_bot_bnd > 0 ? sum_bot_bnd/Real(n_bot_bnd) : Real(0.0))
                        << " bnd_min=" << (n_bot_bnd > 0 ? min_bot_bnd : Real(0.0))
                        << " bnd_max=" << (n_bot_bnd > 0 ? max_bot_bnd : Real(0.0))
                        << " bnd_n=" << n_bot_bnd
                        << " int_mean=" << (n_bot_int > 0 ? sum_bot_int/Real(n_bot_int) : Real(0.0))
                        << " int_min=" << (n_bot_int > 0 ? min_bot_int : Real(0.0))
                        << " int_max=" << (n_bot_int > 0 ? max_bot_int : Real(0.0))
                        << " int_n=" << n_bot_int
                        << "\n";
            }
        }
    } // ivar
    //ParallelDescriptor::Barrier();
    //exit(0);
}

/**
 * Compute the RHS in the fine relaxation zone
 *
 * @param[in]  time      current (elapsed) time
 * @param[in]  delta_t   timestep
 * @param[in]  width     number of cells in (relaxation+specified) zone
 * @param[in]  set_width number of cells in (specified) zone
 * @param[in]  FPr_c     cons fine patch container
 * @param[in]  FPr_u     uvel fine patch container
 * @param[in]  FPr_v     vvel fine patch container
 * @param[in]  FPr_w     wvel fine patch container
 * @param[in]  boxes_at_level boxes at current level
 * @param[in]  domain_bcs_type boundary condition types
 * @param[out] S_rhs     RHS to be computed here
 * @param[in]  S_data    current value of the solution
 */
void
fine_compute_interior_ghost_rhs (const Real& time,
                                 const Real& delta_t,
                                 const int& width,
                                 const int& set_width,
                                 const Geometry& geom,
                                 ERFFillPatcher* FPr_c,
                                 ERFFillPatcher* FPr_u,
                                 ERFFillPatcher* FPr_v,
                                 ERFFillPatcher* FPr_w,
                                 Vector<BCRec>& domain_bcs_type,
                                 Vector<MultiFab>& S_rhs_f,
                                 Vector<MultiFab>& S_data_f)
{
    BL_PROFILE_REGION("fine_compute_interior_ghost_RHS()");

    // Relaxation constants
    Real F1 = one/(Real(10.)*delta_t);
    Real F2 = one/(Real(50.)*delta_t);

    // Vector of MFs to hold data (dm differs w/ fine patch)
    Vector<MultiFab> fmf_p_v;

    // Loop over the variables
    for (int ivar_idx = 0; ivar_idx < IntVars::NumTypes; ++ivar_idx)
    {
        // Fine mfs
        MultiFab& fmf = S_data_f[ivar_idx];
        MultiFab& rhs = S_rhs_f [ivar_idx];

        // NOTE: These temporary MFs and copy operations are horrible
        //       for memory usage and efficiency. However, we need to
        //       have access to ghost cells in the cons array to convert
        //       from primitive u/v/w to momentum. Furthermore, the BA
        //       for the fine patches in ERFFillPatcher don't match the
        //       BA for the data/RHS. For this reason, the data is copied
        //       to a vector of MFs (with ghost cells) so the BAs match
        //       the BA of data/RHS and we have access to rho to convert
        //       prim to conserved.

        // Temp MF on box (distribution map differs w/ fine patch)
        int num_var = fmf.nComp();
        fmf_p_v.emplace_back(fmf.boxArray(), fmf.DistributionMap(), num_var, fmf.nGrowVect());
        MultiFab& fmf_p = fmf_p_v[ivar_idx];
        MultiFab::Copy(fmf_p,fmf, 0, 0, num_var, fmf.nGrowVect());

        // Integer mask MF
        int set_mask_val;
        int relax_mask_val;
        iMultiFab* mask;

        // Fill fine patch on interior halo region
        //==========================================================
        if (ivar_idx == IntVars::cons)
        {
            FPr_c->FillRelax(fmf_p, time, void_bc, domain_bcs_type);
            mask           = FPr_c->GetMask();
            set_mask_val   = FPr_c->GetSetMaskVal();
            relax_mask_val = FPr_c->GetRelaxMaskVal();
        }
        else if (ivar_idx == IntVars::xmom)
        {
            FPr_u->FillRelax(fmf_p, time, void_bc, domain_bcs_type);
            mask           = FPr_u->GetMask();
            set_mask_val   = FPr_u->GetSetMaskVal();
            relax_mask_val = FPr_u->GetRelaxMaskVal();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for ( MFIter mfi(fmf_p,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box tbx = mfi.tilebox();

                const Array4<Real>& prim_arr = fmf_p.array(mfi);
                const Array4<const Real>& rho_arr  = fmf_p_v[0].const_array(mfi);
                const Array4<const int>&  mask_arr = mask->const_array(mfi);

                ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (mask_arr(i,j,k) == relax_mask_val) {
                        Real rho_interp = myhalf * ( rho_arr(i-1,j,k) + rho_arr(i,j,k) );
                        prim_arr(i,j,k) *= rho_interp;
                    }
                });
            } // mfi
        }
        else if (ivar_idx == IntVars::ymom)
        {
            FPr_v->FillRelax(fmf_p, time, void_bc, domain_bcs_type);
            mask           = FPr_v->GetMask();
            set_mask_val   = FPr_v->GetSetMaskVal();
            relax_mask_val = FPr_v->GetRelaxMaskVal();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for ( MFIter mfi(fmf_p,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box tbx = mfi.tilebox();

                const Array4<Real>& prim_arr = fmf_p.array(mfi);
                const Array4<const Real>& rho_arr  = fmf_p_v[0].const_array(mfi);
                const Array4<const int>&  mask_arr = mask->const_array(mfi);

                ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (mask_arr(i,j,k) == relax_mask_val) {
                        Real rho_interp = myhalf * ( rho_arr(i,j-1,k) + rho_arr(i,j,k) );
                        prim_arr(i,j,k) *= rho_interp;
                    }
                });
            } // mfi
        }
        else if (ivar_idx == IntVars::zmom)
        {
            FPr_w->FillRelax(fmf_p, time, void_bc, domain_bcs_type);
            mask           = FPr_w->GetMask();
            set_mask_val   = FPr_w->GetSetMaskVal();
            relax_mask_val = FPr_w->GetRelaxMaskVal();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for ( MFIter mfi(fmf_p,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box tbx = mfi.tilebox();

                const Array4<Real>& prim_arr = fmf_p.array(mfi);
                const Array4<const Real>& rho_arr  = fmf_p_v[0].const_array(mfi);
                const Array4<const int>&  mask_arr = mask->const_array(mfi);

                ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (mask_arr(i,j,k) == relax_mask_val) {
                        Real rho_interp = myhalf * ( rho_arr(i,j,k-1) + rho_arr(i,j,k) );
                        prim_arr(i,j,k) *= rho_interp;
                    }
                });
            } // mfi
        } else {
            Abort("Dont recognize this variable type in fine_compute_interior_ghost_RHS");
        }


        // Zero RHS in set region
        //==========================================================
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(rhs,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box tbx = mfi.tilebox();
            const Array4<Real>& rhs_arr  = rhs.array(mfi);
            const Array4<const int>& mask_arr = mask->const_array(mfi);

            ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (mask_arr(i,j,k) == set_mask_val) {
                    rhs_arr(i,j,k) = zero;
                }
            });
        } // mfi

        // For Laplacian stencil
        rhs.FillBoundary(geom.periodicity());


        // Compute RHS in relaxation region
        //==========================================================
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(fmf_p,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box tbx = mfi.tilebox();
            const Array4<Real>&        rhs_arr = rhs.array(mfi);
            const Array4<const Real>& fine_arr = fmf_p.const_array(mfi);
            const Array4<const Real>& data_arr = fmf.const_array(mfi);
            const Array4<const int>&  mask_arr = mask->const_array(mfi);

            Box vbx = mfi.validbox();
            const auto& vbx_lo = lbound(vbx);
            const auto& vbx_hi = ubound(vbx);

            int icomp = 0;

            int Spec_z  = set_width;
            int Relax_z = width - Spec_z;
            Real num    = Real(Spec_z + Relax_z);
            Real denom  = Real(Relax_z - 1);
            ParallelFor(tbx, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
               if (mask_arr(i,j,k) == relax_mask_val) {

                   // Indices
                   Real n_ind(-1); // Set to -1 to quiet compiler warning
                   int ii(width-1); int jj(width-1);
                   bool near_x_lo_wall(false); bool near_x_hi_wall(false);
                   bool near_y_lo_wall(false); bool near_y_hi_wall(false);
                   bool mask_x_found(false);   bool mask_y_found(false);

                   // Near x-wall
                   if ((i-vbx_lo.x) < width) {
                       near_x_lo_wall = true;
                       ii = i-vbx_lo.x;
                       if (mask_arr(vbx_lo.x,j,k) == 2) mask_x_found = true;
                   } else if ((vbx_hi.x-i) < width) {
                       near_x_hi_wall = true;
                       ii = vbx_hi.x-i;
                       if (mask_arr(vbx_hi.x,j,k) == 2) mask_x_found = true;
                   }

                   // Near y-wall
                   if ((j-vbx_lo.y) < width) {
                       near_y_lo_wall = true;
                       jj = j-vbx_lo.y;
                       if (mask_arr(i,vbx_lo.y,k) == 2) mask_y_found = true;
                   } else if ((vbx_hi.y-j) < width) {
                       near_y_hi_wall = true;
                       jj = vbx_hi.y-j;
                       if (mask_arr(i,vbx_hi.y,k) == 2) mask_y_found = true;
                   }

                   // Found a nearby masked cell (valid n_ind)
                   if (mask_x_found && mask_y_found) {
                       n_ind = std::min(ii,jj) + one;
                   } else if (mask_x_found) {
                       n_ind = ii + one;
                   } else if (mask_y_found) {
                       n_ind = jj + one;
                   // Pesky corner cell
                   } else {
                       if (near_x_lo_wall || near_x_hi_wall) {
                           Real dj_min{width-one};
                           int j_lb = std::max(vbx_lo.y,j-width);
                           int j_ub = std::min(vbx_hi.y,j+width);
                           int li   = (near_x_lo_wall) ? vbx_lo.x : vbx_hi.x;
                           for (int lj(j_lb); lj<=j_ub; ++lj) {
                               if (mask_arr(li,lj,k) == 2) {
                                   mask_y_found = true;
                                   dj_min = std::min(dj_min,(Real) std::abs(lj-j));
                               }
                           }
                           if (mask_y_found) {
                               Real mag = sqrt( Real(dj_min*dj_min + ii*ii) );
                               n_ind = std::min(mag,width-one) + one;
                           } else {
                               Abort("Mask not found near x wall!");
                           }
                       } else if (near_y_lo_wall || near_y_hi_wall) {
                           Real di_min{width-one};
                           int i_lb = std::max(vbx_lo.x,i-width);
                           int i_ub = std::min(vbx_hi.x,i+width);
                           int lj   = (near_y_lo_wall) ? vbx_lo.y : vbx_hi.y;
                           for (int li(i_lb); li<=i_ub; ++li) {
                               if (mask_arr(li,lj,k) == 2) {
                                   mask_x_found = true;
                                   di_min = std::min(di_min,(Real) std::abs(li-i));
                               }
                           }
                           if (mask_x_found) {
                               Real mag = sqrt( Real(di_min*di_min + jj*jj) );
                               n_ind = std::min(mag,width-one) + one;
                           } else {
                               Abort("Mask not found near y wall!");
                           }
                       } else {
                           Abort("Relaxation cell must be near a wall!");
                       }
                   }

                   Real Factor   = (num - n_ind)/denom;
                   Real d        = data_arr(i  ,j  ,k  ,n+icomp) + delta_t*rhs_arr(i  , j  , k  ,n+icomp);
                   Real d_ip1    = data_arr(i+1,j  ,k  ,n+icomp) + delta_t*rhs_arr(i+1, j  , k  ,n+icomp);
                   Real d_im1    = data_arr(i-1,j  ,k  ,n+icomp) + delta_t*rhs_arr(i-1, j  , k  ,n+icomp);
                   Real d_jp1    = data_arr(i  ,j+1,k  ,n+icomp) + delta_t*rhs_arr(i  , j+1, k  ,n+icomp);
                   Real d_jm1    = data_arr(i  ,j-1,k  ,n+icomp) + delta_t*rhs_arr(i  , j-1, k  ,n+icomp);
                   Real delta    = fine_arr(i  ,j  ,k,n) - d;
                   Real delta_xp = fine_arr(i+1,j  ,k,n) - d_ip1;
                   Real delta_xm = fine_arr(i-1,j  ,k,n) - d_im1;
                   Real delta_yp = fine_arr(i  ,j+1,k,n) - d_jp1;
                   Real delta_ym = fine_arr(i  ,j-1,k,n) - d_jm1;
                   Real Laplacian = delta_xp + delta_xm + delta_yp + delta_ym - Real(4.0)*delta;
                   rhs_arr(i,j,k,n) += (F1*delta - F2*Laplacian) * Factor;
               }
            });
        } // mfi
    } // ivar_idx
}
