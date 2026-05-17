#include <ERF_Utils.H>
#include <AMReX_PlotFileUtil.H>
#include <array>

using namespace amrex;

PhysBCFunctNoOp void_bc;

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Real
realbdy_zcc_to_loc (const Array4<Real const>& zcc,
                    const Dim3& dom_lo, const Dim3& dom_hi,
                    int i, int j, int k,
                    int loc, int loc_u, int loc_v) noexcept
{
    amrex::ignore_unused(dom_lo, dom_hi);
    if (loc == loc_u) {
        return myhalf * (zcc(i-1,j,k,0) + zcc(i,j,k,0));
    } else if (loc == loc_v) {
        return myhalf * (zcc(i,j-1,k,0) + zcc(i,j,k,0));
    } else {
        return zcc(i,j,k,0);
    }
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Real
realbdy_z0nd_to_loc (const Array4<Real const>& znd,
                     const Dim3& dom_lo, const Dim3& dom_hi,
                     int i, int j,
                     int loc, int loc_u, int loc_v) noexcept
{
    const int in0 = amrex::min(amrex::max(i,   dom_lo.x), dom_hi.x);
    const int in1 = amrex::min(amrex::max(i+1, dom_lo.x), dom_hi.x);
    const int jn0 = amrex::min(amrex::max(j,   dom_lo.y), dom_hi.y);
    const int jn1 = amrex::min(amrex::max(j+1, dom_lo.y), dom_hi.y);
    constexpr int k0 = 0;

    if (loc == loc_u) {
        // u is x-face-centered, y-cell-centered
        return myhalf * (znd(in0,jn0,k0,0) + znd(in0,jn1,k0,0));
    } else if (loc == loc_v) {
        // v is y-face-centered, x-cell-centered
        return myhalf * (znd(in0,jn0,k0,0) + znd(in1,jn0,k0,0));
    } else {
        // theta/cc is cell-centered in x and y
        return fourth * (znd(in0,jn0,k0,0) + znd(in1,jn0,k0,0) +
                         znd(in0,jn1,k0,0) + znd(in1,jn1,k0,0));
    }
}

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
                                    const MultiFab& mf_MUB,
                                    const MultiFab& mf_C1H,
                                    const MultiFab& mf_C2H,
                                    const MultiFab& mf_C1F,
                                    const MultiFab& mf_C2F,
                                    const MultiFab& mf_DNW,
                                    const MultiFab& mf_PH_wrfin,
                                    const MultiFab& mf_PHB_wrfin,
                                    const MultiFab& mf_HGT_wrfin,
                                    const MultiFab& z_phys_nd,
                                    const MultiFab& z_phys_cc,
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
    static bool dbg_realbdy_rho_pathB_diag = false;
    static bool dbg_realbdy_use_wrf_rho_interp = false;
    static bool realbdy_vertical_remap_theta = false;
    static int realbdy_vertical_remap_mode = 1; // 1=A-2 (default), 2=B, 3=C, 4=D
    static bool dbg_realbdy_zdrift_diag = false;
    static bool dbg_realbdy_zstats_diag = false;
    static bool dbg_realbdy_zalign_diag = false;
    static int dbg_realbdy_zalign_every_nrhs = 1;
    static int dbg_realbdy_zalign_counter = 0;
    static bool dbg_realbdy_dump_theta_fields = false;
    static int dbg_realbdy_dump_theta_every_nrhs = 1;
    static int dbg_realbdy_dump_theta_counter = 0;
    static std::string dbg_realbdy_dump_theta_prefix = "realbdy_theta_dbg";
    static int dbg_realbdy_zdrift_every_nrhs = 20;
    static int dbg_realbdy_zdrift_counter = 0;
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
        pp.query("dbg_realbdy_rho_pathB_diag", dbg_realbdy_rho_pathB_diag);
        pp.query("dbg_realbdy_use_wrf_rho_interp", dbg_realbdy_use_wrf_rho_interp);
        pp.query("realbdy_vertical_remap_theta", realbdy_vertical_remap_theta);
        pp.query("realbdy_vertical_remap_mode", realbdy_vertical_remap_mode);
        pp.query("dbg_realbdy_zdrift_diag", dbg_realbdy_zdrift_diag);
        pp.query("dbg_realbdy_zstats_diag", dbg_realbdy_zstats_diag);
        pp.query("dbg_realbdy_zalign_diag", dbg_realbdy_zalign_diag);
        pp.query("dbg_realbdy_zalign_every_nrhs", dbg_realbdy_zalign_every_nrhs);
        pp.query("dbg_realbdy_dump_theta_fields", dbg_realbdy_dump_theta_fields);
        pp.query("dbg_realbdy_dump_theta_every_nrhs", dbg_realbdy_dump_theta_every_nrhs);
        pp.query("dbg_realbdy_dump_theta_prefix", dbg_realbdy_dump_theta_prefix);
        pp.query("dbg_realbdy_zdrift_every_nrhs", dbg_realbdy_zdrift_every_nrhs);
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
    static bool dbg_realbdy_boxes_print_once = false;

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
    const bool have_wrfbdy_ph = (bdy_data_xlo[n_time].size() > WRFBdyVars::PH &&
                                 bdy_data_xlo[n_time_p1].size() > WRFBdyVars::PH);
    static bool dbg_rho_size_once = false;
    if (dbg_realbdy_rho_pathB_diag && !dbg_rho_size_once && ParallelDescriptor::IOProcessor()) {
        dbg_rho_size_once = true;
        Print() << "[DBG_RHO_PATHB sizes] n_time=" << n_time
                << " n_time_p1=" << n_time_p1
                << " size(n)=" << bdy_data_xlo[n_time].size()
                << " size(np1)=" << bdy_data_xlo[n_time_p1].size()
                << " WRFBdyVars::PH=" << WRFBdyVars::PH
                << std::endl;
    }
    const bool use_wrf_rho_interp = dbg_realbdy_use_wrf_rho_interp && have_wrfbdy_ph;
    const bool use_theta_vertical_remap = realbdy_vertical_remap_theta && have_wrfbdy_ph;
    if (dbg_realbdy_use_wrf_rho_interp && !have_wrfbdy_ph && ParallelDescriptor::IOProcessor()) {
        Print() << "[DBG_RHO_PATHB] PH boundary field unavailable; using ERF rho interpolation instead.\n";
    }
    if (realbdy_vertical_remap_theta && !have_wrfbdy_ph && ParallelDescriptor::IOProcessor()) {
        Print() << "[REALBDY theta-remap] PH boundary field unavailable; disabling theta vertical remap.\n";
    }
    ++dbg_realbdy_dump_theta_counter;
    const bool do_dump_theta_this_rhs = dbg_realbdy_dump_theta_fields &&
        (dbg_realbdy_dump_theta_every_nrhs <= 1 ||
         (dbg_realbdy_dump_theta_counter % dbg_realbdy_dump_theta_every_nrhs == 0));

#ifndef AMREX_USE_GPU
    ++dbg_realbdy_zalign_counter;
    const bool do_zalign_this_rhs = dbg_realbdy_zalign_diag && have_wrfbdy_ph &&
        (dbg_realbdy_zalign_every_nrhs <= 1 ||
         (dbg_realbdy_zalign_counter % dbg_realbdy_zalign_every_nrhs == 0));

    if (do_zalign_this_rhs) {
        struct ZStats2 {
            Long n = 0;
            Real minv = std::numeric_limits<Real>::max();
            Real maxv = -std::numeric_limits<Real>::max();
            Real sum = 0.0_rt;
            Real sumabs = 0.0_rt;
            Real sumsq = 0.0_rt;
            AMREX_FORCE_INLINE void add(Real v) noexcept {
                if (!std::isfinite(static_cast<double>(v))) return;
                ++n;
                minv = std::min(minv, v);
                maxv = std::max(maxv, v);
                sum += v;
                sumabs += std::abs(v);
                sumsq += v*v;
            }
        };
        auto reduce_stats2 = [] (ZStats2& s) {
            ParallelDescriptor::ReduceLongSum(s.n);
            ParallelDescriptor::ReduceRealMin(s.minv);
            ParallelDescriptor::ReduceRealMax(s.maxv);
            ParallelDescriptor::ReduceRealSum(s.sum);
            ParallelDescriptor::ReduceRealSum(s.sumabs);
            ParallelDescriptor::ReduceRealSum(s.sumsq);
        };

        enum {LOC_T=0, LOC_U=1, LOC_V=2};
        enum {FXLO=0, FXHI=1, FYLO=2, FYHI=3};
        constexpr int NLOC = 3, NFACE = 4, NK = 7;
        const std::array<int,NK> k_levels = {0,5,10,20,40,80,119};
        ZStats2 stats_minus[NLOC][NFACE][NK];
        ZStats2 stats_minus_edgeinc[NLOC][NFACE][NK];

        const auto& ph_xlo_n   = bdy_data_xlo[n_time][WRFBdyVars::PH].const_array();
        const auto& ph_xlo_np1 = bdy_data_xlo[n_time_p1][WRFBdyVars::PH].const_array();
        const auto& ph_xhi_n   = bdy_data_xhi[n_time][WRFBdyVars::PH].const_array();
        const auto& ph_xhi_np1 = bdy_data_xhi[n_time_p1][WRFBdyVars::PH].const_array();
        const auto& ph_ylo_n   = bdy_data_ylo[n_time][WRFBdyVars::PH].const_array();
        const auto& ph_ylo_np1 = bdy_data_ylo[n_time_p1][WRFBdyVars::PH].const_array();
        const auto& ph_yhi_n   = bdy_data_yhi[n_time][WRFBdyVars::PH].const_array();
        const auto& ph_yhi_np1 = bdy_data_yhi[n_time_p1][WRFBdyVars::PH].const_array();
        const auto& mu_xlo_n   = bdy_data_xlo[n_time][WRFBdyVars::MU].const_array();
        const auto& mu_xlo_np1 = bdy_data_xlo[n_time_p1][WRFBdyVars::MU].const_array();
        const auto& mu_xhi_n   = bdy_data_xhi[n_time][WRFBdyVars::MU].const_array();
        const auto& mu_xhi_np1 = bdy_data_xhi[n_time_p1][WRFBdyVars::MU].const_array();
        const auto& mu_ylo_n   = bdy_data_ylo[n_time][WRFBdyVars::MU].const_array();
        const auto& mu_ylo_np1 = bdy_data_ylo[n_time_p1][WRFBdyVars::MU].const_array();
        const auto& mu_yhi_n   = bdy_data_yhi[n_time][WRFBdyVars::MU].const_array();
        const auto& mu_yhi_np1 = bdy_data_yhi[n_time_p1][WRFBdyVars::MU].const_array();

        const Box dom = geom.Domain();
        const int ilo = dom.smallEnd(0), ihi = dom.bigEnd(0);
        const int jlo = dom.smallEnd(1), jhi = dom.bigEnd(1);
        const int nd_xlo = ilo, nd_xhi = ihi + 1;
        const int nd_ylo = jlo, nd_yhi = jhi + 1;
        const int nd_zlo = dom.smallEnd(2), nd_zhi = dom.bigEnd(2) + 1;

        for (MFIter mfi(z_phys_nd); mfi.isValid(); ++mfi) {
            const auto znd = z_phys_nd.const_array(mfi);
            const auto zcc = z_phys_cc.const_array(mfi);
            const auto hgt = mf_HGT_wrfin.const_array(mfi);
            const auto phb = mf_PHB_wrfin.const_array(mfi);
            const auto mub = mf_MUB.const_array(mfi);
            const auto c1f = mf_C1F.const_array(mfi);
            const auto c2f = mf_C2F.const_array(mfi);
            const int kmax_cf = mf_C1F[mfi].box().bigEnd(2);
            const Box bx = mfi.validbox() & amrex::surroundingNodes(dom);
            if (!bx.ok()) continue;

            const Dim3 dom_cc_lo = lbound(dom);
            const Dim3 dom_cc_hi = ubound(dom);
            const Box dom_nd_box = amrex::surroundingNodes(dom);
            const Dim3 dom_nd_lo = lbound(dom_nd_box);
            const Dim3 dom_nd_hi = ubound(dom_nd_box);

            auto ztgt = [=] (int loc, int i, int j, int k) -> Real {
                return realbdy_zcc_to_loc(zcc, dom_cc_lo, dom_cc_hi, i, j, k, loc, LOC_U, LOC_V);
            };
            auto hgt_loc = [=] (int loc, int i, int j) -> Real {
                return realbdy_zcc_to_loc(hgt, dom_cc_lo, dom_cc_hi, i, j, 0, loc, LOC_U, LOC_V);
            };
            auto z0_erf_loc = [=] (int loc, int i, int j) -> Real {
                return realbdy_z0nd_to_loc(znd, dom_nd_lo, dom_nd_hi, i, j, loc, LOC_U, LOC_V);
            };
            auto d_agl = [=] (int loc, int i, int j, int k, Real zsrc) -> Real {
                const Real z_wrf_agl = zsrc - hgt_loc(loc, i, j);
                const Real z_erf_agl = ztgt(loc, i, j, k) - z0_erf_loc(loc, i, j);
                return z_wrf_agl - z_erf_agl;
            };

            auto zw_face = [=] (int face, int i, int j, int k) -> Real {
                const int kc = amrex::max(0, amrex::min(k, kmax_cf));
                if (face == FXLO) {
                    Real mu_t = oma * mu_xlo_n(i,j,0,0) + alpha * mu_xlo_np1(i,j,0,0) + mub(i,j,0,0);
                    Real xmu_f = c1f(0,0,kc,0) * mu_t + c2f(0,0,kc,0);
                    Real ph_t = oma * ph_xlo_n(i,j,k,0) + alpha * ph_xlo_np1(i,j,k,0);
                    return (ph_t / xmu_f + phb(i,j,k,0)) / CONST_GRAV;
                } else if (face == FXHI) {
                    Real mu_t = oma * mu_xhi_n(i,j,0,0) + alpha * mu_xhi_np1(i,j,0,0) + mub(i,j,0,0);
                    Real xmu_f = c1f(0,0,kc,0) * mu_t + c2f(0,0,kc,0);
                    Real ph_t = oma * ph_xhi_n(i,j,k,0) + alpha * ph_xhi_np1(i,j,k,0);
                    return (ph_t / xmu_f + phb(i,j,k,0)) / CONST_GRAV;
                } else if (face == FYLO) {
                    Real mu_t = oma * mu_ylo_n(i,j,0,0) + alpha * mu_ylo_np1(i,j,0,0) + mub(i,j,0,0);
                    Real xmu_f = c1f(0,0,kc,0) * mu_t + c2f(0,0,kc,0);
                    Real ph_t = oma * ph_ylo_n(i,j,k,0) + alpha * ph_ylo_np1(i,j,k,0);
                    return (ph_t / xmu_f + phb(i,j,k,0)) / CONST_GRAV;
                } else {
                    Real mu_t = oma * mu_yhi_n(i,j,0,0) + alpha * mu_yhi_np1(i,j,0,0) + mub(i,j,0,0);
                    Real xmu_f = c1f(0,0,kc,0) * mu_t + c2f(0,0,kc,0);
                    Real ph_t = oma * ph_yhi_n(i,j,k,0) + alpha * ph_yhi_np1(i,j,k,0);
                    return (ph_t / xmu_f + phb(i,j,k,0)) / CONST_GRAV;
                }
            };

            for (int b = 0; b < width; ++b) {
                int ixlo = ilo + b;
                int ixhi = ihi - b;
                for (int j = std::max(jlo, bx.smallEnd(1)); j <= std::min(jhi, bx.bigEnd(1)); ++j) {
                    for (int ik = 0; ik < NK; ++ik) {
                        int k = k_levels[ik];
                        if (k < std::max(0, bx.smallEnd(2)) || k+1 > std::min(dom.bigEnd(2)+1, bx.bigEnd(2))) continue;
                        if (ixlo >= bx.smallEnd(0) && ixlo <= bx.bigEnd(0)) {
                            const bool has_i_minus = (ixlo-1 >= ilo);
                            const bool has_j_minus = (j-1 >= jlo);
                            int il_m = amrex::max(ilo, amrex::min(ixlo-1, ihi));
                            int ir_m = amrex::max(ilo, amrex::min(ixlo,   ihi));
                            int jb_m = amrex::max(jlo, amrex::min(j, jhi));
                            int jl_m = amrex::max(jlo, amrex::min(j-1, jhi));
                            int ju_m = amrex::max(jlo, amrex::min(j,   jhi));
                            Real zsrc_t_m = Real(0.5) * (zw_face(FXLO,ir_m,jb_m,k) + zw_face(FXLO,ir_m,jb_m,k+1));
                            Real zsrc_u_m = Real(0.25) * (zw_face(FXLO,il_m,jb_m,k) + zw_face(FXLO,il_m,jb_m,k+1) +
                                                          zw_face(FXLO,ir_m,jb_m,k) + zw_face(FXLO,ir_m,jb_m,k+1));
                            Real zsrc_v_m = Real(0.25) * (zw_face(FXLO,ir_m,jl_m,k) + zw_face(FXLO,ir_m,jl_m,k+1) +
                                                          zw_face(FXLO,ir_m,ju_m,k) + zw_face(FXLO,ir_m,ju_m,k+1));
                            Real d_t = d_agl(LOC_T, ixlo, j, k, zsrc_t_m);
                            Real d_u = d_agl(LOC_U, ixlo, j, k, zsrc_u_m);
                            Real d_v = d_agl(LOC_V, ixlo, j, k, zsrc_v_m);
                            stats_minus[LOC_T][FXLO][ik].add(d_t);
                            stats_minus_edgeinc[LOC_T][FXLO][ik].add(d_t);
                            if (has_i_minus) stats_minus[LOC_U][FXLO][ik].add(d_agl(LOC_U, ixlo, j, k, zsrc_u_m));
                            if (has_j_minus) stats_minus[LOC_V][FXLO][ik].add(d_agl(LOC_V, ixlo, j, k, zsrc_v_m));
                            stats_minus_edgeinc[LOC_U][FXLO][ik].add(d_u);
                            stats_minus_edgeinc[LOC_V][FXLO][ik].add(d_v);

                        }
                        if (ixhi >= bx.smallEnd(0) && ixhi <= bx.bigEnd(0)) {
                            const bool has_i_minus = (ixhi-1 >= ilo);
                            const bool has_j_minus = (j-1 >= jlo);
                            int il_m = amrex::max(ilo, amrex::min(ixhi-1, ihi));
                            int ir_m = amrex::max(ilo, amrex::min(ixhi,   ihi));
                            int jb_m = amrex::max(jlo, amrex::min(j, jhi));
                            int jl_m = amrex::max(jlo, amrex::min(j-1, jhi));
                            int ju_m = amrex::max(jlo, amrex::min(j,   jhi));
                            Real zsrc_t_m = Real(0.5) * (zw_face(FXHI,ir_m,jb_m,k) + zw_face(FXHI,ir_m,jb_m,k+1));
                            Real zsrc_u_m = Real(0.25) * (zw_face(FXHI,il_m,jb_m,k) + zw_face(FXHI,il_m,jb_m,k+1) +
                                                          zw_face(FXHI,ir_m,jb_m,k) + zw_face(FXHI,ir_m,jb_m,k+1));
                            Real zsrc_v_m = Real(0.25) * (zw_face(FXHI,ir_m,jl_m,k) + zw_face(FXHI,ir_m,jl_m,k+1) +
                                                          zw_face(FXHI,ir_m,ju_m,k) + zw_face(FXHI,ir_m,ju_m,k+1));
                            Real d_t = d_agl(LOC_T, ixhi, j, k, zsrc_t_m);
                            Real d_u = d_agl(LOC_U, ixhi, j, k, zsrc_u_m);
                            Real d_v = d_agl(LOC_V, ixhi, j, k, zsrc_v_m);
                            stats_minus[LOC_T][FXHI][ik].add(d_t);
                            stats_minus_edgeinc[LOC_T][FXHI][ik].add(d_t);
                            if (has_i_minus) stats_minus[LOC_U][FXHI][ik].add(d_agl(LOC_U, ixhi, j, k, zsrc_u_m));
                            if (has_j_minus) stats_minus[LOC_V][FXHI][ik].add(d_agl(LOC_V, ixhi, j, k, zsrc_v_m));
                            stats_minus_edgeinc[LOC_U][FXHI][ik].add(d_u);
                            stats_minus_edgeinc[LOC_V][FXHI][ik].add(d_v);

                        }
                    }
                }
            }

            for (int b = 0; b < width; ++b) {
                int jylo = jlo + b;
                int jyhi = jhi - b;
                for (int i = std::max(ilo, bx.smallEnd(0)); i <= std::min(ihi, bx.bigEnd(0)); ++i) {
                    for (int ik = 0; ik < NK; ++ik) {
                        int k = k_levels[ik];
                        if (k < std::max(0, bx.smallEnd(2)) || k+1 > std::min(dom.bigEnd(2)+1, bx.bigEnd(2))) continue;
                        if (jylo >= bx.smallEnd(1) && jylo <= bx.bigEnd(1)) {
                            const bool has_i_minus = (i-1 >= ilo);
                            const bool has_j_minus = (jylo-1 >= jlo);
                            int jl_m = amrex::max(jlo, amrex::min(jylo-1, jhi));
                            int ju_m = amrex::max(jlo, amrex::min(jylo,   jhi));
                            int ib_m = amrex::max(ilo, amrex::min(i, ihi));
                            int il_m = amrex::max(ilo, amrex::min(i-1, ihi));
                            int ir_m = amrex::max(ilo, amrex::min(i,   ihi));
                            Real zsrc_t_m = Real(0.5) * (zw_face(FYLO,ib_m,ju_m,k) + zw_face(FYLO,ib_m,ju_m,k+1));
                            Real zsrc_u_m = Real(0.25) * (zw_face(FYLO,il_m,ju_m,k) + zw_face(FYLO,il_m,ju_m,k+1) +
                                                          zw_face(FYLO,ir_m,ju_m,k) + zw_face(FYLO,ir_m,ju_m,k+1));
                            Real zsrc_v_m = Real(0.25) * (zw_face(FYLO,ib_m,jl_m,k) + zw_face(FYLO,ib_m,jl_m,k+1) +
                                                          zw_face(FYLO,ib_m,ju_m,k) + zw_face(FYLO,ib_m,ju_m,k+1));
                            Real d_t = d_agl(LOC_T, i, jylo, k, zsrc_t_m);
                            Real d_u = d_agl(LOC_U, i, jylo, k, zsrc_u_m);
                            Real d_v = d_agl(LOC_V, i, jylo, k, zsrc_v_m);
                            stats_minus[LOC_T][FYLO][ik].add(d_t);
                            stats_minus_edgeinc[LOC_T][FYLO][ik].add(d_t);
                            if (has_i_minus) stats_minus[LOC_U][FYLO][ik].add(d_agl(LOC_U, i, jylo, k, zsrc_u_m));
                            if (has_j_minus) stats_minus[LOC_V][FYLO][ik].add(d_agl(LOC_V, i, jylo, k, zsrc_v_m));
                            stats_minus_edgeinc[LOC_U][FYLO][ik].add(d_u);
                            stats_minus_edgeinc[LOC_V][FYLO][ik].add(d_v);

                        }
                        if (jyhi >= bx.smallEnd(1) && jyhi <= bx.bigEnd(1)) {
                            const bool has_i_minus = (i-1 >= ilo);
                            const bool has_j_minus = (jyhi-1 >= jlo);
                            int jl_m = amrex::max(jlo, amrex::min(jyhi-1, jhi));
                            int ju_m = amrex::max(jlo, amrex::min(jyhi,   jhi));
                            int ib_m = amrex::max(ilo, amrex::min(i, ihi));
                            int il_m = amrex::max(ilo, amrex::min(i-1, ihi));
                            int ir_m = amrex::max(ilo, amrex::min(i,   ihi));
                            Real zsrc_t_m = Real(0.5) * (zw_face(FYHI,ib_m,ju_m,k) + zw_face(FYHI,ib_m,ju_m,k+1));
                            Real zsrc_u_m = Real(0.25) * (zw_face(FYHI,il_m,ju_m,k) + zw_face(FYHI,il_m,ju_m,k+1) +
                                                          zw_face(FYHI,ir_m,ju_m,k) + zw_face(FYHI,ir_m,ju_m,k+1));
                            Real zsrc_v_m = Real(0.25) * (zw_face(FYHI,ib_m,jl_m,k) + zw_face(FYHI,ib_m,jl_m,k+1) +
                                                          zw_face(FYHI,ib_m,ju_m,k) + zw_face(FYHI,ib_m,ju_m,k+1));
                            Real d_t = d_agl(LOC_T, i, jyhi, k, zsrc_t_m);
                            Real d_u = d_agl(LOC_U, i, jyhi, k, zsrc_u_m);
                            Real d_v = d_agl(LOC_V, i, jyhi, k, zsrc_v_m);
                            stats_minus[LOC_T][FYHI][ik].add(d_t);
                            stats_minus_edgeinc[LOC_T][FYHI][ik].add(d_t);
                            if (has_i_minus) stats_minus[LOC_U][FYHI][ik].add(d_agl(LOC_U, i, jyhi, k, zsrc_u_m));
                            if (has_j_minus) stats_minus[LOC_V][FYHI][ik].add(d_agl(LOC_V, i, jyhi, k, zsrc_v_m));
                            stats_minus_edgeinc[LOC_U][FYHI][ik].add(d_u);
                            stats_minus_edgeinc[LOC_V][FYHI][ik].add(d_v);

                        }
                    }
                }
            }
        }

        const char* loc_name[NLOC] = {"theta_cc","u_face","v_face"};
        const char* face_name[NFACE] = {"xlo","xhi","ylo","yhi"};
        for (int il=0; il<NLOC; ++il) for (int jf=0; jf<NFACE; ++jf) for (int ik=0; ik<NK; ++ik) {
            reduce_stats2(stats_minus[il][jf][ik]);
            reduce_stats2(stats_minus_edgeinc[il][jf][ik]);
        }

        auto merge_stats2 = [] (ZStats2& acc, const ZStats2& s) {
            if (s.n == 0) return;
            if (acc.n == 0) {
                acc.minv = s.minv;
                acc.maxv = s.maxv;
            } else {
                acc.minv = std::min(acc.minv, s.minv);
                acc.maxv = std::max(acc.maxv, s.maxv);
            }
            acc.n += s.n;
            acc.sum += s.sum;
            acc.sumabs += s.sumabs;
            acc.sumsq += s.sumsq;
        };

        auto print_stats2 = [] (const char* tag, const ZStats2& s) {
            if (s.n == 0) {
                Print() << "  " << tag << ": n=0\n";
                return;
            }
            Real mean = s.sum / static_cast<Real>(s.n);
            Real absmean = s.sumabs / static_cast<Real>(s.n);
            Real rms = std::sqrt(s.sumsq / static_cast<Real>(s.n));
            Print() << "  " << tag
                    << ": n=" << s.n
                    << " min=" << s.minv
                    << " max=" << s.maxv
                    << " mean=" << mean
                    << " absmean=" << absmean
                    << " rms=" << rms << "\n";
        };

        ZStats2 stats_face[NLOC][NFACE];
        ZStats2 stats_strip[NLOC];
        ZStats2 stats_face_edgeinc[NLOC][NFACE];
        ZStats2 stats_strip_edgeinc[NLOC];
        for (int il=0; il<NLOC; ++il) {
            for (int jf=0; jf<NFACE; ++jf) {
                for (int ik=0; ik<NK; ++ik) {
                    merge_stats2(stats_face[il][jf], stats_minus[il][jf][ik]);
                    merge_stats2(stats_face_edgeinc[il][jf], stats_minus_edgeinc[il][jf][ik]);
                }
                merge_stats2(stats_strip[il], stats_face[il][jf]);
                merge_stats2(stats_strip_edgeinc[il], stats_face_edgeinc[il][jf]);
            }
        }

        if (ParallelDescriptor::IOProcessor()) {
            Print() << "[DBG_ZALIGN] AGL comparison | time=" << time << " alpha=" << alpha
                    << " n_time=" << n_time << " n_time_p1=" << n_time_p1
                    << " rhs_count=" << dbg_realbdy_zalign_counter << "\n";
            for (int il=0; il<NLOC; ++il) {
                Print() << " loc=" << loc_name[il] << "\n";
                print_stats2("strip_agl", stats_strip[il]);
                for (int jf=0; jf<NFACE; ++jf) {
                    print_stats2(face_name[jf], stats_face[il][jf]);
                }
                Print() << " loc=" << loc_name[il] << " (include_strip_edges)\n";
                print_stats2("strip_agl", stats_strip_edgeinc[il]);
                for (int jf=0; jf<NFACE; ++jf) {
                    print_stats2(face_name[jf], stats_face_edgeinc[il][jf]);
                }
            }
        }
    }

    ++dbg_realbdy_zdrift_counter;
    if (dbg_realbdy_zdrift_diag && have_wrfbdy_ph && ParallelDescriptor::IOProcessor() &&
        (dbg_realbdy_zdrift_every_nrhs <= 1 || (dbg_realbdy_zdrift_counter % dbg_realbdy_zdrift_every_nrhs == 0))) {
        const int klist_raw[3] = {0, 5, 20};
        const auto& ph_xlo_n   = bdy_data_xlo[n_time][WRFBdyVars::PH].const_array();
        const auto& ph_xlo_np1 = bdy_data_xlo[n_time_p1][WRFBdyVars::PH].const_array();
        const auto& ph_xhi_n   = bdy_data_xhi[n_time][WRFBdyVars::PH].const_array();
        const auto& ph_xhi_np1 = bdy_data_xhi[n_time_p1][WRFBdyVars::PH].const_array();
        const auto& ph_ylo_n   = bdy_data_ylo[n_time][WRFBdyVars::PH].const_array();
        const auto& ph_ylo_np1 = bdy_data_ylo[n_time_p1][WRFBdyVars::PH].const_array();
        const auto& ph_yhi_n   = bdy_data_yhi[n_time][WRFBdyVars::PH].const_array();
        const auto& ph_yhi_np1 = bdy_data_yhi[n_time_p1][WRFBdyVars::PH].const_array();
        const auto& mu_xlo_n   = bdy_data_xlo[n_time][WRFBdyVars::MU].const_array();
        const auto& mu_xlo_np1 = bdy_data_xlo[n_time_p1][WRFBdyVars::MU].const_array();
        const auto& mu_xhi_n   = bdy_data_xhi[n_time][WRFBdyVars::MU].const_array();
        const auto& mu_xhi_np1 = bdy_data_xhi[n_time_p1][WRFBdyVars::MU].const_array();
        const auto& mu_ylo_n   = bdy_data_ylo[n_time][WRFBdyVars::MU].const_array();
        const auto& mu_ylo_np1 = bdy_data_ylo[n_time_p1][WRFBdyVars::MU].const_array();
        const auto& mu_yhi_n   = bdy_data_yhi[n_time][WRFBdyVars::MU].const_array();
        const auto& mu_yhi_np1 = bdy_data_yhi[n_time_p1][WRFBdyVars::MU].const_array();

        bool printed = false;
        for (MFIter mfi(z_phys_nd); mfi.isValid() && !printed; ++mfi) {
            const auto znd = z_phys_nd.const_array(mfi);
            const auto phb = mf_PHB_wrfin.const_array(mfi);
            const auto mub = mf_MUB.const_array(mfi);
            const auto c1f = mf_C1F.const_array(mfi);
            const auto c2f = mf_C2F.const_array(mfi);
            const Box bx = mfi.validbox();
            int jmid = (bx.smallEnd(1) + bx.bigEnd(1)) / 2;
            int imid = (bx.smallEnd(0) + bx.bigEnd(0)) / 2;
            int ilo = geom.Domain().smallEnd(0);
            int ihi = geom.Domain().bigEnd(0);
            int jlo = geom.Domain().smallEnd(1);
            int jhi = geom.Domain().bigEnd(1);
            const int ixl = std::min(std::max(ilo, bx.smallEnd(0)), bx.bigEnd(0));
            const int ixh = std::min(std::max(ihi, bx.smallEnd(0)), bx.bigEnd(0));
            const int iym = std::min(std::max(imid,   bx.smallEnd(0)), bx.bigEnd(0));
            const int jyl = std::min(std::max(jlo, bx.smallEnd(1)), bx.bigEnd(1));
            const int jyh = std::min(std::max(jhi, bx.smallEnd(1)), bx.bigEnd(1));
            const int jxm = std::min(std::max(jmid,    bx.smallEnd(1)), bx.bigEnd(1));

            Print() << "[DBG_ZDRIFT] time=" << time << " alpha=" << alpha
                    << " n_time=" << n_time << " n_time_p1=" << n_time_p1 << "\n";
            for (int kk = 0; kk < 3; ++kk) {
                int k = std::min(klist_raw[kk], geom.Domain().bigEnd(2)-1);
                auto wrf_zw_x = [&](auto phn, auto php1, auto mun, auto mup1,
                                    int ig, int jg, int kz) {
                    const Real mu_bdy = oma * mun(ig,jg,0,0) + alpha * mup1(ig,jg,0,0);
                    const Real mu_d = mu_bdy + mub(ig,jg,0,0);
                    const Real xmu_mult_f = c1f(0,0,kz,0) * mu_d + c2f(0,0,kz,0);
                    Real phi = (oma * phn(ig,jg,kz,0) + alpha * php1(ig,jg,kz,0)) / xmu_mult_f
                               + phb(ig,jg,kz,0);
                    return phi / Real(9.81);
                };
                auto wrf_zw_y = [&](auto phn, auto php1, auto mun, auto mup1,
                                    int ig, int jg, int kz) {
                    const Real mu_bdy = oma * mun(ig,jg,0,0) + alpha * mup1(ig,jg,0,0);
                    const Real mu_d = mu_bdy + mub(ig,jg,0,0);
                    const Real xmu_mult_f = c1f(0,0,kz,0) * mu_d + c2f(0,0,kz,0);
                    Real phi = (oma * phn(ig,jg,kz,0) + alpha * php1(ig,jg,kz,0)) / xmu_mult_f
                               + phb(ig,jg,kz,0);
                    return phi / Real(9.81);
                };
                Real zw_xlo_0 = wrf_zw_x(ph_xlo_n, ph_xlo_np1, mu_xlo_n, mu_xlo_np1, ixl, jxm, k  );
                Real zw_xlo_1 = wrf_zw_x(ph_xlo_n, ph_xlo_np1, mu_xlo_n, mu_xlo_np1, ixl, jxm, k+1);
                Real zw_xhi_0 = wrf_zw_x(ph_xhi_n, ph_xhi_np1, mu_xhi_n, mu_xhi_np1, ixh, jxm, k  );
                Real zw_xhi_1 = wrf_zw_x(ph_xhi_n, ph_xhi_np1, mu_xhi_n, mu_xhi_np1, ixh, jxm, k+1);
                Real zw_ylo_0 = wrf_zw_y(ph_ylo_n, ph_ylo_np1, mu_ylo_n, mu_ylo_np1, iym, jyl, k  );
                Real zw_ylo_1 = wrf_zw_y(ph_ylo_n, ph_ylo_np1, mu_ylo_n, mu_ylo_np1, iym, jyl, k+1);
                Real zw_yhi_0 = wrf_zw_y(ph_yhi_n, ph_yhi_np1, mu_yhi_n, mu_yhi_np1, iym, jyh, k  );
                Real zw_yhi_1 = wrf_zw_y(ph_yhi_n, ph_yhi_np1, mu_yhi_n, mu_yhi_np1, iym, jyh, k+1);

                Real z_erf_0_x = znd(ixl,jxm,k);
                Real z_erf_1_x = znd(ixl,jxm,k+1);
                Real z_erf_0_xh = znd(ixh,jxm,k);
                Real z_erf_1_xh = znd(ixh,jxm,k+1);
                Real z_erf_0_y = znd(iym,jyl,k);
                Real z_erf_1_y = znd(iym,jyl,k+1);
                Real z_erf_0_yh = znd(iym,jyh,k);
                Real z_erf_1_yh = znd(iym,jyh,k+1);
                Real z_erf_cc_x  = myhalf * (z_erf_0_x  + z_erf_1_x);
                Real z_erf_cc_xh = myhalf * (z_erf_0_xh + z_erf_1_xh);
                Real z_erf_cc_y  = myhalf * (z_erf_0_y  + z_erf_1_y);
                Real z_erf_cc_yh = myhalf * (z_erf_0_yh + z_erf_1_yh);
                Real dz_nd_x  = z_erf_1_x  - z_erf_0_x;
                Real dz_nd_xh = z_erf_1_xh - z_erf_0_xh;
                Real dz_nd_y  = z_erf_1_y  - z_erf_0_y;
                Real dz_nd_yh = z_erf_1_yh - z_erf_0_yh;

                Print() << "  k=" << k
                        << " xlo dz(wrf-erf)=(" << (zw_xlo_0-z_erf_0_x) << "," << (zw_xlo_1-z_erf_1_x) << ")"
                        << " xhi=(" << (zw_xhi_0-z_erf_0_xh) << "," << (zw_xhi_1-z_erf_1_xh) << ")"
                        << " ylo=(" << (zw_ylo_0-z_erf_0_y) << "," << (zw_ylo_1-z_erf_1_y) << ")"
                        << " yhi=(" << (zw_yhi_0-z_erf_0_yh) << "," << (zw_yhi_1-z_erf_1_yh) << ")"
                        << std::endl;
                Print() << "      cc-diff wrf(k)-zcc: "
                        << "xlo=" << (zw_xlo_0-z_erf_cc_x)
                        << " xhi=" << (zw_xhi_0-z_erf_cc_xh)
                        << " ylo=" << (zw_ylo_0-z_erf_cc_y)
                        << " yhi=" << (zw_yhi_0-z_erf_cc_yh)
                        << " | dz_nd: xlo=" << dz_nd_x
                        << " xhi=" << dz_nd_xh
                        << " ylo=" << dz_nd_y
                        << " yhi=" << dz_nd_yh
                        << std::endl;
            }
            printed = true;
        }
    }

    if (dbg_realbdy_zstats_diag && have_wrfbdy_ph &&
        (std::abs(alpha) < 1.0e-12) && (n_time == 0)) {
        struct ZStats {
            Long n = 0;
            Real minv = std::numeric_limits<Real>::max();
            Real maxv = -std::numeric_limits<Real>::max();
            Real sum = 0.0_rt;
            Real sumabs = 0.0_rt;
            Long nbad = 0;
            AMREX_FORCE_INLINE void add(Real v) noexcept {
                if (!std::isfinite(static_cast<double>(v))) { ++nbad; return; }
                ++n;
                minv = std::min(minv, v);
                maxv = std::max(maxv, v);
                sum += v;
                sumabs += std::abs(v);
            }
        };
        auto reduce_stats = [] (ZStats& s) {
            ParallelDescriptor::ReduceLongSum(s.n);
            ParallelDescriptor::ReduceLongSum(s.nbad);
            ParallelDescriptor::ReduceRealMin(s.minv);
            ParallelDescriptor::ReduceRealMax(s.maxv);
            ParallelDescriptor::ReduceRealSum(s.sum);
            ParallelDescriptor::ReduceRealSum(s.sumabs);
        };
        auto print_stats = [] (const char* name, const ZStats& s) {
            if (s.n == 0) {
                Print() << "  " << name << ": n=0 nbad=" << s.nbad << "\n";
                return;
            }
            Print() << "  " << name
                    << ": n=" << s.n
                    << " nbad=" << s.nbad
                    << " min=" << s.minv
                    << " max=" << s.maxv
                    << " mean=" << (s.sum / static_cast<Real>(s.n))
                    << " absmean=" << (s.sumabs / static_cast<Real>(s.n))
                    << "\n";
        };

        const auto& ph_xlo_n = bdy_data_xlo[n_time][WRFBdyVars::PH].const_array();
        const auto& ph_xhi_n = bdy_data_xhi[n_time][WRFBdyVars::PH].const_array();
        const auto& ph_ylo_n = bdy_data_ylo[n_time][WRFBdyVars::PH].const_array();
        const auto& ph_yhi_n = bdy_data_yhi[n_time][WRFBdyVars::PH].const_array();
        const auto& mu_xlo_n = bdy_data_xlo[n_time][WRFBdyVars::MU].const_array();
        const auto& mu_xhi_n = bdy_data_xhi[n_time][WRFBdyVars::MU].const_array();
        const auto& mu_ylo_n = bdy_data_ylo[n_time][WRFBdyVars::MU].const_array();
        const auto& mu_yhi_n = bdy_data_yhi[n_time][WRFBdyVars::MU].const_array();

        ZStats z_erf_int, z_erf_strip;
        ZStats z_bdy_xlo, z_bdy_xhi, z_bdy_ylo, z_bdy_yhi;
        ZStats dz_xlo, dz_xhi, dz_ylo, dz_yhi;
        const std::array<int,7> k_levels = {0,5,10,20,40,80,119};
        std::array<ZStats,7> dz_xlo_k, dz_xhi_k, dz_ylo_k, dz_yhi_k;
        int bad_print_cap = 8;
        int bad_print_xlo = 0, bad_print_xhi = 0, bad_print_ylo = 0, bad_print_yhi = 0;

        const Box dom = geom.Domain();
        const int ilo = dom.smallEnd(0), ihi = dom.bigEnd(0);
        const int jlo = dom.smallEnd(1), jhi = dom.bigEnd(1);
        const int knd_hi = dom.bigEnd(2) + 1;
        const int k_bdy_max = dom.bigEnd(2); // exclude top nodal level where xmu_mult_f can be zero
        const Real ginv = Real(1.0) / Real(9.81);

        for (MFIter mfi(z_phys_nd); mfi.isValid(); ++mfi) {
            const auto znd = z_phys_nd.const_array(mfi);
            const auto phb = mf_PHB_wrfin.const_array(mfi);
            const auto mub = mf_MUB.const_array(mfi);
            const auto c1f = mf_C1F.const_array(mfi);
            const auto c2f = mf_C2F.const_array(mfi);
            const int kmax_cf = mf_C1F[mfi].box().bigEnd(2);
            const Box bx = mfi.validbox() & amrex::surroundingNodes(dom);
            if (!bx.ok()) continue;

            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
                for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                    for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                        const bool in_xlo = (width > 0) && (i <= ilo + width - 1);
                        const bool in_xhi = (width > 0) && (i >= ihi - width + 1);
                        const bool in_ylo = (width > 0) && (j <= jlo + width - 1);
                        const bool in_yhi = (width > 0) && (j >= jhi - width + 1);
                        const bool in_strip = in_xlo || in_xhi || in_ylo || in_yhi;
                        if (in_strip) z_erf_strip.add(znd(i,j,k));
                        else          z_erf_int.add(znd(i,j,k));
                    }
                }
            }

            // Face xlo/xhi: index with global i,j in boundary FABs, b implicit by i offset
            for (int b = 0; b < width; ++b) {
                int ixlo = ilo + b;
                int ixhi = ihi - b;
                for (int j = std::max(jlo, bx.smallEnd(1)); j <= std::min(jhi, bx.bigEnd(1)); ++j) {
                    for (int k = std::max(0, bx.smallEnd(2)); k <= std::min(k_bdy_max, bx.bigEnd(2)); ++k) {
                        if (ixlo >= bx.smallEnd(0) && ixlo <= bx.bigEnd(0)) {
                            const int kcf = std::min(k, kmax_cf);
                            const Real mu_d_l = mu_xlo_n(ixlo,j,0,0) + mub(ixlo,j,0,0);
                            const Real xmul_l = c1f(0,0,kcf,0) * mu_d_l + c2f(0,0,kcf,0);
                            const Real zbl = (ph_xlo_n(ixlo,j,k,0) / xmul_l + phb(ixlo,j,k,0)) * ginv;
                            if ((!std::isfinite(static_cast<double>(zbl)) || !std::isfinite(static_cast<double>(xmul_l)) || std::abs(xmul_l) < 1.0e-14) &&
                                bad_print_xlo < bad_print_cap && ParallelDescriptor::IOProcessor()) {
                                ++bad_print_xlo;
                                Print() << "[DBG_ZBAD xlo] i=" << ixlo << " j=" << j << " k=" << k
                                        << " PH_B=" << ph_xlo_n(ixlo,j,k,0)
                                        << " PHB=" << phb(ixlo,j,k,0)
                                        << " MU_B=" << mu_xlo_n(ixlo,j,0,0)
                                        << " MUB=" << mub(ixlo,j,0,0)
                                        << " xmu_mult=" << xmul_l
                                        << " z_bdy=" << zbl << "\n";
                            }
                            const Real dd = zbl - znd(ixlo,j,k);
                            z_bdy_xlo.add(zbl); dz_xlo.add(dd);
                            for (int ik=0; ik<7; ++ik) if (k == k_levels[ik]) dz_xlo_k[ik].add(dd);
                        }
                        if (ixhi >= bx.smallEnd(0) && ixhi <= bx.bigEnd(0)) {
                            const int kcf = std::min(k, kmax_cf);
                            const Real mu_d_h = mu_xhi_n(ixhi,j,0,0) + mub(ixhi,j,0,0);
                            const Real xmul_h = c1f(0,0,kcf,0) * mu_d_h + c2f(0,0,kcf,0);
                            const Real zbh = (ph_xhi_n(ixhi,j,k,0) / xmul_h + phb(ixhi,j,k,0)) * ginv;
                            if ((!std::isfinite(static_cast<double>(zbh)) || !std::isfinite(static_cast<double>(xmul_h)) || std::abs(xmul_h) < 1.0e-14) &&
                                bad_print_xhi < bad_print_cap && ParallelDescriptor::IOProcessor()) {
                                ++bad_print_xhi;
                                Print() << "[DBG_ZBAD xhi] i=" << ixhi << " j=" << j << " k=" << k
                                        << " PH_B=" << ph_xhi_n(ixhi,j,k,0)
                                        << " PHB=" << phb(ixhi,j,k,0)
                                        << " MU_B=" << mu_xhi_n(ixhi,j,0,0)
                                        << " MUB=" << mub(ixhi,j,0,0)
                                        << " xmu_mult=" << xmul_h
                                        << " z_bdy=" << zbh << "\n";
                            }
                            const Real dd = zbh - znd(ixhi,j,k);
                            z_bdy_xhi.add(zbh); dz_xhi.add(dd);
                            for (int ik=0; ik<7; ++ik) if (k == k_levels[ik]) dz_xhi_k[ik].add(dd);
                        }
                    }
                }
            }

            // Face ylo/yhi
            for (int b = 0; b < width; ++b) {
                int jylo = jlo + b;
                int jyhi = jhi - b;
                for (int i = std::max(ilo, bx.smallEnd(0)); i <= std::min(ihi, bx.bigEnd(0)); ++i) {
                    for (int k = std::max(0, bx.smallEnd(2)); k <= std::min(k_bdy_max, bx.bigEnd(2)); ++k) {
                        if (jylo >= bx.smallEnd(1) && jylo <= bx.bigEnd(1)) {
                            const int kcf = std::min(k, kmax_cf);
                            const Real mu_d_l = mu_ylo_n(i,jylo,0,0) + mub(i,jylo,0,0);
                            const Real xmul_l = c1f(0,0,kcf,0) * mu_d_l + c2f(0,0,kcf,0);
                            const Real zbl = (ph_ylo_n(i,jylo,k,0) / xmul_l + phb(i,jylo,k,0)) * ginv;
                            if ((!std::isfinite(static_cast<double>(zbl)) || !std::isfinite(static_cast<double>(xmul_l)) || std::abs(xmul_l) < 1.0e-14) &&
                                bad_print_ylo < bad_print_cap && ParallelDescriptor::IOProcessor()) {
                                ++bad_print_ylo;
                                Print() << "[DBG_ZBAD ylo] i=" << i << " j=" << jylo << " k=" << k
                                        << " PH_B=" << ph_ylo_n(i,jylo,k,0)
                                        << " PHB=" << phb(i,jylo,k,0)
                                        << " MU_B=" << mu_ylo_n(i,jylo,0,0)
                                        << " MUB=" << mub(i,jylo,0,0)
                                        << " xmu_mult=" << xmul_l
                                        << " z_bdy=" << zbl << "\n";
                            }
                            const Real dd = zbl - znd(i,jylo,k);
                            z_bdy_ylo.add(zbl); dz_ylo.add(dd);
                            for (int ik=0; ik<7; ++ik) if (k == k_levels[ik]) dz_ylo_k[ik].add(dd);
                        }
                        if (jyhi >= bx.smallEnd(1) && jyhi <= bx.bigEnd(1)) {
                            const int kcf = std::min(k, kmax_cf);
                            const Real mu_d_h = mu_yhi_n(i,jyhi,0,0) + mub(i,jyhi,0,0);
                            const Real xmul_h = c1f(0,0,kcf,0) * mu_d_h + c2f(0,0,kcf,0);
                            const Real zbh = (ph_yhi_n(i,jyhi,k,0) / xmul_h + phb(i,jyhi,k,0)) * ginv;
                            if ((!std::isfinite(static_cast<double>(zbh)) || !std::isfinite(static_cast<double>(xmul_h)) || std::abs(xmul_h) < 1.0e-14) &&
                                bad_print_yhi < bad_print_cap && ParallelDescriptor::IOProcessor()) {
                                ++bad_print_yhi;
                                Print() << "[DBG_ZBAD yhi] i=" << i << " j=" << jyhi << " k=" << k
                                        << " PH_B=" << ph_yhi_n(i,jyhi,k,0)
                                        << " PHB=" << phb(i,jyhi,k,0)
                                        << " MU_B=" << mu_yhi_n(i,jyhi,0,0)
                                        << " MUB=" << mub(i,jyhi,0,0)
                                        << " xmu_mult=" << xmul_h
                                        << " z_bdy=" << zbh << "\n";
                            }
                            const Real dd = zbh - znd(i,jyhi,k);
                            z_bdy_yhi.add(zbh); dz_yhi.add(dd);
                            for (int ik=0; ik<7; ++ik) if (k == k_levels[ik]) dz_yhi_k[ik].add(dd);
                        }
                    }
                }
            }
        }

        reduce_stats(z_erf_int); reduce_stats(z_erf_strip);
        reduce_stats(z_bdy_xlo); reduce_stats(z_bdy_xhi); reduce_stats(z_bdy_ylo); reduce_stats(z_bdy_yhi);
        reduce_stats(dz_xlo); reduce_stats(dz_xhi); reduce_stats(dz_ylo); reduce_stats(dz_yhi);
        for (int ik=0; ik<7; ++ik) {
            reduce_stats(dz_xlo_k[ik]); reduce_stats(dz_xhi_k[ik]);
            reduce_stats(dz_ylo_k[ik]); reduce_stats(dz_yhi_k[ik]);
        }

        if (ParallelDescriptor::IOProcessor()) {
            Print() << "[DBG_ZSTATS init-vs-bdy] width=" << width << " n_time=" << n_time << " alpha=" << alpha << "\n";
            print_stats("z_erf_interior", z_erf_int);
            print_stats("z_erf_strip",    z_erf_strip);
            print_stats("z_bdy_xlo",      z_bdy_xlo);
            print_stats("z_bdy_xhi",      z_bdy_xhi);
            print_stats("z_bdy_ylo",      z_bdy_ylo);
            print_stats("z_bdy_yhi",      z_bdy_yhi);
            print_stats("dz_xlo(bdy-erf)", dz_xlo);
            print_stats("dz_xhi(bdy-erf)", dz_xhi);
            print_stats("dz_ylo(bdy-erf)", dz_ylo);
            print_stats("dz_yhi(bdy-erf)", dz_yhi);
            for (int ik=0; ik<7; ++ik) {
                std::string hdr = "k=" + std::to_string(k_levels[ik]);
                Print() << "  [DZ_BY_K " << hdr << "]\n";
                print_stats("    xlo", dz_xlo_k[ik]);
                print_stats("    xhi", dz_xhi_k[ik]);
                print_stats("    ylo", dz_ylo_k[ik]);
                print_stats("    yhi", dz_yhi_k[ik]);
            }
        }
    }
#endif

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

        MultiFab mf_theta_dbg;
        Array4<Real> theta_dbg_arr;
        const bool do_dump_theta_for_var = do_dump_theta_this_rhs && (ivar == ivarU || ivar == ivarV || ivar == ivarT);
        if (do_dump_theta_for_var) {
            mf_theta_dbg.define(S_cur_data[ivar_idx].boxArray(),
                                S_cur_data[ivar_idx].DistributionMap(),
                                3, 0);
            mf_theta_dbg.setVal(std::numeric_limits<Real>::quiet_NaN());
        }

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
            Array4<Real const> state_arr = S_cur_data[ivar_idx].const_array(mfi);
            Array4<Real const> state_cons_arr = S_cur_data[ivar_map[ivar]].const_array(mfi);
            if (do_dump_theta_for_var) {
                theta_dbg_arr = mf_theta_dbg.array(mfi);
            }
            Array4<Real const> mub_arr = mf_MUB.const_array(mfi);
            Array4<Real const> c1h_arr = mf_C1H.const_array(mfi);
            Array4<Real const> c2h_arr = mf_C2H.const_array(mfi);
            Array4<Real const> c1f_arr = mf_C1F.const_array(mfi);
            Array4<Real const> c2f_arr = mf_C2F.const_array(mfi);
            Array4<Real const> dnw_arr = mf_DNW.const_array(mfi);
            Array4<Real const> phb_arr = mf_PHB_wrfin.const_array(mfi);
            Array4<Real const> zcc_arr = z_phys_cc.const_array(mfi);

            const auto& bdatxlo_mu_n   = bdy_data_xlo[n_time   ][WRFBdyVars::MU].const_array();
            const auto& bdatxlo_mu_np1 = bdy_data_xlo[n_time_p1][WRFBdyVars::MU].const_array();
            const auto& bdatxhi_mu_n   = bdy_data_xhi[n_time   ][WRFBdyVars::MU].const_array();
            const auto& bdatxhi_mu_np1 = bdy_data_xhi[n_time_p1][WRFBdyVars::MU].const_array();
            const auto& bdatylo_mu_n   = bdy_data_ylo[n_time   ][WRFBdyVars::MU].const_array();
            const auto& bdatylo_mu_np1 = bdy_data_ylo[n_time_p1][WRFBdyVars::MU].const_array();
            const auto& bdatyhi_mu_n   = bdy_data_yhi[n_time   ][WRFBdyVars::MU].const_array();
            const auto& bdatyhi_mu_np1 = bdy_data_yhi[n_time_p1][WRFBdyVars::MU].const_array();
            const auto& bdatxlo_ph_n   = have_wrfbdy_ph ? bdy_data_xlo[n_time   ][WRFBdyVars::PH].const_array() : bdy_data_xlo[n_time][WRFBdyVars::T].const_array();
            const auto& bdatxlo_ph_np1 = have_wrfbdy_ph ? bdy_data_xlo[n_time_p1][WRFBdyVars::PH].const_array() : bdy_data_xlo[n_time_p1][WRFBdyVars::T].const_array();
            const auto& bdatxhi_ph_n   = have_wrfbdy_ph ? bdy_data_xhi[n_time   ][WRFBdyVars::PH].const_array() : bdy_data_xhi[n_time][WRFBdyVars::T].const_array();
            const auto& bdatxhi_ph_np1 = have_wrfbdy_ph ? bdy_data_xhi[n_time_p1][WRFBdyVars::PH].const_array() : bdy_data_xhi[n_time_p1][WRFBdyVars::T].const_array();
            const auto& bdatylo_ph_n   = have_wrfbdy_ph ? bdy_data_ylo[n_time   ][WRFBdyVars::PH].const_array() : bdy_data_ylo[n_time][WRFBdyVars::T].const_array();
            const auto& bdatylo_ph_np1 = have_wrfbdy_ph ? bdy_data_ylo[n_time_p1][WRFBdyVars::PH].const_array() : bdy_data_ylo[n_time_p1][WRFBdyVars::T].const_array();
            const auto& bdatyhi_ph_n   = have_wrfbdy_ph ? bdy_data_yhi[n_time   ][WRFBdyVars::PH].const_array() : bdy_data_yhi[n_time][WRFBdyVars::T].const_array();
            const auto& bdatyhi_ph_np1 = have_wrfbdy_ph ? bdy_data_yhi[n_time_p1][WRFBdyVars::PH].const_array() : bdy_data_yhi[n_time_p1][WRFBdyVars::T].const_array();
            const int kmax_ph_xlo = have_wrfbdy_ph ? bdy_data_xlo[n_time][WRFBdyVars::PH].box().bigEnd(2) : dom_hi.z;
            const int kmax_ph_xhi = have_wrfbdy_ph ? bdy_data_xhi[n_time][WRFBdyVars::PH].box().bigEnd(2) : dom_hi.z;
            const int kmax_ph_ylo = have_wrfbdy_ph ? bdy_data_ylo[n_time][WRFBdyVars::PH].box().bigEnd(2) : dom_hi.z;
            const int kmax_ph_yhi = have_wrfbdy_ph ? bdy_data_yhi[n_time][WRFBdyVars::PH].box().bigEnd(2) : dom_hi.z;
            const int kmax_t_xlo = bdy_data_xlo[n_time][ivar].box().bigEnd(2);
            const int kmax_t_xhi = bdy_data_xhi[n_time][ivar].box().bigEnd(2);
            const int kmax_t_ylo = bdy_data_ylo[n_time][ivar].box().bigEnd(2);
            const int kmax_t_yhi = bdy_data_yhi[n_time][ivar].box().bigEnd(2);
            auto z_tgt_loc = [=] AMREX_GPU_DEVICE (int ivar_loc, int i_loc, int j_loc, int k_loc) noexcept -> Real {
                return realbdy_zcc_to_loc(zcc_arr, dom_cc_lo, dom_cc_hi, i_loc, j_loc, k_loc, ivar_loc, ivarU, ivarV);
            };
            if (!dbg_realbdy_boxes_print_once && ParallelDescriptor::IOProcessor()) {
                const Box bx_xlo_v = bdy_data_xlo[n_time][ivar].box();
                const Box bx_xhi_v = bdy_data_xhi[n_time][ivar].box();
                const Box bx_ylo_v = bdy_data_ylo[n_time][ivar].box();
                const Box bx_yhi_v = bdy_data_yhi[n_time][ivar].box();
                const Box bx_xhi_mu = bdy_data_xhi[n_time][WRFBdyVars::MU].box();
                const Box bx_yhi_mu = bdy_data_yhi[n_time][WRFBdyVars::MU].box();
                const Box bx_xhi_ph = have_wrfbdy_ph ? bdy_data_xhi[n_time][WRFBdyVars::PH].box() : Box();
                const Box bx_yhi_ph = have_wrfbdy_ph ? bdy_data_yhi[n_time][WRFBdyVars::PH].box() : Box();
                std::string vname = (ivar==ivarU) ? "U" : ((ivar==ivarV) ? "V" : "T");
                Print() << "[DBG_REALBDY_BOXES " << vname << "] "
                        << "xlo=" << bx_xlo_v << " xhi=" << bx_xhi_v
                        << " ylo=" << bx_ylo_v << " yhi=" << bx_yhi_v
                        << " | MU xhi=" << bx_xhi_mu << " yhi=" << bx_yhi_mu;
                if (have_wrfbdy_ph) {
                    Print() << " | PH xhi=" << bx_xhi_ph << " yhi=" << bx_yhi_ph;
                }
                Print() << "\n";
                if (ivar == ivarT) dbg_realbdy_boxes_print_once = true;
            }

            // Limiting offset
            int offset = width - 1;

            // Populate with interpolation (protect from ghost cells)
            ParallelFor(tbx_xlo, tbx_xhi,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int ii = std::max(i , dom_lo.x);
                    ii = std::min(ii, dom_lo.x + offset + ((ivar==ivarU) ? 1 : 0));
                int jj = std::max(j , dom_lo.y);
                    jj = std::min(jj, dom_hi.y + ((ivar==ivarV) ? 1 : 0));

                Real rho_interp;
                if (!use_wrf_rho_interp) {
                    if (ivar==ivarU) {
                        rho_interp = myhalf * ( r_arr(i-1,j  ,k) + r_arr(i,j,k) );
                    } else if (ivar==ivarV) {
                        rho_interp = myhalf * ( r_arr(i  ,j-1,k) + r_arr(i,j,k) );
                    } else {
                        rho_interp = r_arr(i,j,k);
                    }
                } else {
                    auto rho_cc = [&] AMREX_GPU_DEVICE (int ic, int jc, int kc) noexcept -> Real {
                        kc = amrex::max(0, amrex::min(kc, kmax_ph_xlo-1));
                        Real mu_t = oma * bdatxlo_mu_n(ic,jc,0,0) + alpha * bdatxlo_mu_np1(ic,jc,0,0) + mub_arr(ic,jc,0);
                        Real xmu_mult_h = c1h_arr(0,0,kc) * mu_t + c2h_arr(0,0,kc);
                        Real dpd = xmu_mult_h * amrex::Math::abs(dnw_arr(0,0,kc));
                        Real phi_k   = oma * bdatxlo_ph_n(ic,jc,kc,0)   + alpha * bdatxlo_ph_np1(ic,jc,kc,0)   + phb_arr(ic,jc,kc);
                        Real phi_kp1 = oma * bdatxlo_ph_n(ic,jc,kc+1,0) + alpha * bdatxlo_ph_np1(ic,jc,kc+1,0) + phb_arr(ic,jc,kc+1);
                        Real dphi = amrex::max(phi_kp1 - phi_k, Real(1.0e-12));
                        return dpd / dphi;
                    };
                    if (ivar==ivarU) {
                        rho_interp = myhalf * ( rho_cc(i-1,j,k) + rho_cc(i,j,k) );
                    } else if (ivar==ivarV) {
                        rho_interp = myhalf * ( rho_cc(i,j-1,k) + rho_cc(i,j,k) );
                    } else {
                        rho_interp = rho_cc(i,j,k);
                    }
                }

                if (bdatxlo) {
                    int ii2 = std::min(std::max(i , dom_cc_lo.x), dom_cc_hi.x);
                    int jj2 = std::min(std::max(j , dom_cc_lo.y), dom_cc_hi.y);
                    arr_xlo(i,j,k) = rho_interp * bdatxlo(ii2,jj2,k,bdy_comp);
                } else {
                    Real theta_base = oma * bdatxlo_n(ii,jj,k,0) + alpha * bdatxlo_np1(ii,jj,k,0);
                    Real theta_t = theta_base;
                    if (use_theta_vertical_remap) {
                        const int ksrc_max = amrex::min(kmax_t_xlo, kmax_ph_xlo-1);
                        auto z_target_cc = [&](int ic, int jc, int kk) AMREX_GPU_DEVICE noexcept -> Real {
                            Real mu_t = oma * bdatxlo_mu_n(ic,jc,0,0) + alpha * bdatxlo_mu_np1(ic,jc,0,0) + mub_arr(ic,jc,0);
                            Real xmu_f = c1f_arr(0,0,kk) * mu_t + c2f_arr(0,0,kk);
                            Real ph_t = oma * bdatxlo_ph_n(ic,jc,kk,0) + alpha * bdatxlo_ph_np1(ic,jc,kk,0);
                            return (ph_t / xmu_f + phb_arr(ic,jc,kk)) / CONST_GRAV;
                        };

                        if (realbdy_vertical_remap_mode == 2 && (ivar == ivarU || ivar == ivarV) && ksrc_max > 0) {
                            // Path B: remap target profile at adjacent cc columns, then average remapped
                            // primitive values to the face before converting to conserved form.
                            auto src_target_cc = [&](int icc, int jcc, int kk) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int i0 = amrex::min(amrex::max(icc,   dom_lo.x), dom_hi.x);
                                    int i1 = amrex::min(amrex::max(icc+1, dom_lo.x), dom_hi.x);
                                    int jc = amrex::min(amrex::max(jcc,   dom_lo.y), dom_hi.y);
                                    Real u0 = oma * bdatxlo_n(i0,jc,kk,0) + alpha * bdatxlo_np1(i0,jc,kk,0);
                                    Real u1 = oma * bdatxlo_n(i1,jc,kk,0) + alpha * bdatxlo_np1(i1,jc,kk,0);
                                    return Real(0.5) * (u0 + u1);
                                } else {
                                    int ic = amrex::min(amrex::max(icc,   dom_lo.x), dom_hi.x);
                                    int j0 = amrex::min(amrex::max(jcc,   dom_lo.y), dom_hi.y);
                                    int j1 = amrex::min(amrex::max(jcc+1, dom_lo.y), dom_hi.y);
                                    Real v0 = oma * bdatxlo_n(ic,j0,kk,0) + alpha * bdatxlo_np1(ic,j0,kk,0);
                                    Real v1 = oma * bdatxlo_n(ic,j1,kk,0) + alpha * bdatxlo_np1(ic,j1,kk,0);
                                    return Real(0.5) * (v0 + v1);
                                }
                            };
                            auto remap_target_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                const Real z_state_cc = zcc_arr(icc,jcc,k,0);
                                Real z0 = Real(0.5) * (z_target_cc(icc,jcc,0) + z_target_cc(icc,jcc,1));
                                Real f0 = src_target_cc(icc,jcc,0);
                                if (z_state_cc <= z0) { return f0; }
                                for (int kk = 0; kk < ksrc_max; ++kk) {
                                    Real zl = Real(0.5) * (z_target_cc(icc,jcc,kk) + z_target_cc(icc,jcc,kk+1));
                                    Real zh = Real(0.5) * (z_target_cc(icc,jcc,kk+1) + z_target_cc(icc,jcc,kk+2));
                                    if (z_state_cc <= zh || kk == ksrc_max-1) {
                                        Real fl = src_target_cc(icc,jcc,kk);
                                        Real fh = src_target_cc(icc,jcc,kk+1);
                                        Real dz = amrex::max(zh-zl, Real(1.e-12));
                                        Real lam = (z_state_cc - zl) / dz;
                                        lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                        return (Real(1.0)-lam)*fl + lam*fh;
                                    }
                                }
                                return src_target_cc(icc,jcc,ksrc_max);
                            };
                            if (ivar == ivarU) {
                                int iL = amrex::min(amrex::max(i-1, dom_cc_lo.x), dom_cc_hi.x);
                                int iR = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                theta_t = Real(0.5) * (remap_target_cc(iL,jC) + remap_target_cc(iR,jC));
                            } else {
                                int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                int jL = amrex::min(amrex::max(j-1, dom_cc_lo.y), dom_cc_hi.y);
                                int jR = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                theta_t = Real(0.5) * (remap_target_cc(iC,jL) + remap_target_cc(iC,jR));
                            }
                        } else if ((realbdy_vertical_remap_mode == 3 ||
                                    realbdy_vertical_remap_mode == 4) &&
                                   (ivar == ivarU || ivar == ivarV) && ksrc_max > 0) {
                            // Path C/D: remap target at cc columns and form deltas at cc.
                            // C: average primitive delta to face.
                            // D: average conserved delta (rho_cc * primitive_delta) to face.
                            auto src_target_cc = [&](int icc, int jcc, int kk) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int i0 = amrex::min(amrex::max(icc,   dom_lo.x), dom_hi.x);
                                    int i1 = amrex::min(amrex::max(icc+1, dom_lo.x), dom_hi.x);
                                    int jc = amrex::min(amrex::max(jcc,   dom_lo.y), dom_hi.y);
                                    Real u0 = oma * bdatxlo_n(i0,jc,kk,0) + alpha * bdatxlo_np1(i0,jc,kk,0);
                                    Real u1 = oma * bdatxlo_n(i1,jc,kk,0) + alpha * bdatxlo_np1(i1,jc,kk,0);
                                    return Real(0.5) * (u0 + u1);
                                } else {
                                    int ic = amrex::min(amrex::max(icc,   dom_lo.x), dom_hi.x);
                                    int j0 = amrex::min(amrex::max(jcc,   dom_lo.y), dom_hi.y);
                                    int j1 = amrex::min(amrex::max(jcc+1, dom_lo.y), dom_hi.y);
                                    Real v0 = oma * bdatxlo_n(ic,j0,kk,0) + alpha * bdatxlo_np1(ic,j0,kk,0);
                                    Real v1 = oma * bdatxlo_n(ic,j1,kk,0) + alpha * bdatxlo_np1(ic,j1,kk,0);
                                    return Real(0.5) * (v0 + v1);
                                }
                            };
                            auto remap_target_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                const Real z_state_cc = zcc_arr(icc,jcc,k,0);
                                Real z0 = Real(0.5) * (z_target_cc(icc,jcc,0) + z_target_cc(icc,jcc,1));
                                Real f0 = src_target_cc(icc,jcc,0);
                                if (z_state_cc <= z0) { return f0; }
                                for (int kk = 0; kk < ksrc_max; ++kk) {
                                    Real zl = Real(0.5) * (z_target_cc(icc,jcc,kk) + z_target_cc(icc,jcc,kk+1));
                                    Real zh = Real(0.5) * (z_target_cc(icc,jcc,kk+1) + z_target_cc(icc,jcc,kk+2));
                                    if (z_state_cc <= zh || kk == ksrc_max-1) {
                                        Real fl = src_target_cc(icc,jcc,kk);
                                        Real fh = src_target_cc(icc,jcc,kk+1);
                                        Real dz = amrex::max(zh-zl, Real(1.e-12));
                                        Real lam = (z_state_cc - zl) / dz;
                                        lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                        return (Real(1.0)-lam)*fl + lam*fh;
                                    }
                                }
                                return src_target_cc(icc,jcc,ksrc_max);
                            };
                            auto state_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int i0 = amrex::min(amrex::max(icc,   dom_cc_lo.x), dom_cc_hi.x);
                                    int i1 = amrex::min(amrex::max(icc+1, dom_cc_lo.x), dom_cc_hi.x+1);
                                    int jc = amrex::min(amrex::max(jcc,   dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (state_arr(i0,jc,k,0) + state_arr(i1,jc,k,0));
                                } else {
                                    int ic = amrex::min(amrex::max(icc,   dom_cc_lo.x), dom_cc_hi.x);
                                    int j0 = amrex::min(amrex::max(jcc,   dom_cc_lo.y), dom_cc_hi.y);
                                    int j1 = amrex::min(amrex::max(jcc+1, dom_cc_lo.y), dom_cc_hi.y+1);
                                    return Real(0.5) * (state_arr(ic,j0,k,0) + state_arr(ic,j1,k,0));
                                }
                            };
                            auto rho_state_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                int kc = amrex::max(0, amrex::min(k, kmax_ph_xlo-1));
                                Real mu_t = oma * bdatxlo_mu_n(icc,jcc,0,0) + alpha * bdatxlo_mu_np1(icc,jcc,0,0) + mub_arr(icc,jcc,0);
                                Real xmu_mult_h = c1h_arr(0,0,kc) * mu_t + c2h_arr(0,0,kc);
                                Real dpd = xmu_mult_h * amrex::Math::abs(dnw_arr(0,0,kc));
                                Real phi_k   = oma * bdatxlo_ph_n(icc,jcc,kc,0)   + alpha * bdatxlo_ph_np1(icc,jcc,kc,0)   + phb_arr(icc,jcc,kc);
                                Real phi_kp1 = oma * bdatxlo_ph_n(icc,jcc,kc+1,0) + alpha * bdatxlo_ph_np1(icc,jcc,kc+1,0) + phb_arr(icc,jcc,kc+1);
                                Real dphi = amrex::max(phi_kp1 - phi_k, Real(1.0e-12));
                                return dpd / dphi;
                            };
                            if (ivar == ivarU) {
                                int iL = amrex::min(amrex::max(i-1, dom_cc_lo.x), dom_cc_hi.x);
                                int iR = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                Real deltaL = remap_target_cc(iL,jC) - state_cc(iL,jC);
                                Real deltaR = remap_target_cc(iR,jC) - state_cc(iR,jC);
                                if (realbdy_vertical_remap_mode == 3) {
                                    Real delta_face = Real(0.5) * (deltaL + deltaR);
                                    theta_t = state_arr(i,j,k,0) + delta_face;
                                } else {
                                    Real dconsL = rho_state_cc(iL,jC) * deltaL;
                                    Real dconsR = rho_state_cc(iR,jC) * deltaR;
                                    Real dcons_face = Real(0.5) * (dconsL + dconsR);
                                    Real cons_target = state_cons_arr(i,j,k,0) + dcons_face;
                                    Real rho_safe = (amrex::Math::abs(rho_interp) > Real(1.0e-12)) ? rho_interp : Real(1.0e-12);
                                    theta_t = cons_target / rho_safe;
                                }
                            } else {
                                int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                int jL = amrex::min(amrex::max(j-1, dom_cc_lo.y), dom_cc_hi.y);
                                int jR = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                Real deltaL = remap_target_cc(iC,jL) - state_cc(iC,jL);
                                Real deltaR = remap_target_cc(iC,jR) - state_cc(iC,jR);
                                if (realbdy_vertical_remap_mode == 3) {
                                    Real delta_face = Real(0.5) * (deltaL + deltaR);
                                    theta_t = state_arr(i,j,k,0) + delta_face;
                                } else {
                                    Real dconsL = rho_state_cc(iC,jL) * deltaL;
                                    Real dconsR = rho_state_cc(iC,jR) * deltaR;
                                    Real dcons_face = Real(0.5) * (dconsL + dconsR);
                                    Real cons_target = state_cons_arr(i,j,k,0) + dcons_face;
                                    Real rho_safe = (amrex::Math::abs(rho_interp) > Real(1.0e-12)) ? rho_interp : Real(1.0e-12);
                                    theta_t = cons_target / rho_safe;
                                }
                            }
                        } else {
                            // Path A-2: remap target profile directly at the state variable location.
                            const Real z_state = z_tgt_loc(ivar, i, j, k);
                            auto z_target = [&](int kk) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int iL = amrex::min(amrex::max(i-1, dom_cc_lo.x), dom_cc_hi.x);
                                    int iR = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (z_target_cc(iL,jC,kk) + z_target_cc(iR,jC,kk));
                                } else if (ivar == ivarV) {
                                    int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jL = amrex::min(amrex::max(j-1, dom_cc_lo.y), dom_cc_hi.y);
                                    int jR = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (z_target_cc(iC,jL,kk) + z_target_cc(iC,jR,kk));
                                } else {
                                    int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return z_target_cc(iC,jC,kk);
                                }
                            };
                            if (ksrc_max > 0) {
                                Real z0 = Real(0.5) * (z_target(0) + z_target(1));
                                Real t0 = oma * bdatxlo_n(ii,jj,0,0) + alpha * bdatxlo_np1(ii,jj,0,0);
                                if (z_state <= z0) {
                                    theta_t = t0;
                                } else {
                                    bool found = false;
                                    for (int kk = 0; kk < ksrc_max; ++kk) {
                                        Real zl = Real(0.5) * (z_target(kk) + z_target(kk+1));
                                        Real zh = Real(0.5) * (z_target(kk+1) + z_target(kk+2));
                                        if (z_state <= zh || kk == ksrc_max-1) {
                                            Real tl = oma * bdatxlo_n(ii,jj,kk,0) + alpha * bdatxlo_np1(ii,jj,kk,0);
                                            Real th = oma * bdatxlo_n(ii,jj,kk+1,0) + alpha * bdatxlo_np1(ii,jj,kk+1,0);
                                            Real dz = amrex::max(zh-zl, Real(1.e-12));
                                            Real lam = (z_state - zl) / dz;
                                            lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                            theta_t = (Real(1.0)-lam)*tl + lam*th;
                                            found = true;
                                            break;
                                        }
                                    }
                                    if (!found) { theta_t = oma * bdatxlo_n(ii,jj,ksrc_max,0) + alpha * bdatxlo_np1(ii,jj,ksrc_max,0); }
                                }
                            }
                        }
                    }
                    arr_xlo(i,j,k) = rho_interp * theta_t;
                    if (do_dump_theta_for_var) {
                        theta_dbg_arr(i,j,k,0) = theta_base;
                        theta_dbg_arr(i,j,k,1) = theta_t;
                        theta_dbg_arr(i,j,k,2) = theta_t - theta_base;
                    }
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int ii = std::max(i , dom_hi.x - offset + ((ivar==ivarU) ? 1 : 0));
                    ii = std::min(ii, dom_hi.x + ((ivar==ivarU) ? 1 : 0));
                int jj = std::max(j , dom_lo.y);
                    jj = std::min(jj, dom_hi.y + ((ivar==ivarV) ? 1 : 0));

                Real rho_interp;
                if (!use_wrf_rho_interp) {
                    if (ivar==ivarU) {
                        rho_interp = myhalf * ( r_arr(i-1,j  ,k) + r_arr(i,j,k) );
                    } else if (ivar==ivarV) {
                        rho_interp = myhalf * ( r_arr(i  ,j-1,k) + r_arr(i,j,k) );
                    } else {
                        rho_interp = r_arr(i,j,k);
                    }
                } else {
                    auto rho_cc = [&] AMREX_GPU_DEVICE (int ic, int jc, int kc) noexcept -> Real {
                        kc = amrex::max(0, amrex::min(kc, kmax_ph_xhi-1));
                        Real mu_t = oma * bdatxhi_mu_n(ic,jc,0,0) + alpha * bdatxhi_mu_np1(ic,jc,0,0) + mub_arr(ic,jc,0);
                        Real xmu_mult_h = c1h_arr(0,0,kc) * mu_t + c2h_arr(0,0,kc);
                        Real dpd = xmu_mult_h * amrex::Math::abs(dnw_arr(0,0,kc));
                        Real phi_k   = oma * bdatxhi_ph_n(ic,jc,kc,0)   + alpha * bdatxhi_ph_np1(ic,jc,kc,0)   + phb_arr(ic,jc,kc);
                        Real phi_kp1 = oma * bdatxhi_ph_n(ic,jc,kc+1,0) + alpha * bdatxhi_ph_np1(ic,jc,kc+1,0) + phb_arr(ic,jc,kc+1);
                        Real dphi = amrex::max(phi_kp1 - phi_k, Real(1.0e-12));
                        return dpd / dphi;
                    };
                    if (ivar==ivarU) {
                        rho_interp = myhalf * ( rho_cc(i-1,j,k) + rho_cc(i,j,k) );
                    } else if (ivar==ivarV) {
                        rho_interp = myhalf * ( rho_cc(i,j-1,k) + rho_cc(i,j,k) );
                    } else {
                        rho_interp = rho_cc(i,j,k);
                    }
                }

                if (bdatxhi) {
                    int ii2 = std::min(std::max(i , dom_cc_lo.x), dom_cc_hi.x);
                    int jj2 = std::min(std::max(j , dom_cc_lo.y), dom_cc_hi.y);
                    arr_xhi(i,j,k) = rho_interp * bdatxhi(ii2,jj2,k,bdy_comp);
                } else {
                    Real theta_base = oma * bdatxhi_n(ii,jj,k,0) + alpha * bdatxhi_np1(ii,jj,k,0);
                    Real theta_t = theta_base;
                    if (use_theta_vertical_remap) {
                        const int ksrc_max = amrex::min(kmax_t_xhi, kmax_ph_xhi-1);
                        auto z_target_cc = [&](int ic, int jc, int kk) AMREX_GPU_DEVICE noexcept -> Real {
                            Real mu_t = oma * bdatxhi_mu_n(ic,jc,0,0) + alpha * bdatxhi_mu_np1(ic,jc,0,0) + mub_arr(ic,jc,0);
                            Real xmu_f = c1f_arr(0,0,kk) * mu_t + c2f_arr(0,0,kk);
                            Real ph_t = oma * bdatxhi_ph_n(ic,jc,kk,0) + alpha * bdatxhi_ph_np1(ic,jc,kk,0);
                            return (ph_t / xmu_f + phb_arr(ic,jc,kk)) / CONST_GRAV;
                        };

                        if ((realbdy_vertical_remap_mode == 2 ||
                             realbdy_vertical_remap_mode == 3 ||
                             realbdy_vertical_remap_mode == 4) &&
                            (ivar == ivarU || ivar == ivarV) && ksrc_max > 0) {
                            auto src_target_cc = [&](int icc, int jcc, int kk) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int i0 = amrex::min(amrex::max(icc,   dom_lo.x), dom_hi.x);
                                    int i1 = amrex::min(amrex::max(icc+1, dom_lo.x), dom_hi.x);
                                    int jc = amrex::min(amrex::max(jcc,   dom_lo.y), dom_hi.y);
                                    Real u0 = oma * bdatxhi_n(i0,jc,kk,0) + alpha * bdatxhi_np1(i0,jc,kk,0);
                                    Real u1 = oma * bdatxhi_n(i1,jc,kk,0) + alpha * bdatxhi_np1(i1,jc,kk,0);
                                    return Real(0.5) * (u0 + u1);
                                } else {
                                    int ic = amrex::min(amrex::max(icc,   dom_lo.x), dom_hi.x);
                                    int j0 = amrex::min(amrex::max(jcc,   dom_lo.y), dom_hi.y);
                                    int j1 = amrex::min(amrex::max(jcc+1, dom_lo.y), dom_hi.y);
                                    Real v0 = oma * bdatxhi_n(ic,j0,kk,0) + alpha * bdatxhi_np1(ic,j0,kk,0);
                                    Real v1 = oma * bdatxhi_n(ic,j1,kk,0) + alpha * bdatxhi_np1(ic,j1,kk,0);
                                    return Real(0.5) * (v0 + v1);
                                }
                            };
                            auto remap_target_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                const Real z_state_cc = zcc_arr(icc,jcc,k,0);
                                Real z0 = Real(0.5) * (z_target_cc(icc,jcc,0) + z_target_cc(icc,jcc,1));
                                Real f0 = src_target_cc(icc,jcc,0);
                                if (z_state_cc <= z0) { return f0; }
                                for (int kk = 0; kk < ksrc_max; ++kk) {
                                    Real zl = Real(0.5) * (z_target_cc(icc,jcc,kk) + z_target_cc(icc,jcc,kk+1));
                                    Real zh = Real(0.5) * (z_target_cc(icc,jcc,kk+1) + z_target_cc(icc,jcc,kk+2));
                                    if (z_state_cc <= zh || kk == ksrc_max-1) {
                                        Real fl = src_target_cc(icc,jcc,kk);
                                        Real fh = src_target_cc(icc,jcc,kk+1);
                                        Real dz = amrex::max(zh-zl, Real(1.e-12));
                                        Real lam = (z_state_cc - zl) / dz;
                                        lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                        return (Real(1.0)-lam)*fl + lam*fh;
                                    }
                                }
                                return src_target_cc(icc,jcc,ksrc_max);
                            };
                            auto state_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int i0 = amrex::min(amrex::max(icc,   dom_cc_lo.x), dom_cc_hi.x);
                                    int i1 = amrex::min(amrex::max(icc+1, dom_cc_lo.x), dom_cc_hi.x+1);
                                    int jc = amrex::min(amrex::max(jcc,   dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (state_arr(i0,jc,k,0) + state_arr(i1,jc,k,0));
                                } else {
                                    int ic = amrex::min(amrex::max(icc,   dom_cc_lo.x), dom_cc_hi.x);
                                    int j0 = amrex::min(amrex::max(jcc,   dom_cc_lo.y), dom_cc_hi.y);
                                    int j1 = amrex::min(amrex::max(jcc+1, dom_cc_lo.y), dom_cc_hi.y+1);
                                    return Real(0.5) * (state_arr(ic,j0,k,0) + state_arr(ic,j1,k,0));
                                }
                            };
                            auto rho_state_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                int kc = amrex::max(0, amrex::min(k, kmax_ph_xhi-1));
                                Real mu_t = oma * bdatxhi_mu_n(icc,jcc,0,0) + alpha * bdatxhi_mu_np1(icc,jcc,0,0) + mub_arr(icc,jcc,0);
                                Real xmu_mult_h = c1h_arr(0,0,kc) * mu_t + c2h_arr(0,0,kc);
                                Real dpd = xmu_mult_h * amrex::Math::abs(dnw_arr(0,0,kc));
                                Real phi_k   = oma * bdatxhi_ph_n(icc,jcc,kc,0)   + alpha * bdatxhi_ph_np1(icc,jcc,kc,0)   + phb_arr(icc,jcc,kc);
                                Real phi_kp1 = oma * bdatxhi_ph_n(icc,jcc,kc+1,0) + alpha * bdatxhi_ph_np1(icc,jcc,kc+1,0) + phb_arr(icc,jcc,kc+1);
                                Real dphi = amrex::max(phi_kp1 - phi_k, Real(1.0e-12));
                                return dpd / dphi;
                            };

                            if (ivar == ivarU) {
                                int iL = amrex::min(amrex::max(i-1, dom_cc_lo.x), dom_cc_hi.x);
                                int iR = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                if (realbdy_vertical_remap_mode == 2) {
                                    theta_t = Real(0.5) * (remap_target_cc(iL,jC) + remap_target_cc(iR,jC));
                                } else if (realbdy_vertical_remap_mode == 3) {
                                    Real deltaL = remap_target_cc(iL,jC) - state_cc(iL,jC);
                                    Real deltaR = remap_target_cc(iR,jC) - state_cc(iR,jC);
                                    Real delta_face = Real(0.5) * (deltaL + deltaR);
                                    theta_t = state_arr(i,j,k,0) + delta_face;
                                } else {
                                    Real deltaL = remap_target_cc(iL,jC) - state_cc(iL,jC);
                                    Real deltaR = remap_target_cc(iR,jC) - state_cc(iR,jC);
                                    Real dconsL = rho_state_cc(iL,jC) * deltaL;
                                    Real dconsR = rho_state_cc(iR,jC) * deltaR;
                                    Real dcons_face = Real(0.5) * (dconsL + dconsR);
                                    Real cons_target = state_cons_arr(i,j,k,0) + dcons_face;
                                    Real rho_safe = (amrex::Math::abs(rho_interp) > Real(1.0e-12)) ? rho_interp : Real(1.0e-12);
                                    theta_t = cons_target / rho_safe;
                                }
                            } else {
                                int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                int jL = amrex::min(amrex::max(j-1, dom_cc_lo.y), dom_cc_hi.y);
                                int jR = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                if (realbdy_vertical_remap_mode == 2) {
                                    theta_t = Real(0.5) * (remap_target_cc(iC,jL) + remap_target_cc(iC,jR));
                                } else if (realbdy_vertical_remap_mode == 3) {
                                    Real deltaL = remap_target_cc(iC,jL) - state_cc(iC,jL);
                                    Real deltaR = remap_target_cc(iC,jR) - state_cc(iC,jR);
                                    Real delta_face = Real(0.5) * (deltaL + deltaR);
                                    theta_t = state_arr(i,j,k,0) + delta_face;
                                } else {
                                    Real deltaL = remap_target_cc(iC,jL) - state_cc(iC,jL);
                                    Real deltaR = remap_target_cc(iC,jR) - state_cc(iC,jR);
                                    Real dconsL = rho_state_cc(iC,jL) * deltaL;
                                    Real dconsR = rho_state_cc(iC,jR) * deltaR;
                                    Real dcons_face = Real(0.5) * (dconsL + dconsR);
                                    Real cons_target = state_cons_arr(i,j,k,0) + dcons_face;
                                    Real rho_safe = (amrex::Math::abs(rho_interp) > Real(1.0e-12)) ? rho_interp : Real(1.0e-12);
                                    theta_t = cons_target / rho_safe;
                                }
                            }
                        } else {
                            // Path A-2: remap target profile directly at the state variable location.
                            const Real z_state = z_tgt_loc(ivar, i, j, k);
                            auto z_target = [&](int kk) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int iL = amrex::min(amrex::max(i-1, dom_cc_lo.x), dom_cc_hi.x);
                                    int iR = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (z_target_cc(iL,jC,kk) + z_target_cc(iR,jC,kk));
                                } else if (ivar == ivarV) {
                                    int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jL = amrex::min(amrex::max(j-1, dom_cc_lo.y), dom_cc_hi.y);
                                    int jR = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (z_target_cc(iC,jL,kk) + z_target_cc(iC,jR,kk));
                                } else {
                                    int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return z_target_cc(iC,jC,kk);
                                }
                            };
                            if (ksrc_max > 0) {
                                Real z0 = Real(0.5) * (z_target(0) + z_target(1));
                                Real t0 = oma * bdatxhi_n(ii,jj,0,0) + alpha * bdatxhi_np1(ii,jj,0,0);
                                if (z_state <= z0) {
                                    theta_t = t0;
                                } else {
                                    bool found = false;
                                    for (int kk = 0; kk < ksrc_max; ++kk) {
                                        Real zl = Real(0.5) * (z_target(kk) + z_target(kk+1));
                                        Real zh = Real(0.5) * (z_target(kk+1) + z_target(kk+2));
                                        if (z_state <= zh || kk == ksrc_max-1) {
                                            Real tl = oma * bdatxhi_n(ii,jj,kk,0) + alpha * bdatxhi_np1(ii,jj,kk,0);
                                            Real th = oma * bdatxhi_n(ii,jj,kk+1,0) + alpha * bdatxhi_np1(ii,jj,kk+1,0);
                                            Real dz = amrex::max(zh-zl, Real(1.e-12));
                                            Real lam = (z_state - zl) / dz;
                                            lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                            theta_t = (Real(1.0)-lam)*tl + lam*th;
                                            found = true;
                                            break;
                                        }
                                    }
                                    if (!found) { theta_t = oma * bdatxhi_n(ii,jj,ksrc_max,0) + alpha * bdatxhi_np1(ii,jj,ksrc_max,0); }
                                }
                            }
                        }
                    }
                    arr_xhi(i,j,k) = rho_interp * theta_t;
                    if (do_dump_theta_for_var) {
                        theta_dbg_arr(i,j,k,0) = theta_base;
                        theta_dbg_arr(i,j,k,1) = theta_t;
                        theta_dbg_arr(i,j,k,2) = theta_t - theta_base;
                    }
                }
            });

            ParallelFor(tbx_ylo, tbx_yhi,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int ii = std::max(i , dom_lo.x);
                    ii = std::min(ii, dom_hi.x + ((ivar==ivarU) ? 1 : 0));
                int jj = std::max(j , dom_lo.y);
                    jj = std::min(jj, dom_lo.y + offset + ((ivar==ivarV) ? 1 : 0));

                Real rho_interp;
                if (!use_wrf_rho_interp) {
                    if (ivar==ivarU) {
                        rho_interp = myhalf * ( r_arr(i-1,j  ,k) + r_arr(i,j,k) );
                    } else if (ivar==ivarV) {
                        rho_interp = myhalf * ( r_arr(i  ,j-1,k) + r_arr(i,j,k) );
                    } else {
                        rho_interp = r_arr(i,j,k);
                    }
                } else {
                    auto rho_cc = [&] AMREX_GPU_DEVICE (int ic, int jc, int kc) noexcept -> Real {
                        kc = amrex::max(0, amrex::min(kc, kmax_ph_ylo-1));
                        Real mu_t = oma * bdatylo_mu_n(ic,jc,0,0) + alpha * bdatylo_mu_np1(ic,jc,0,0) + mub_arr(ic,jc,0);
                        Real xmu_mult_h = c1h_arr(0,0,kc) * mu_t + c2h_arr(0,0,kc);
                        Real dpd = xmu_mult_h * amrex::Math::abs(dnw_arr(0,0,kc));
                        Real phi_k   = oma * bdatylo_ph_n(ic,jc,kc,0)   + alpha * bdatylo_ph_np1(ic,jc,kc,0)   + phb_arr(ic,jc,kc);
                        Real phi_kp1 = oma * bdatylo_ph_n(ic,jc,kc+1,0) + alpha * bdatylo_ph_np1(ic,jc,kc+1,0) + phb_arr(ic,jc,kc+1);
                        Real dphi = amrex::max(phi_kp1 - phi_k, Real(1.0e-12));
                        return dpd / dphi;
                    };
                    if (ivar==ivarU) {
                        rho_interp = myhalf * ( rho_cc(i-1,j,k) + rho_cc(i,j,k) );
                    } else if (ivar==ivarV) {
                        rho_interp = myhalf * ( rho_cc(i,j-1,k) + rho_cc(i,j,k) );
                    } else {
                        rho_interp = rho_cc(i,j,k);
                    }
                }

                if (bdatylo) {
                    int ii2 = std::min(std::max(i , dom_cc_lo.x), dom_cc_hi.x);
                    int jj2 = std::min(std::max(j , dom_cc_lo.y), dom_cc_hi.y);
                    arr_ylo(i,j,k) = rho_interp * bdatylo(ii2,jj2,k,bdy_comp);
                } else {
                    Real theta_base = oma * bdatylo_n(ii,jj,k,0) + alpha * bdatylo_np1(ii,jj,k,0);
                    Real theta_t = theta_base;
                    if (use_theta_vertical_remap) {
                        const int ksrc_max = amrex::min(kmax_t_ylo, kmax_ph_ylo-1);
                        auto z_target_cc = [&](int ic, int jc, int kk) AMREX_GPU_DEVICE noexcept -> Real {
                            Real mu_t = oma * bdatylo_mu_n(ic,jc,0,0) + alpha * bdatylo_mu_np1(ic,jc,0,0) + mub_arr(ic,jc,0);
                            Real xmu_f = c1f_arr(0,0,kk) * mu_t + c2f_arr(0,0,kk);
                            Real ph_t = oma * bdatylo_ph_n(ic,jc,kk,0) + alpha * bdatylo_ph_np1(ic,jc,kk,0);
                            return (ph_t / xmu_f + phb_arr(ic,jc,kk)) / CONST_GRAV;
                        };

                        if ((realbdy_vertical_remap_mode == 2 ||
                             realbdy_vertical_remap_mode == 3 ||
                             realbdy_vertical_remap_mode == 4) &&
                            (ivar == ivarU || ivar == ivarV) && ksrc_max > 0) {
                            auto src_target_cc = [&](int icc, int jcc, int kk) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int i0 = amrex::min(amrex::max(icc,   dom_lo.x), dom_hi.x);
                                    int i1 = amrex::min(amrex::max(icc+1, dom_lo.x), dom_hi.x);
                                    int jc = amrex::min(amrex::max(jcc,   dom_lo.y), dom_hi.y);
                                    Real u0 = oma * bdatylo_n(i0,jc,kk,0) + alpha * bdatylo_np1(i0,jc,kk,0);
                                    Real u1 = oma * bdatylo_n(i1,jc,kk,0) + alpha * bdatylo_np1(i1,jc,kk,0);
                                    return Real(0.5) * (u0 + u1);
                                } else {
                                    int ic = amrex::min(amrex::max(icc,   dom_lo.x), dom_hi.x);
                                    int j0 = amrex::min(amrex::max(jcc,   dom_lo.y), dom_hi.y);
                                    int j1 = amrex::min(amrex::max(jcc+1, dom_lo.y), dom_hi.y);
                                    Real v0 = oma * bdatylo_n(ic,j0,kk,0) + alpha * bdatylo_np1(ic,j0,kk,0);
                                    Real v1 = oma * bdatylo_n(ic,j1,kk,0) + alpha * bdatylo_np1(ic,j1,kk,0);
                                    return Real(0.5) * (v0 + v1);
                                }
                            };
                            auto remap_target_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                const Real z_state_cc = zcc_arr(icc,jcc,k,0);
                                Real z0 = Real(0.5) * (z_target_cc(icc,jcc,0) + z_target_cc(icc,jcc,1));
                                Real f0 = src_target_cc(icc,jcc,0);
                                if (z_state_cc <= z0) { return f0; }
                                for (int kk = 0; kk < ksrc_max; ++kk) {
                                    Real zl = Real(0.5) * (z_target_cc(icc,jcc,kk) + z_target_cc(icc,jcc,kk+1));
                                    Real zh = Real(0.5) * (z_target_cc(icc,jcc,kk+1) + z_target_cc(icc,jcc,kk+2));
                                    if (z_state_cc <= zh || kk == ksrc_max-1) {
                                        Real fl = src_target_cc(icc,jcc,kk);
                                        Real fh = src_target_cc(icc,jcc,kk+1);
                                        Real dz = amrex::max(zh-zl, Real(1.e-12));
                                        Real lam = (z_state_cc - zl) / dz;
                                        lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                        return (Real(1.0)-lam)*fl + lam*fh;
                                    }
                                }
                                return src_target_cc(icc,jcc,ksrc_max);
                            };
                            auto state_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int i0 = amrex::min(amrex::max(icc,   dom_cc_lo.x), dom_cc_hi.x);
                                    int i1 = amrex::min(amrex::max(icc+1, dom_cc_lo.x), dom_cc_hi.x+1);
                                    int jc = amrex::min(amrex::max(jcc,   dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (state_arr(i0,jc,k,0) + state_arr(i1,jc,k,0));
                                } else {
                                    int ic = amrex::min(amrex::max(icc,   dom_cc_lo.x), dom_cc_hi.x);
                                    int j0 = amrex::min(amrex::max(jcc,   dom_cc_lo.y), dom_cc_hi.y);
                                    int j1 = amrex::min(amrex::max(jcc+1, dom_cc_lo.y), dom_cc_hi.y+1);
                                    return Real(0.5) * (state_arr(ic,j0,k,0) + state_arr(ic,j1,k,0));
                                }
                            };
                            auto rho_state_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                int kc = amrex::max(0, amrex::min(k, kmax_ph_ylo-1));
                                Real mu_t = oma * bdatylo_mu_n(icc,jcc,0,0) + alpha * bdatylo_mu_np1(icc,jcc,0,0) + mub_arr(icc,jcc,0);
                                Real xmu_mult_h = c1h_arr(0,0,kc) * mu_t + c2h_arr(0,0,kc);
                                Real dpd = xmu_mult_h * amrex::Math::abs(dnw_arr(0,0,kc));
                                Real phi_k   = oma * bdatylo_ph_n(icc,jcc,kc,0)   + alpha * bdatylo_ph_np1(icc,jcc,kc,0)   + phb_arr(icc,jcc,kc);
                                Real phi_kp1 = oma * bdatylo_ph_n(icc,jcc,kc+1,0) + alpha * bdatylo_ph_np1(icc,jcc,kc+1,0) + phb_arr(icc,jcc,kc+1);
                                Real dphi = amrex::max(phi_kp1 - phi_k, Real(1.0e-12));
                                return dpd / dphi;
                            };

                            if (ivar == ivarU) {
                                int iL = amrex::min(amrex::max(i-1, dom_cc_lo.x), dom_cc_hi.x);
                                int iR = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                if (realbdy_vertical_remap_mode == 2) {
                                    theta_t = Real(0.5) * (remap_target_cc(iL,jC) + remap_target_cc(iR,jC));
                                } else if (realbdy_vertical_remap_mode == 3) {
                                    Real deltaL = remap_target_cc(iL,jC) - state_cc(iL,jC);
                                    Real deltaR = remap_target_cc(iR,jC) - state_cc(iR,jC);
                                    Real delta_face = Real(0.5) * (deltaL + deltaR);
                                    theta_t = state_arr(i,j,k,0) + delta_face;
                                } else {
                                    Real deltaL = remap_target_cc(iL,jC) - state_cc(iL,jC);
                                    Real deltaR = remap_target_cc(iR,jC) - state_cc(iR,jC);
                                    Real dconsL = rho_state_cc(iL,jC) * deltaL;
                                    Real dconsR = rho_state_cc(iR,jC) * deltaR;
                                    Real dcons_face = Real(0.5) * (dconsL + dconsR);
                                    Real cons_target = state_cons_arr(i,j,k,0) + dcons_face;
                                    Real rho_safe = (amrex::Math::abs(rho_interp) > Real(1.0e-12)) ? rho_interp : Real(1.0e-12);
                                    theta_t = cons_target / rho_safe;
                                }
                            } else {
                                int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                int jL = amrex::min(amrex::max(j-1, dom_cc_lo.y), dom_cc_hi.y);
                                int jR = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                if (realbdy_vertical_remap_mode == 2) {
                                    theta_t = Real(0.5) * (remap_target_cc(iC,jL) + remap_target_cc(iC,jR));
                                } else if (realbdy_vertical_remap_mode == 3) {
                                    Real deltaL = remap_target_cc(iC,jL) - state_cc(iC,jL);
                                    Real deltaR = remap_target_cc(iC,jR) - state_cc(iC,jR);
                                    Real delta_face = Real(0.5) * (deltaL + deltaR);
                                    theta_t = state_arr(i,j,k,0) + delta_face;
                                } else {
                                    Real deltaL = remap_target_cc(iC,jL) - state_cc(iC,jL);
                                    Real deltaR = remap_target_cc(iC,jR) - state_cc(iC,jR);
                                    Real dconsL = rho_state_cc(iC,jL) * deltaL;
                                    Real dconsR = rho_state_cc(iC,jR) * deltaR;
                                    Real dcons_face = Real(0.5) * (dconsL + dconsR);
                                    Real cons_target = state_cons_arr(i,j,k,0) + dcons_face;
                                    Real rho_safe = (amrex::Math::abs(rho_interp) > Real(1.0e-12)) ? rho_interp : Real(1.0e-12);
                                    theta_t = cons_target / rho_safe;
                                }
                            }
                        } else {
                            // Path A-2: remap target profile directly at the state variable location.
                            const Real z_state = z_tgt_loc(ivar, i, j, k);
                            auto z_target = [&](int kk) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int iL = amrex::min(amrex::max(i-1, dom_cc_lo.x), dom_cc_hi.x);
                                    int iR = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (z_target_cc(iL,jC,kk) + z_target_cc(iR,jC,kk));
                                } else if (ivar == ivarV) {
                                    int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jL = amrex::min(amrex::max(j-1, dom_cc_lo.y), dom_cc_hi.y);
                                    int jR = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (z_target_cc(iC,jL,kk) + z_target_cc(iC,jR,kk));
                                } else {
                                    int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return z_target_cc(iC,jC,kk);
                                }
                            };
                            if (ksrc_max > 0) {
                                Real z0 = Real(0.5) * (z_target(0) + z_target(1));
                                Real t0 = oma * bdatylo_n(ii,jj,0,0) + alpha * bdatylo_np1(ii,jj,0,0);
                                if (z_state <= z0) {
                                    theta_t = t0;
                                } else {
                                    bool found = false;
                                    for (int kk = 0; kk < ksrc_max; ++kk) {
                                        Real zl = Real(0.5) * (z_target(kk) + z_target(kk+1));
                                        Real zh = Real(0.5) * (z_target(kk+1) + z_target(kk+2));
                                        if (z_state <= zh || kk == ksrc_max-1) {
                                            Real tl = oma * bdatylo_n(ii,jj,kk,0) + alpha * bdatylo_np1(ii,jj,kk,0);
                                            Real th = oma * bdatylo_n(ii,jj,kk+1,0) + alpha * bdatylo_np1(ii,jj,kk+1,0);
                                            Real dz = amrex::max(zh-zl, Real(1.e-12));
                                            Real lam = (z_state - zl) / dz;
                                            lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                            theta_t = (Real(1.0)-lam)*tl + lam*th;
                                            found = true;
                                            break;
                                        }
                                    }
                                    if (!found) { theta_t = oma * bdatylo_n(ii,jj,ksrc_max,0) + alpha * bdatylo_np1(ii,jj,ksrc_max,0); }
                                }
                            }
                        }
                    }
                    arr_ylo(i,j,k) = rho_interp * theta_t;
                    if (do_dump_theta_for_var) {
                        theta_dbg_arr(i,j,k,0) = theta_base;
                        theta_dbg_arr(i,j,k,1) = theta_t;
                        theta_dbg_arr(i,j,k,2) = theta_t - theta_base;
                    }
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int ii = std::max(i , dom_lo.x);
                    ii = std::min(ii, dom_hi.x + ((ivar==ivarU) ? 1 : 0));
                int jj = std::max(j , dom_hi.y - offset + ((ivar==ivarV) ? 1 : 0));
                    jj = std::min(jj, dom_hi.y + ((ivar==ivarV) ? 1 : 0));

                Real rho_interp;
                if (!use_wrf_rho_interp) {
                    if (ivar==ivarU) {
                        rho_interp = myhalf * ( r_arr(i-1,j  ,k) + r_arr(i,j,k) );
                    } else if (ivar==ivarV) {
                        rho_interp = myhalf * ( r_arr(i  ,j-1,k) + r_arr(i,j,k) );
                    } else {
                        rho_interp = r_arr(i,j,k);
                    }
                } else {
                    auto rho_cc = [&] AMREX_GPU_DEVICE (int ic, int jc, int kc) noexcept -> Real {
                        kc = amrex::max(0, amrex::min(kc, kmax_ph_yhi-1));
                        Real mu_t = oma * bdatyhi_mu_n(ic,jc,0,0) + alpha * bdatyhi_mu_np1(ic,jc,0,0) + mub_arr(ic,jc,0);
                        Real xmu_mult_h = c1h_arr(0,0,kc) * mu_t + c2h_arr(0,0,kc);
                        Real dpd = xmu_mult_h * amrex::Math::abs(dnw_arr(0,0,kc));
                        Real phi_k   = oma * bdatyhi_ph_n(ic,jc,kc,0)   + alpha * bdatyhi_ph_np1(ic,jc,kc,0)   + phb_arr(ic,jc,kc);
                        Real phi_kp1 = oma * bdatyhi_ph_n(ic,jc,kc+1,0) + alpha * bdatyhi_ph_np1(ic,jc,kc+1,0) + phb_arr(ic,jc,kc+1);
                        Real dphi = amrex::max(phi_kp1 - phi_k, Real(1.0e-12));
                        return dpd / dphi;
                    };
                    if (ivar==ivarU) {
                        rho_interp = myhalf * ( rho_cc(i-1,j,k) + rho_cc(i,j,k) );
                    } else if (ivar==ivarV) {
                        rho_interp = myhalf * ( rho_cc(i,j-1,k) + rho_cc(i,j,k) );
                    } else {
                        rho_interp = rho_cc(i,j,k);
                    }
                }

                if (bdatyhi) {
                    int ii2 = std::min(std::max(i , dom_cc_lo.x), dom_cc_hi.x);
                    int jj2 = std::min(std::max(j , dom_cc_lo.y), dom_cc_hi.y);
                    arr_yhi(i,j,k) = rho_interp * bdatyhi(ii2,jj2,k,bdy_comp);
                } else {
                    Real theta_base = oma * bdatyhi_n(ii,jj,k,0) + alpha * bdatyhi_np1(ii,jj,k,0);
                    Real theta_t = theta_base;
                    if (use_theta_vertical_remap) {
                        const int ksrc_max = amrex::min(kmax_t_yhi, kmax_ph_yhi-1);
                        auto z_target_cc = [&](int ic, int jc, int kk) AMREX_GPU_DEVICE noexcept -> Real {
                            Real mu_t = oma * bdatyhi_mu_n(ic,jc,0,0) + alpha * bdatyhi_mu_np1(ic,jc,0,0) + mub_arr(ic,jc,0);
                            Real xmu_f = c1f_arr(0,0,kk) * mu_t + c2f_arr(0,0,kk);
                            Real ph_t = oma * bdatyhi_ph_n(ic,jc,kk,0) + alpha * bdatyhi_ph_np1(ic,jc,kk,0);
                            return (ph_t / xmu_f + phb_arr(ic,jc,kk)) / CONST_GRAV;
                        };

                        if ((realbdy_vertical_remap_mode == 2 ||
                             realbdy_vertical_remap_mode == 3 ||
                             realbdy_vertical_remap_mode == 4) &&
                            (ivar == ivarU || ivar == ivarV) && ksrc_max > 0) {
                            auto src_target_cc = [&](int icc, int jcc, int kk) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int i0 = amrex::min(amrex::max(icc,   dom_lo.x), dom_hi.x);
                                    int i1 = amrex::min(amrex::max(icc+1, dom_lo.x), dom_hi.x);
                                    int jc = amrex::min(amrex::max(jcc,   dom_lo.y), dom_hi.y);
                                    Real u0 = oma * bdatyhi_n(i0,jc,kk,0) + alpha * bdatyhi_np1(i0,jc,kk,0);
                                    Real u1 = oma * bdatyhi_n(i1,jc,kk,0) + alpha * bdatyhi_np1(i1,jc,kk,0);
                                    return Real(0.5) * (u0 + u1);
                                } else {
                                    int ic = amrex::min(amrex::max(icc,   dom_lo.x), dom_hi.x);
                                    int j0 = amrex::min(amrex::max(jcc,   dom_lo.y), dom_hi.y);
                                    int j1 = amrex::min(amrex::max(jcc+1, dom_lo.y), dom_hi.y);
                                    Real v0 = oma * bdatyhi_n(ic,j0,kk,0) + alpha * bdatyhi_np1(ic,j0,kk,0);
                                    Real v1 = oma * bdatyhi_n(ic,j1,kk,0) + alpha * bdatyhi_np1(ic,j1,kk,0);
                                    return Real(0.5) * (v0 + v1);
                                }
                            };
                            auto remap_target_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                const Real z_state_cc = zcc_arr(icc,jcc,k,0);
                                Real z0 = Real(0.5) * (z_target_cc(icc,jcc,0) + z_target_cc(icc,jcc,1));
                                Real f0 = src_target_cc(icc,jcc,0);
                                if (z_state_cc <= z0) { return f0; }
                                for (int kk = 0; kk < ksrc_max; ++kk) {
                                    Real zl = Real(0.5) * (z_target_cc(icc,jcc,kk) + z_target_cc(icc,jcc,kk+1));
                                    Real zh = Real(0.5) * (z_target_cc(icc,jcc,kk+1) + z_target_cc(icc,jcc,kk+2));
                                    if (z_state_cc <= zh || kk == ksrc_max-1) {
                                        Real fl = src_target_cc(icc,jcc,kk);
                                        Real fh = src_target_cc(icc,jcc,kk+1);
                                        Real dz = amrex::max(zh-zl, Real(1.e-12));
                                        Real lam = (z_state_cc - zl) / dz;
                                        lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                        return (Real(1.0)-lam)*fl + lam*fh;
                                    }
                                }
                                return src_target_cc(icc,jcc,ksrc_max);
                            };
                            auto state_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int i0 = amrex::min(amrex::max(icc,   dom_cc_lo.x), dom_cc_hi.x);
                                    int i1 = amrex::min(amrex::max(icc+1, dom_cc_lo.x), dom_cc_hi.x+1);
                                    int jc = amrex::min(amrex::max(jcc,   dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (state_arr(i0,jc,k,0) + state_arr(i1,jc,k,0));
                                } else {
                                    int ic = amrex::min(amrex::max(icc,   dom_cc_lo.x), dom_cc_hi.x);
                                    int j0 = amrex::min(amrex::max(jcc,   dom_cc_lo.y), dom_cc_hi.y);
                                    int j1 = amrex::min(amrex::max(jcc+1, dom_cc_lo.y), dom_cc_hi.y+1);
                                    return Real(0.5) * (state_arr(ic,j0,k,0) + state_arr(ic,j1,k,0));
                                }
                            };
                            auto rho_state_cc = [&](int icc, int jcc) AMREX_GPU_DEVICE noexcept -> Real {
                                int kc = amrex::max(0, amrex::min(k, kmax_ph_yhi-1));
                                Real mu_t = oma * bdatyhi_mu_n(icc,jcc,0,0) + alpha * bdatyhi_mu_np1(icc,jcc,0,0) + mub_arr(icc,jcc,0);
                                Real xmu_mult_h = c1h_arr(0,0,kc) * mu_t + c2h_arr(0,0,kc);
                                Real dpd = xmu_mult_h * amrex::Math::abs(dnw_arr(0,0,kc));
                                Real phi_k   = oma * bdatyhi_ph_n(icc,jcc,kc,0)   + alpha * bdatyhi_ph_np1(icc,jcc,kc,0)   + phb_arr(icc,jcc,kc);
                                Real phi_kp1 = oma * bdatyhi_ph_n(icc,jcc,kc+1,0) + alpha * bdatyhi_ph_np1(icc,jcc,kc+1,0) + phb_arr(icc,jcc,kc+1);
                                Real dphi = amrex::max(phi_kp1 - phi_k, Real(1.0e-12));
                                return dpd / dphi;
                            };

                            if (ivar == ivarU) {
                                int iL = amrex::min(amrex::max(i-1, dom_cc_lo.x), dom_cc_hi.x);
                                int iR = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                if (realbdy_vertical_remap_mode == 2) {
                                    theta_t = Real(0.5) * (remap_target_cc(iL,jC) + remap_target_cc(iR,jC));
                                } else if (realbdy_vertical_remap_mode == 3) {
                                    Real deltaL = remap_target_cc(iL,jC) - state_cc(iL,jC);
                                    Real deltaR = remap_target_cc(iR,jC) - state_cc(iR,jC);
                                    Real delta_face = Real(0.5) * (deltaL + deltaR);
                                    theta_t = state_arr(i,j,k,0) + delta_face;
                                } else {
                                    Real deltaL = remap_target_cc(iL,jC) - state_cc(iL,jC);
                                    Real deltaR = remap_target_cc(iR,jC) - state_cc(iR,jC);
                                    Real dconsL = rho_state_cc(iL,jC) * deltaL;
                                    Real dconsR = rho_state_cc(iR,jC) * deltaR;
                                    Real dcons_face = Real(0.5) * (dconsL + dconsR);
                                    Real cons_target = state_cons_arr(i,j,k,0) + dcons_face;
                                    Real rho_safe = (amrex::Math::abs(rho_interp) > Real(1.0e-12)) ? rho_interp : Real(1.0e-12);
                                    theta_t = cons_target / rho_safe;
                                }
                            } else {
                                int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                int jL = amrex::min(amrex::max(j-1, dom_cc_lo.y), dom_cc_hi.y);
                                int jR = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                if (realbdy_vertical_remap_mode == 2) {
                                    theta_t = Real(0.5) * (remap_target_cc(iC,jL) + remap_target_cc(iC,jR));
                                } else if (realbdy_vertical_remap_mode == 3) {
                                    Real deltaL = remap_target_cc(iC,jL) - state_cc(iC,jL);
                                    Real deltaR = remap_target_cc(iC,jR) - state_cc(iC,jR);
                                    Real delta_face = Real(0.5) * (deltaL + deltaR);
                                    theta_t = state_arr(i,j,k,0) + delta_face;
                                } else {
                                    Real deltaL = remap_target_cc(iC,jL) - state_cc(iC,jL);
                                    Real deltaR = remap_target_cc(iC,jR) - state_cc(iC,jR);
                                    Real dconsL = rho_state_cc(iC,jL) * deltaL;
                                    Real dconsR = rho_state_cc(iC,jR) * deltaR;
                                    Real dcons_face = Real(0.5) * (dconsL + dconsR);
                                    Real cons_target = state_cons_arr(i,j,k,0) + dcons_face;
                                    Real rho_safe = (amrex::Math::abs(rho_interp) > Real(1.0e-12)) ? rho_interp : Real(1.0e-12);
                                    theta_t = cons_target / rho_safe;
                                }
                            }
                        } else {
                            // Path A-2: remap target profile directly at the state variable location.
                            const Real z_state = z_tgt_loc(ivar, i, j, k);
                            auto z_target = [&](int kk) AMREX_GPU_DEVICE noexcept -> Real {
                                if (ivar == ivarU) {
                                    int iL = amrex::min(amrex::max(i-1, dom_cc_lo.x), dom_cc_hi.x);
                                    int iR = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (z_target_cc(iL,jC,kk) + z_target_cc(iR,jC,kk));
                                } else if (ivar == ivarV) {
                                    int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jL = amrex::min(amrex::max(j-1, dom_cc_lo.y), dom_cc_hi.y);
                                    int jR = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return Real(0.5) * (z_target_cc(iC,jL,kk) + z_target_cc(iC,jR,kk));
                                } else {
                                    int iC = amrex::min(amrex::max(i  , dom_cc_lo.x), dom_cc_hi.x);
                                    int jC = amrex::min(amrex::max(j  , dom_cc_lo.y), dom_cc_hi.y);
                                    return z_target_cc(iC,jC,kk);
                                }
                            };
                            if (ksrc_max > 0) {
                                Real z0 = Real(0.5) * (z_target(0) + z_target(1));
                                Real t0 = oma * bdatyhi_n(ii,jj,0,0) + alpha * bdatyhi_np1(ii,jj,0,0);
                                if (z_state <= z0) {
                                    theta_t = t0;
                                } else {
                                    bool found = false;
                                    for (int kk = 0; kk < ksrc_max; ++kk) {
                                        Real zl = Real(0.5) * (z_target(kk) + z_target(kk+1));
                                        Real zh = Real(0.5) * (z_target(kk+1) + z_target(kk+2));
                                        if (z_state <= zh || kk == ksrc_max-1) {
                                            Real tl = oma * bdatyhi_n(ii,jj,kk,0) + alpha * bdatyhi_np1(ii,jj,kk,0);
                                            Real th = oma * bdatyhi_n(ii,jj,kk+1,0) + alpha * bdatyhi_np1(ii,jj,kk+1,0);
                                            Real dz = amrex::max(zh-zl, Real(1.e-12));
                                            Real lam = (z_state - zl) / dz;
                                            lam = amrex::max(Real(0.0), amrex::min(Real(1.0), lam));
                                            theta_t = (Real(1.0)-lam)*tl + lam*th;
                                            found = true;
                                            break;
                                        }
                                    }
                                    if (!found) { theta_t = oma * bdatyhi_n(ii,jj,ksrc_max,0) + alpha * bdatyhi_np1(ii,jj,ksrc_max,0); }
                                }
                            }
                        }
                    }
                    arr_yhi(i,j,k) = rho_interp * theta_t;
                    if (do_dump_theta_for_var) {
                        theta_dbg_arr(i,j,k,0) = theta_base;
                        theta_dbg_arr(i,j,k,1) = theta_t;
                        theta_dbg_arr(i,j,k,2) = theta_t - theta_base;
                    }
                }
            });

#ifndef AMREX_USE_GPU
            if (dbg_realbdy_rho_pathB_diag && ivar == ivarT && have_wrfbdy_ph && ParallelDescriptor::IOProcessor()) {
                int ic = std::min(std::max(domain.smallEnd(0), domain.smallEnd(0)+width/2), domain.bigEnd(0));
                int jc = (domain.smallEnd(1) + domain.bigEnd(1)) / 2;
                int k0 = 0;
                int km = domain.bigEnd(2) / 2;
                int kt = domain.bigEnd(2);
                auto rho_wrf_cc = [&](int i, int j, int k) {
                    int kph = std::max(0, std::min(k, bdy_data_xlo[n_time][WRFBdyVars::PH].box().bigEnd(2)-1));
                    Real mu_t = oma * bdatxlo_mu_n(i,j,0,0) + alpha * bdatxlo_mu_np1(i,j,0,0) + mub_arr(i,j,0);
                    Real xmu_mult_h = c1h_arr(0,0,kph) * mu_t + c2h_arr(0,0,kph);
                    Real dpd = xmu_mult_h * std::abs(dnw_arr(0,0,kph));
                    Real phi_k   = oma * bdatxlo_ph_n(i,j,kph,0)   + alpha * bdatxlo_ph_np1(i,j,kph,0)   + phb_arr(i,j,kph);
                    Real phi_kp1 = oma * bdatxlo_ph_n(i,j,kph+1,0) + alpha * bdatxlo_ph_np1(i,j,kph+1,0) + phb_arr(i,j,kph+1);
                    Real dphi = std::max(phi_kp1 - phi_k, Real(1.0e-12));
                    return dpd / dphi;
                };
                auto dpd_dphi = [&](int i, int j, int k) {
                    int kph = std::max(0, std::min(k, bdy_data_xlo[n_time][WRFBdyVars::PH].box().bigEnd(2)-1));
                    Real mu_t = oma * bdatxlo_mu_n(i,j,0,0) + alpha * bdatxlo_mu_np1(i,j,0,0) + mub_arr(i,j,0);
                    Real xmu_mult_h = c1h_arr(0,0,kph) * mu_t + c2h_arr(0,0,kph);
                    Real dpd = xmu_mult_h * std::abs(dnw_arr(0,0,kph));
                    Real phi_k   = oma * bdatxlo_ph_n(i,j,kph,0)   + alpha * bdatxlo_ph_np1(i,j,kph,0)   + phb_arr(i,j,kph);
                    Real phi_kp1 = oma * bdatxlo_ph_n(i,j,kph+1,0) + alpha * bdatxlo_ph_np1(i,j,kph+1,0) + phb_arr(i,j,kph+1);
                    Real dphi = phi_kp1 - phi_k;
                    return std::array<Real,3>{dpd,dphi,xmu_mult_h};
                };
                auto a0 = dpd_dphi(ic,jc,k0);
                auto am = dpd_dphi(ic,jc,km);
                auto at = dpd_dphi(ic,jc,kt);
                Print() << "[DBG_RHO_PATHB sample xlo]"
                        << " i=" << ic << " j=" << jc
                        << " k0 wrf=" << rho_wrf_cc(ic,jc,k0) << " erf=" << r_arr(ic,jc,k0)
                        << " dpd=" << a0[0] << " dphi=" << a0[1] << " xmuH=" << a0[2]
                        << " km wrf=" << rho_wrf_cc(ic,jc,km) << " erf=" << r_arr(ic,jc,km)
                        << " dpd=" << am[0] << " dphi=" << am[1] << " xmuH=" << am[2]
                        << " kt wrf=" << rho_wrf_cc(ic,jc,kt) << " erf=" << r_arr(ic,jc,kt)
                        << " dpd=" << at[0] << " dphi=" << at[1] << " xmuH=" << at[2]
                        << std::endl;
            }
#endif
        } // mfi

        if (do_dump_theta_for_var) {
            std::string var_label = "theta";
            if (ivar == ivarU) var_label = "u";
            if (ivar == ivarV) var_label = "v";
            const std::string pf_name =
                amrex::Concatenate(dbg_realbdy_dump_theta_prefix + "_" + var_label + "_", dbg_realbdy_dump_theta_counter, 6);
            Vector<std::string> varnames{"prim_bdy","remap_bdy","remap_minus_prim"};
            WriteSingleLevelPlotfile(pf_name, mf_theta_dbg, varnames, geom, time, dbg_realbdy_dump_theta_counter);
            if (ParallelDescriptor::IOProcessor()) {
                Print() << "[DBG_REALBDY_THETA_DUMP] wrote " << pf_name << "\n";
            }
        }
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
                                       data_arr, rhs_arr, nudge_scale, dbg_nudge_x, dbg_nudge_y,
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
                               Real mag = std::sqrt( Real(dj_min*dj_min + ii*ii) );
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
                               Real mag = std::sqrt( Real(di_min*di_min + jj*jj) );
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
