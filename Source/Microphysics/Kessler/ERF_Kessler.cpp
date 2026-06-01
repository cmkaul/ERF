#include <ERF_EOS.H>
#include <ERF_TileNoZ.H>
#include <cmath>
#include <limits>

#include "ERF_DataStruct.H"
#include "ERF_Kessler.H"
#include "ERF_KesslerUtils.H"
#include "ERF_QCAudit.H"

using namespace amrex;

/**
 * Compute Precipitation-related Microphysics quantities.
 */

void Kessler::AdvanceKessler (const SolverChoice &solverChoice)
{
    bool do_cond = m_do_cond;
    auto tabs    = mic_fab_vars[MicVar_Kess::tabs];
    auto domain  = m_geom.Domain();
    int k_lo = domain.smallEnd(2);
    int k_hi = domain.bigEnd(2);
    const bool do_audit = erf_qc_audit_enabled();
    const int audit_lev = erf_qc_audit_level();
    const int audit_step = erf_qc_audit_step();
    const Real audit_time = erf_qc_audit_time();
    const auto audit_ba = tabs->boxArray();
    const auto audit_dm = tabs->DistributionMap();
    std::unique_ptr<MultiFab> audit_dq_vapor_to_cloud;
    std::unique_ptr<MultiFab> audit_dq_cloud_to_vapor;
    std::unique_ptr<MultiFab> audit_dq_cloud_to_rain;
    std::unique_ptr<MultiFab> audit_dq_rain_to_vapor;
    std::unique_ptr<MultiFab> audit_qv_minus_qsat;
    std::unique_ptr<MultiFab> audit_qc_before;
    std::unique_ptr<MultiFab> audit_qc_after;
    std::unique_ptr<MultiFab> audit_qv_before;
    std::unique_ptr<MultiFab> audit_qv_after;
    std::unique_ptr<MultiFab> audit_qp_before;
    std::unique_ptr<MultiFab> audit_theta_before;
    std::unique_ptr<MultiFab> audit_tabs;
    std::unique_ptr<MultiFab> audit_pressure;
    std::unique_ptr<MultiFab> audit_qsat;
    std::unique_ptr<MultiFab> audit_dtqsat;
    std::unique_ptr<MultiFab> audit_theta_delta;
    std::unique_ptr<MultiFab> audit_visit;
    auto define_audit_mf = [&] (std::unique_ptr<MultiFab>& mf) {
        if (do_audit) {
            mf = std::make_unique<MultiFab>(audit_ba, audit_dm, 1, 0);
            mf->setVal(0.);
        }
    };
    define_audit_mf(audit_dq_vapor_to_cloud);
    define_audit_mf(audit_dq_cloud_to_vapor);
    define_audit_mf(audit_dq_cloud_to_rain);
    define_audit_mf(audit_dq_rain_to_vapor);
    define_audit_mf(audit_qv_minus_qsat);
    define_audit_mf(audit_qc_before);
    define_audit_mf(audit_qc_after);
    define_audit_mf(audit_qv_before);
    define_audit_mf(audit_qv_after);
    define_audit_mf(audit_qp_before);
    define_audit_mf(audit_theta_before);
    define_audit_mf(audit_tabs);
    define_audit_mf(audit_pressure);
    define_audit_mf(audit_qsat);
    define_audit_mf(audit_dtqsat);
    define_audit_mf(audit_theta_delta);
    define_audit_mf(audit_visit);
    auto audit_source_mf = [&] (const std::unique_ptr<MultiFab>& mf,
                                const std::string& label,
                                const std::string& quantity_name) {
        if (do_audit && mf) {
            erf_audit_ring_mf(m_geom, *mf, 0, label, audit_lev, audit_step,
                              audit_time, 5, quantity_name, "QC_AUDIT_MICRO");
        }
    };
    auto audit_internal_mf = [&] (int mic_var,
                                  const std::string& label,
                                  const std::string& quantity_name) {
        if (do_audit) {
            erf_audit_ring_mf(m_geom, *mic_fab_vars[mic_var], 0, label, audit_lev,
                              audit_step, audit_time, 5, quantity_name, "QC_AUDIT_MICRO");
        }
    };
    auto audit_internal_ghost = [&] (int mic_var,
                                     const std::string& label,
                                     const std::string& quantity_name) {
        if (do_audit) {
            erf_audit_mf_ghost(m_geom, *mic_fab_vars[mic_var], 0, label, audit_lev,
                               audit_step, audit_time, quantity_name);
        }
    };
    auto dump_point_stencil = [&] (const MultiFab& trigger,
                                   const MultiFab* visit,
                                   const MultiFab* dq_vapor_to_cloud,
                                   const MultiFab* dq_cloud_to_rain,
                                   const MultiFab* theta_delta,
                                   const std::string& label) {
#if !defined(AMREX_USE_GPU)
        if (!do_audit || !erf_qc_audit_point_stencil_enabled()) { return; }
        Gpu::streamSynchronize();
        const IntVect center = trigger.maxIndex(0);
        const Box& dom = m_geom.Domain();
        const int ilo = dom.smallEnd(0);
        const int ihi = dom.bigEnd(0);
        const int jlo = dom.smallEnd(1);
        const int jhi = dom.bigEnd(1);
        const int klo = dom.smallEnd(2);
        const int khi = dom.bigEnd(2);
        const int ring_width = 5;
        IntVect normal(0);
        if (center[0] < ilo + ring_width) {
            normal[0] = -1;
        } else if (center[0] > ihi - ring_width) {
            normal[0] = 1;
        } else if (center[1] < jlo + ring_width) {
            normal[1] = -1;
        } else if (center[1] > jhi - ring_width) {
            normal[1] = 1;
        }
        auto read_mf = [] (const MultiFab* mf, const IntVect& iv, int comp = 0) -> Real {
            if (mf == nullptr || comp >= mf->nComp()) { return std::numeric_limits<Real>::quiet_NaN(); }
            const IntVect ng = mf->nGrowVect();
            for (MFIter mfi(*mf); mfi.isValid(); ++mfi) {
                Box bx = mfi.validbox();
                bx.grow(ng);
                if (bx.contains(iv)) {
                    const auto arr = mf->const_array(mfi);
                    return arr(iv[0],iv[1],iv[2],comp);
                }
            }
            return std::numeric_limits<Real>::quiet_NaN();
        };
        auto point_kind = [&] (const IntVect& iv) -> const char* {
            const bool ghost = !dom.contains(iv);
            if (ghost) { return "ghost"; }
            const bool ring = (iv[0] < ilo + ring_width) || (iv[0] > ihi - ring_width) ||
                              (iv[1] < jlo + ring_width) || (iv[1] > jhi - ring_width);
            return ring ? "ring" : "interior";
        };

        for (int hn = -2; hn <= 2; ++hn) {
            IntVect horiz = center;
            horiz[0] -= hn * normal[0];
            horiz[1] -= hn * normal[1];
            for (int dk = -1; dk <= 2; ++dk) {
                IntVect iv(horiz[0], horiz[1], center[2] + dk);
                if (iv[2] < klo - 1 || iv[2] > khi) { continue; }
                const Real rho = read_mf(mic_fab_vars[MicVar_Kess::rho].get(), iv);
                const Real theta = read_mf(mic_fab_vars[MicVar_Kess::theta].get(), iv);
                const Real qv = read_mf(mic_fab_vars[MicVar_Kess::qv].get(), iv);
                const Real qc = read_mf(mic_fab_vars[MicVar_Kess::qcl].get(), iv);
                const Real qr = read_mf(mic_fab_vars[MicVar_Kess::qp].get(), iv);
                const Real tabs_val = read_mf(mic_fab_vars[MicVar_Kess::tabs].get(), iv);
                const Real pressure = read_mf(mic_fab_vars[MicVar_Kess::pres].get(), iv);
                Real qsat = zero;
                if (tabs_val > Real(0) && pressure > Real(0)) {
                    erf_qsatw(tabs_val, pressure, qsat);
                }
                Print() << std::setprecision(17)
                        << "QC_AUDIT_POINT_STENCIL"
                        << " label=" << label
                        << " lev=" << audit_lev
                        << " step=" << audit_step
                        << " time=" << audit_time
                        << " center_i=" << center[0]
                        << " center_j=" << center[1]
                        << " center_k=" << center[2]
                        << " i=" << iv[0]
                        << " j=" << iv[1]
                        << " k=" << iv[2]
                        << " point_kind=" << point_kind(iv)
                        << " rho=" << rho
                        << " theta=" << theta
                        << " T=" << tabs_val
                        << " pressure=" << pressure
                        << " qv=" << qv
                        << " qc=" << qc
                        << " qr=" << qr
                        << " qsat=" << qsat
                        << " qv_minus_qsat=" << qv - qsat
                        << " qc_over_qv=" << ((std::abs(qv) > Real(0)) ? qc / qv : zero)
                        << " visit_count=" << read_mf(visit, iv)
                        << " dq_vapor_to_cloud=" << read_mf(dq_vapor_to_cloud, iv)
                        << " dq_cloud_to_rain=" << read_mf(dq_cloud_to_rain, iv)
                        << " theta_delta=" << read_mf(theta_delta, iv)
                        << "\n";
            }
        }
#else
        amrex::ignore_unused(trigger, visit, dq_vapor_to_cloud, dq_cloud_to_rain, theta_delta, label);
#endif
    };

    audit_internal_ghost(MicVar_Kess::rho, "kessler_pre_source_ghost", "rho");
    audit_internal_ghost(MicVar_Kess::theta, "kessler_pre_source_ghost", "theta");
    audit_internal_ghost(MicVar_Kess::qv, "kessler_pre_source_ghost", "qv");
    audit_internal_ghost(MicVar_Kess::qcl, "kessler_pre_source_ghost", "qc");
    audit_internal_ghost(MicVar_Kess::qp, "kessler_pre_source_ghost", "qr");
    audit_internal_ghost(MicVar_Kess::tabs, "kessler_pre_source_ghost", "T");
    audit_internal_ghost(MicVar_Kess::pres, "kessler_pre_source_ghost", "pressure");
    std::unique_ptr<MultiFab> audit_ghost_thermo;
    if (do_audit) {
        audit_ghost_thermo = std::make_unique<MultiFab>(audit_ba, audit_dm, 3, tabs->nGrowVect());
        audit_ghost_thermo->setVal(0.);
        const IntVect ng = tabs->nGrowVect();
        for (MFIter mfi(*tabs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& gbx = mfi.growntilebox(ng);
            auto qv_array = mic_fab_vars[MicVar_Kess::qv]->const_array(mfi);
            auto qc_array = mic_fab_vars[MicVar_Kess::qcl]->const_array(mfi);
            auto tabs_array = mic_fab_vars[MicVar_Kess::tabs]->const_array(mfi);
            auto pres_array = mic_fab_vars[MicVar_Kess::pres]->const_array(mfi);
            auto audit_arr = audit_ghost_thermo->array(mfi);
            ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                Real qsat = zero;
                if (tabs_array(i,j,k) > Real(0) && pres_array(i,j,k) > Real(0)) {
                    erf_qsatw(tabs_array(i,j,k), pres_array(i,j,k), qsat);
                }
                audit_arr(i,j,k,0) = qsat;
                audit_arr(i,j,k,1) = qv_array(i,j,k) - qsat;
                audit_arr(i,j,k,2) = (std::abs(qv_array(i,j,k)) > Real(0)) ?
                                     qc_array(i,j,k) / qv_array(i,j,k) : zero;
            });
        }
        erf_audit_mf_ghost(m_geom, *audit_ghost_thermo, 0, "kessler_pre_source_ghost",
                           audit_lev, audit_step, audit_time, "qsat");
        erf_audit_mf_ghost(m_geom, *audit_ghost_thermo, 1, "kessler_pre_source_ghost",
                           audit_lev, audit_step, audit_time, "qv_minus_qsat");
        erf_audit_mf_ghost(m_geom, *audit_ghost_thermo, 2, "kessler_pre_source_ghost",
                           audit_lev, audit_step, audit_time, "qc_over_qv");
    }

    if (solverChoice.moisture_type == MoistureType::Kessler)
    {
        MultiFab fz;
        fz.define(convert(audit_ba, IntVect(0,0,1)), audit_dm, 1, 0); // No ghost cells

        Real dtn  = dt;
        Real coef = dtn/m_dzmin;

        for ( MFIter mfi(*tabs,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto qv_array    = mic_fab_vars[MicVar_Kess::qv]->array(mfi);
            auto qc_array    = mic_fab_vars[MicVar_Kess::qcl]->array(mfi);
            auto qp_array    = mic_fab_vars[MicVar_Kess::qp]->array(mfi);
            auto qt_array    = mic_fab_vars[MicVar_Kess::qt]->array(mfi);
            auto tabs_array  = mic_fab_vars[MicVar_Kess::tabs]->array(mfi);
            auto pres_array  = mic_fab_vars[MicVar_Kess::pres]->array(mfi);
            auto theta_array = mic_fab_vars[MicVar_Kess::theta]->array(mfi);
            auto rho_array   = mic_fab_vars[MicVar_Kess::rho]->array(mfi);
            auto audit_dq_vapor_to_cloud_arr = do_audit ? audit_dq_vapor_to_cloud->array(mfi) : Array4<Real>{};
            auto audit_dq_cloud_to_vapor_arr = do_audit ? audit_dq_cloud_to_vapor->array(mfi) : Array4<Real>{};
            auto audit_dq_cloud_to_rain_arr  = do_audit ? audit_dq_cloud_to_rain->array(mfi) : Array4<Real>{};
            auto audit_dq_rain_to_vapor_arr  = do_audit ? audit_dq_rain_to_vapor->array(mfi) : Array4<Real>{};
            auto audit_qv_minus_qsat_arr     = do_audit ? audit_qv_minus_qsat->array(mfi) : Array4<Real>{};
            auto audit_qc_before_arr         = do_audit ? audit_qc_before->array(mfi) : Array4<Real>{};
            auto audit_qc_after_arr          = do_audit ? audit_qc_after->array(mfi) : Array4<Real>{};
            auto audit_qv_before_arr         = do_audit ? audit_qv_before->array(mfi) : Array4<Real>{};
            auto audit_qv_after_arr          = do_audit ? audit_qv_after->array(mfi) : Array4<Real>{};
            auto audit_qp_before_arr         = do_audit ? audit_qp_before->array(mfi) : Array4<Real>{};
            auto audit_theta_before_arr      = do_audit ? audit_theta_before->array(mfi) : Array4<Real>{};
            auto audit_tabs_arr              = do_audit ? audit_tabs->array(mfi) : Array4<Real>{};
            auto audit_pressure_arr          = do_audit ? audit_pressure->array(mfi) : Array4<Real>{};
            auto audit_qsat_arr              = do_audit ? audit_qsat->array(mfi) : Array4<Real>{};
            auto audit_dtqsat_arr            = do_audit ? audit_dtqsat->array(mfi) : Array4<Real>{};
            auto audit_theta_delta_arr       = do_audit ? audit_theta_delta->array(mfi) : Array4<Real>{};
            auto audit_visit_arr             = do_audit ? audit_visit->array(mfi) : Array4<Real>{};

            auto tbx = mfi.tilebox();

            Real d_fac_cond = m_fac_cond;

            ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                qv_array(i,j,k) = std::max(Real(0), qv_array(i,j,k));
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));
                qp_array(i,j,k) = std::max(Real(0), qp_array(i,j,k));
                if (do_audit) {
                    audit_visit_arr(i,j,k) += Real(1);
                }

                const Real qv0 = qv_array(i,j,k);
                const Real qc0 = qc_array(i,j,k);
                const Real qp0 = qp_array(i,j,k);
                const Real th0 = theta_array(i,j,k);

                Real qsat_local, dtqsat_local;
                Real pressure = pres_array(i,j,k);
                // Kessler stores pressure in mbar / hPa for the qsat helpers.
                erf_qsatw(tabs_array(i,j,k), pressure, qsat_local);
                erf_dtqsatw(tabs_array(i,j,k), pressure, dtqsat_local);

                if (qsat_local <= Real(0)) {
                    amrex::Warning("qsat computed as non-positive; setting to Real(0)!");
                    qsat_local = Real(0);
                }

                const KesslerSourceTerms source_terms = kessler_warm_rain_sources(
                    qv_array(i,j,k), qc_array(i,j,k), qp_array(i,j,k), rho_array(i,j,k),
                    pressure, qsat_local, dtqsat_local, dtn, do_cond, d_fac_cond);

                qv_array(i,j,k) += -source_terms.dq_vapor_to_cloud
                                 +  source_terms.dq_cloud_to_vapor
                                 +  source_terms.dq_rain_to_vapor;
                qc_array(i,j,k) +=  source_terms.dq_vapor_to_cloud
                                 -  source_terms.dq_cloud_to_vapor
                                 -  source_terms.dq_cloud_to_rain;
                qp_array(i,j,k) +=  source_terms.dq_cloud_to_rain
                                 -  source_terms.dq_rain_to_vapor;

                Real theta_over_T = theta_array(i,j,k)/tabs_array(i,j,k);
                theta_array(i,j,k) += theta_over_T * d_fac_cond
                    * (source_terms.dq_vapor_to_cloud
                       - source_terms.dq_cloud_to_vapor
                       - source_terms.dq_rain_to_vapor);

                qv_array(i,j,k) = std::max(Real(0), qv_array(i,j,k));
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));
                qp_array(i,j,k) = std::max(Real(0), qp_array(i,j,k));

                qt_array(i,j,k) = qv_array(i,j,k) + qc_array(i,j,k);

                if (do_audit) {
                    audit_qv_minus_qsat_arr(i,j,k)     = qv0 - qsat_local;
                    audit_qsat_arr(i,j,k)               = qsat_local;
                    audit_dtqsat_arr(i,j,k)             = dtqsat_local;
                    audit_dq_vapor_to_cloud_arr(i,j,k) = source_terms.dq_vapor_to_cloud;
                    audit_dq_cloud_to_vapor_arr(i,j,k) = source_terms.dq_cloud_to_vapor;
                    audit_dq_cloud_to_rain_arr(i,j,k)  = source_terms.dq_cloud_to_rain;
                    audit_dq_rain_to_vapor_arr(i,j,k)  = source_terms.dq_rain_to_vapor;
                    audit_qc_before_arr(i,j,k)         = qc0;
                    audit_qc_after_arr(i,j,k)          = qc_array(i,j,k);
                    audit_qv_before_arr(i,j,k)         = qv0;
                    audit_qv_after_arr(i,j,k)          = qv_array(i,j,k);
                    audit_qp_before_arr(i,j,k)         = qp0;
                    audit_theta_before_arr(i,j,k)      = th0;
                    audit_tabs_arr(i,j,k)              = tabs_array(i,j,k);
                    audit_pressure_arr(i,j,k)          = pressure;
                    audit_theta_delta_arr(i,j,k)       = theta_array(i,j,k) - th0;
                }
            });
        }

        audit_source_mf(audit_qv_minus_qsat, "kessler_source_qv_minus_qsat", "qv_minus_qsat");
        audit_source_mf(audit_dq_vapor_to_cloud, "kessler_source_dq_vapor_to_cloud", "dq_vapor_to_cloud");
        audit_source_mf(audit_dq_cloud_to_vapor, "kessler_source_dq_cloud_to_vapor", "dq_cloud_to_vapor");
        audit_source_mf(audit_dq_cloud_to_rain, "kessler_source_dq_cloud_to_rain", "dq_cloud_to_rain");
        audit_source_mf(audit_dq_rain_to_vapor, "kessler_source_dq_rain_to_vapor", "dq_rain_to_vapor");
        audit_source_mf(audit_qc_before, "kessler_source_qc_before", "qc_before");
        audit_source_mf(audit_qc_after, "kessler_source_qc_after", "qc_after");
        audit_source_mf(audit_qv_before, "kessler_source_qv_before", "qv_before");
        audit_source_mf(audit_qv_after, "kessler_source_qv_after", "qv_after");
        audit_source_mf(audit_qp_before, "kessler_source_qp", "qp");
        audit_source_mf(audit_theta_before, "kessler_source_theta", "theta");
        audit_source_mf(audit_tabs, "kessler_source_T", "T");
        audit_source_mf(audit_pressure, "kessler_source_pressure", "pressure");
        audit_source_mf(audit_qsat, "kessler_source_qsat", "qsat");
        audit_source_mf(audit_dtqsat, "kessler_source_dtqsat", "dtqsat");
        audit_source_mf(audit_theta_delta, "kessler_source_theta_delta", "theta_delta");
        audit_source_mf(audit_visit, "kessler_source_visit", "visit_count");
        if (do_audit) {
            erf_audit_warn_if_gt(m_geom, *audit_visit, 0, "kessler_source_visit",
                                 audit_lev, audit_step, audit_time, Real(1));
            dump_point_stencil(*audit_dq_vapor_to_cloud, audit_visit.get(),
                               audit_dq_vapor_to_cloud.get(), audit_dq_cloud_to_rain.get(),
                               audit_theta_delta.get(), "kessler_source_dq_vapor_to_cloud_max");
            dump_point_stencil(*audit_qc_after, audit_visit.get(),
                               audit_dq_vapor_to_cloud.get(), audit_dq_cloud_to_rain.get(),
                               audit_theta_delta.get(), "kessler_source_qc_after_max");
        }

        for ( MFIter mfi(fz, TilingIfNotGPU()); mfi.isValid(); ++mfi ){
            auto rho_array = mic_fab_vars[MicVar_Kess::rho]->array(mfi);
            auto qp_array  = mic_fab_vars[MicVar_Kess::qp]->array(mfi);

            auto fz_array  = fz.array(mfi);
            const Box& tbz = mfi.tilebox();

            ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const Real rho_km1 = (k == k_lo) ? rho_array(i,j,k) : rho_array(i,j,k-1);
                const Real rho_k = (k == k_hi+1) ? rho_array(i,j,k-1) : rho_array(i,j,k);
                const Real qp_km1 = (k == k_lo) ? qp_array(i,j,k) : qp_array(i,j,k-1);
                const Real qp_k = (k == k_hi+1) ? qp_array(i,j,k-1) : qp_array(i,j,k);
                const KesslerFaceState face_state =
                    kessler_face_state(k, k_hi, rho_km1, rho_k, qp_km1, qp_k);

                const Real terminal_velocity = kessler_terminal_velocity(face_state.rho, face_state.qp);
                fz_array(i,j,k) = kessler_precip_flux(face_state.rho, terminal_velocity, face_state.qp);
            });
        }

        auto const& ma_rho_arr = mic_fab_vars[MicVar_Kess::rho]->const_arrays();
        auto const& ma_qp_arr = mic_fab_vars[MicVar_Kess::qp]->const_arrays();
        // The sedimentation CFL uses fall speed, not precipitating mass flux.
        // fz stores rho * Vt * qp for the flux divergence below, so the reduction
        // recomputes Vt from the same face state used for the flux.
        GpuTuple<Real> max_terminal_velocity = ParReduce(TypeList<ReduceOpMax>{},
                                                         TypeList<Real>{},
                                                         fz, IntVect(0),
                                                         [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
                                                         -> GpuTuple<Real>
                                                         {
                                                             const auto& rho_arr = ma_rho_arr[box_no];
                                                             const auto& qp_arr = ma_qp_arr[box_no];
                                                             const Real rho_km1 = (k == k_lo) ? rho_arr(i,j,k) : rho_arr(i,j,k-1);
                                                             const Real rho_k = (k == k_hi+1) ? rho_arr(i,j,k-1) : rho_arr(i,j,k);
                                                             const Real qp_km1 = (k == k_lo) ? qp_arr(i,j,k) : qp_arr(i,j,k-1);
                                                             const Real qp_k = (k == k_hi+1) ? qp_arr(i,j,k-1) : qp_arr(i,j,k);
                                                             const KesslerFaceState face_state =
                                                                 kessler_face_state(k, k_hi, rho_km1, rho_k, qp_km1, qp_k);
                                                             return { kessler_terminal_velocity(face_state.rho, face_state.qp) };
                                                         });
        int n_substep = kessler_num_sedimentation_substeps(get<0>(max_terminal_velocity),
                                                           dt, m_dzmin);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_substep >= 1,
                                         "Kessler: Number of precipitation substeps must be greater than 0!");
        coef /= Real(n_substep);
        dtn  /= Real(n_substep);

        for (int nsub(0); nsub<n_substep; ++nsub) {
            for ( MFIter mfi(*tabs, TilingIfNotGPU()); mfi.isValid(); ++mfi ){
                auto rho_array = mic_fab_vars[MicVar_Kess::rho]->array(mfi);
                auto qp_array  = mic_fab_vars[MicVar_Kess::qp]->array(mfi);
                auto rain_accum_array = mic_fab_vars[MicVar_Kess::rain_accum]->array(mfi);
                auto fz_array  = fz.array(mfi);

                const auto dJ_array = (m_detJ_cc) ? m_detJ_cc->const_array(mfi) : Array4<const Real>{};

                const Box& tbx = mfi.tilebox();
                const Box& tbz = mfi.tilebox(IntVect(0,0,1),IntVect(0));

                ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const Real rho_km1 = (k == k_lo) ? rho_array(i,j,k) : rho_array(i,j,k-1);
                    const Real rho_k = (k == k_hi+1) ? rho_array(i,j,k-1) : rho_array(i,j,k);
                    const Real qp_km1 = (k == k_lo) ? qp_array(i,j,k) : qp_array(i,j,k-1);
                    const Real qp_k = (k == k_hi+1) ? qp_array(i,j,k-1) : qp_array(i,j,k);
                    const int donor_k = kessler_face_donor_k(k, k_hi);
                    const KesslerFaceState face_state =
                        kessler_face_state(k, k_hi, rho_km1, rho_k, qp_km1, qp_k);

                    const Real terminal_velocity = kessler_terminal_velocity(face_state.rho, face_state.qp);
                    const Real donor_rho = rho_array(i,j,donor_k);
                    const Real donor_qp = std::max(Real(0), qp_array(i,j,donor_k));
                    const Real donor_detJ = (dJ_array) ? dJ_array(i,j,donor_k) : Real(1);
                    // The face flux uses face rho and donor qp. The limiter uses donor rho,
                    // donor qp, and donor detJ because it caps how much rain can leave the donor
                    // cell in this substep.
                    // Cap outgoing flux by the donor cell's detJ-weighted available rain water:
                    // F * dt / dz <= rho_donor * qp_donor * detJ_donor.
                    // This keeps compressed cells from losing more rain than they contain.
                    const Real max_flux = donor_rho * donor_qp * donor_detJ / coef;
                    fz_array(i,j,k) = amrex::min(
                        kessler_precip_flux(face_state.rho, terminal_velocity, face_state.qp), max_flux);

                    if(k==k_lo){
                        // Surface accumulation stores the bottom-face precipitation mass per area
                        // increment converted to liquid-water depth.
                        rain_accum_array(i,j,k) = rain_accum_array(i,j,k)
                                                + kessler_rain_accumulation_increment(fz_array(i,j,k) * dtn);
                    }
                });

                ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    Real dJinv = (dJ_array) ? Real(1)/dJ_array(i,j,k) : Real(1);

                    // Threshold local face-flux copies only. Neighboring cells share fz
                    // faces, so the cell update must not mutate fz while applying the
                    // small-value cutoff.
                    Real f_hi = fz_array(i,j,k+1);
                    Real f_lo = fz_array(i,j,k  );

                    if (kessler_is_small_sedimentation_value(f_hi)) {
                        f_hi = Real(0);
                    }
                    if (kessler_is_small_sedimentation_value(f_lo)) {
                        f_lo = Real(0);
                    }
                    Real dq_sed = kessler_sedimentation_tendency(
                        f_hi, f_lo, rho_array(i,j,k), dJinv, coef);
                    if (kessler_is_small_sedimentation_value(dq_sed)) {
                        dq_sed = Real(0);
                    }

                    qp_array(i,j,k) +=  dq_sed;
                    qp_array(i,j,k)  = std::max(Real(0), qp_array(i,j,k));
                });
            }
        }
    }

    if (solverChoice.moisture_type == MoistureType::Kessler_NoRain) {
        if (!do_cond) { return; }
        for ( MFIter mfi(*tabs,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto qv_array    = mic_fab_vars[MicVar_Kess::qv]->array(mfi);
            auto qc_array    = mic_fab_vars[MicVar_Kess::qcl]->array(mfi);
            auto qt_array    = mic_fab_vars[MicVar_Kess::qt]->array(mfi);
            auto tabs_array  = mic_fab_vars[MicVar_Kess::tabs]->array(mfi);
            auto theta_array = mic_fab_vars[MicVar_Kess::theta]->array(mfi);
            auto pres_array  = mic_fab_vars[MicVar_Kess::pres]->array(mfi);
            auto audit_dq_vapor_to_cloud_arr = do_audit ? audit_dq_vapor_to_cloud->array(mfi) : Array4<Real>{};
            auto audit_dq_cloud_to_vapor_arr = do_audit ? audit_dq_cloud_to_vapor->array(mfi) : Array4<Real>{};
            auto audit_dq_cloud_to_rain_arr  = do_audit ? audit_dq_cloud_to_rain->array(mfi) : Array4<Real>{};
            auto audit_dq_rain_to_vapor_arr  = do_audit ? audit_dq_rain_to_vapor->array(mfi) : Array4<Real>{};
            auto audit_qv_minus_qsat_arr     = do_audit ? audit_qv_minus_qsat->array(mfi) : Array4<Real>{};
            auto audit_qc_before_arr         = do_audit ? audit_qc_before->array(mfi) : Array4<Real>{};
            auto audit_qc_after_arr          = do_audit ? audit_qc_after->array(mfi) : Array4<Real>{};
            auto audit_qv_before_arr         = do_audit ? audit_qv_before->array(mfi) : Array4<Real>{};
            auto audit_qv_after_arr          = do_audit ? audit_qv_after->array(mfi) : Array4<Real>{};
            auto audit_qp_before_arr         = do_audit ? audit_qp_before->array(mfi) : Array4<Real>{};
            auto audit_theta_before_arr      = do_audit ? audit_theta_before->array(mfi) : Array4<Real>{};
            auto audit_tabs_arr              = do_audit ? audit_tabs->array(mfi) : Array4<Real>{};
            auto audit_pressure_arr          = do_audit ? audit_pressure->array(mfi) : Array4<Real>{};
            auto audit_qsat_arr              = do_audit ? audit_qsat->array(mfi) : Array4<Real>{};
            auto audit_dtqsat_arr            = do_audit ? audit_dtqsat->array(mfi) : Array4<Real>{};
            auto audit_theta_delta_arr       = do_audit ? audit_theta_delta->array(mfi) : Array4<Real>{};
            auto audit_visit_arr             = do_audit ? audit_visit->array(mfi) : Array4<Real>{};

            auto tbx = mfi.tilebox();

            Real d_fac_cond = m_fac_cond;

            ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));
                qv_array(i,j,k) = std::max(Real(0), qv_array(i,j,k));
                if (do_audit) {
                    audit_visit_arr(i,j,k) += Real(1);
                }

                const Real qv0 = qv_array(i,j,k);
                const Real qc0 = qc_array(i,j,k);
                const Real th0 = theta_array(i,j,k);

                Real qsat, dtqsat;

                Real pressure = pres_array(i,j,k);
                // Kessler stores pressure in mbar / hPa for the qsat helpers.
                erf_qsatw(tabs_array(i,j,k), pressure, qsat);
                erf_dtqsatw(tabs_array(i,j,k), pressure, dtqsat);

                const KesslerSaturationAdjustment saturation_adjustment =
                    kessler_saturation_adjustment(qv_array(i,j,k), qc_array(i,j,k), qsat, dtqsat,
                                                  do_cond, d_fac_cond);

                qv_array(i,j,k) += -saturation_adjustment.dq_vapor_to_cloud
                                 +  saturation_adjustment.dq_cloud_to_vapor;
                qc_array(i,j,k) +=  saturation_adjustment.dq_vapor_to_cloud
                                 -  saturation_adjustment.dq_cloud_to_vapor;

                Real theta_over_T = theta_array(i,j,k)/tabs_array(i,j,k);
                theta_array(i,j,k) += theta_over_T * d_fac_cond
                    * (saturation_adjustment.dq_vapor_to_cloud - saturation_adjustment.dq_cloud_to_vapor);

                qv_array(i,j,k) = std::max(Real(0), qv_array(i,j,k));
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));

                qt_array(i,j,k) = qv_array(i,j,k) + qc_array(i,j,k);

                if (do_audit) {
                    audit_qv_minus_qsat_arr(i,j,k)     = qv0 - qsat;
                    audit_qsat_arr(i,j,k)               = qsat;
                    audit_dtqsat_arr(i,j,k)             = dtqsat;
                    audit_dq_vapor_to_cloud_arr(i,j,k) = saturation_adjustment.dq_vapor_to_cloud;
                    audit_dq_cloud_to_vapor_arr(i,j,k) = saturation_adjustment.dq_cloud_to_vapor;
                    audit_dq_cloud_to_rain_arr(i,j,k)  = zero;
                    audit_dq_rain_to_vapor_arr(i,j,k)  = zero;
                    audit_qc_before_arr(i,j,k)         = qc0;
                    audit_qc_after_arr(i,j,k)          = qc_array(i,j,k);
                    audit_qv_before_arr(i,j,k)         = qv0;
                    audit_qv_after_arr(i,j,k)          = qv_array(i,j,k);
                    audit_qp_before_arr(i,j,k)         = zero;
                    audit_theta_before_arr(i,j,k)      = th0;
                    audit_tabs_arr(i,j,k)              = tabs_array(i,j,k);
                    audit_pressure_arr(i,j,k)          = pressure;
                    audit_theta_delta_arr(i,j,k)       = theta_array(i,j,k) - th0;
                }
            });
        }

        audit_source_mf(audit_qv_minus_qsat, "kessler_source_qv_minus_qsat", "qv_minus_qsat");
        audit_source_mf(audit_dq_vapor_to_cloud, "kessler_source_dq_vapor_to_cloud", "dq_vapor_to_cloud");
        audit_source_mf(audit_dq_cloud_to_vapor, "kessler_source_dq_cloud_to_vapor", "dq_cloud_to_vapor");
        audit_source_mf(audit_dq_cloud_to_rain, "kessler_source_dq_cloud_to_rain", "dq_cloud_to_rain");
        audit_source_mf(audit_dq_rain_to_vapor, "kessler_source_dq_rain_to_vapor", "dq_rain_to_vapor");
        audit_source_mf(audit_qc_before, "kessler_source_qc_before", "qc_before");
        audit_source_mf(audit_qc_after, "kessler_source_qc_after", "qc_after");
        audit_source_mf(audit_qv_before, "kessler_source_qv_before", "qv_before");
        audit_source_mf(audit_qv_after, "kessler_source_qv_after", "qv_after");
        audit_source_mf(audit_theta_before, "kessler_source_theta", "theta");
        audit_source_mf(audit_tabs, "kessler_source_T", "T");
        audit_source_mf(audit_pressure, "kessler_source_pressure", "pressure");
        audit_source_mf(audit_qsat, "kessler_source_qsat", "qsat");
        audit_source_mf(audit_dtqsat, "kessler_source_dtqsat", "dtqsat");
        audit_source_mf(audit_theta_delta, "kessler_source_theta_delta", "theta_delta");
        audit_source_mf(audit_visit, "kessler_norain_source_visit", "visit_count");
        if (do_audit) {
            erf_audit_warn_if_gt(m_geom, *audit_visit, 0, "kessler_norain_source_visit",
                                 audit_lev, audit_step, audit_time, Real(1));
            dump_point_stencil(*audit_dq_vapor_to_cloud, audit_visit.get(),
                               audit_dq_vapor_to_cloud.get(), audit_dq_cloud_to_rain.get(),
                               audit_theta_delta.get(), "kessler_norain_source_dq_vapor_to_cloud_max");
            dump_point_stencil(*audit_qc_after, audit_visit.get(),
                               audit_dq_vapor_to_cloud.get(), audit_dq_cloud_to_rain.get(),
                               audit_theta_delta.get(), "kessler_norain_source_qc_after_max");
        }
    }

    audit_internal_mf(MicVar_Kess::qv, "kessler_before_Copy_Micro_to_State_qv", "qv");
    audit_internal_mf(MicVar_Kess::qcl, "kessler_before_Copy_Micro_to_State_qc", "qc");
    audit_internal_mf(MicVar_Kess::qp, "kessler_before_Copy_Micro_to_State_qp", "qp");
    audit_internal_mf(MicVar_Kess::theta, "kessler_before_Copy_Micro_to_State_theta", "theta");
}
