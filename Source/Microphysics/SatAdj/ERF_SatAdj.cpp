#include "ERF_SatAdj.H"
#include "ERF_QCAudit.H"

using namespace amrex;

/**
 * AdvanceSatAdj is a local valid-cell update over MFIter tileboxes.
 * It uses pressure diagnosed in Copy_State_to_Micro and holds that pressure
 * fixed inside each cell adjustment.
 * There are no stencils, face fluxes, or ghost-cell reads in this kernel.
 */
void SatAdj::AdvanceSatAdj (const SolverChoice& /*solverChoice*/)
{
    // Saturation adjustment can be disabled by solver choice, e.g. when SHOC
    // owns moist thermodynamics instead of the standalone SatAdj module.
    if (!m_do_cond) { return; }

    auto tabs  = mic_fab_vars[MicVar_SatAdj::tabs];
    const bool do_audit = erf_qc_audit_enabled();
    const int audit_lev = erf_qc_audit_level();
    const int audit_step = erf_qc_audit_step();
    const Real audit_time = erf_qc_audit_time();
    const auto audit_ba = tabs->boxArray();
    const auto audit_dm = tabs->DistributionMap();
    std::unique_ptr<MultiFab> audit_dqv;
    std::unique_ptr<MultiFab> audit_dqc;
    std::unique_ptr<MultiFab> audit_dtheta;
    std::unique_ptr<MultiFab> audit_dT;
    std::unique_ptr<MultiFab> audit_qv_before;
    std::unique_ptr<MultiFab> audit_qc_before;
    std::unique_ptr<MultiFab> audit_qv_after;
    std::unique_ptr<MultiFab> audit_qc_after;
    auto define_audit_mf = [&] (std::unique_ptr<MultiFab>& mf) {
        if (do_audit) {
            mf = std::make_unique<MultiFab>(audit_ba, audit_dm, 1, 0);
            mf->setVal(0.);
        }
    };
    define_audit_mf(audit_dqv);
    define_audit_mf(audit_dqc);
    define_audit_mf(audit_dtheta);
    define_audit_mf(audit_dT);
    define_audit_mf(audit_qv_before);
    define_audit_mf(audit_qc_before);
    define_audit_mf(audit_qv_after);
    define_audit_mf(audit_qc_after);
    auto audit_mf = [&] (const std::unique_ptr<MultiFab>& mf,
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

    // Expose for GPU
    Real d_fac_cond = m_fac_cond;
    Real rdOcp      = m_rdOcp;

    for ( MFIter mfi(*tabs,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        auto tbx = mfi.tilebox();

        auto qv_array    = mic_fab_vars[MicVar_SatAdj::qv]->array(mfi);
        auto qc_array    = mic_fab_vars[MicVar_SatAdj::qc]->array(mfi);
        auto tabs_array  = mic_fab_vars[MicVar_SatAdj::tabs]->array(mfi);
        auto theta_array = mic_fab_vars[MicVar_SatAdj::theta]->array(mfi);
        auto pres_array  = mic_fab_vars[MicVar_SatAdj::pres]->array(mfi);
        auto audit_dqv_arr       = do_audit ? audit_dqv->array(mfi) : Array4<Real>{};
        auto audit_dqc_arr       = do_audit ? audit_dqc->array(mfi) : Array4<Real>{};
        auto audit_dtheta_arr    = do_audit ? audit_dtheta->array(mfi) : Array4<Real>{};
        auto audit_dT_arr        = do_audit ? audit_dT->array(mfi) : Array4<Real>{};
        auto audit_qv_before_arr = do_audit ? audit_qv_before->array(mfi) : Array4<Real>{};
        auto audit_qc_before_arr = do_audit ? audit_qc_before->array(mfi) : Array4<Real>{};
        auto audit_qv_after_arr  = do_audit ? audit_qv_after->array(mfi) : Array4<Real>{};
        auto audit_qc_after_arr  = do_audit ? audit_qc_after->array(mfi) : Array4<Real>{};

        ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            Real T  = tabs_array(i,j,k);
            Real p  = pres_array(i,j,k);
            Real th = theta_array(i,j,k);
            Real qv = qv_array(i,j,k);
            Real qc = qc_array(i,j,k);
            const Real T0 = T;
            const Real th0 = th;
            const Real qv0 = qv;
            const Real qc0 = qc;

            AdjustSatAdjCell(d_fac_cond, rdOcp, T, p, th, qv, qc);

            tabs_array(i,j,k)  = T;
            theta_array(i,j,k) = th;
            qv_array(i,j,k)    = qv;
            qc_array(i,j,k)    = qc;

            if (do_audit) {
                audit_dqv_arr(i,j,k)       = qv - qv0;
                audit_dqc_arr(i,j,k)       = qc - qc0;
                audit_dtheta_arr(i,j,k)    = th - th0;
                audit_dT_arr(i,j,k)        = T - T0;
                audit_qv_before_arr(i,j,k) = qv0;
                audit_qc_before_arr(i,j,k) = qc0;
                audit_qv_after_arr(i,j,k)  = qv;
                audit_qc_after_arr(i,j,k)  = qc;
            }
        });
    }

    audit_mf(audit_dqv, "satadj_dqv", "dqv");
    audit_mf(audit_dqc, "satadj_dqc", "dqc");
    audit_mf(audit_dtheta, "satadj_dtheta", "dtheta");
    audit_mf(audit_dT, "satadj_dT", "dT");
    audit_mf(audit_qv_before, "satadj_qv_before", "qv_before");
    audit_mf(audit_qc_before, "satadj_qc_before", "qc_before");
    audit_mf(audit_qv_after, "satadj_qv_after", "qv_after");
    audit_mf(audit_qc_after, "satadj_qc_after", "qc_after");
    audit_internal_mf(MicVar_SatAdj::qv, "satadj_before_Copy_Micro_to_State_qv", "qv");
    audit_internal_mf(MicVar_SatAdj::qc, "satadj_before_Copy_Micro_to_State_qc", "qc");
    audit_internal_mf(MicVar_SatAdj::theta, "satadj_before_Copy_Micro_to_State_theta", "theta");
    audit_internal_mf(MicVar_SatAdj::tabs, "satadj_before_Copy_Micro_to_State_tabs", "tabs");
}
