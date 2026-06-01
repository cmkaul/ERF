#include <ERF.H>
#include <ERF_QCAudit.H>

using namespace amrex;

void ERF::advance_microphysics (int lev,
                                MultiFab& cons,
                                const Real& dt_advance,
                                const int& iteration,
                                const Real& time )
{
    if (solverChoice.moisture_type != MoistureType::None) {
        const Real audit_time = time + dt_advance;
        erf_set_qc_audit_context(lev, istep[lev], audit_time);
        erf_audit_cons_ghost(geom[lev], cons, "pre_micro_before_fillboundary_ghost",
                             lev, istep[lev], audit_time);
        if (lev > 0) {
            MultiFab& U_new = vars_new[lev][Vars::xvel];
            MultiFab& V_new = vars_new[lev][Vars::yvel];
            MultiFab& W_new = vars_new[lev][Vars::zvel];
            FillPatchFineLevel(lev, audit_time,
                               {&cons, &U_new, &V_new, &W_new},
                               {&cons, &rU_new[lev], &rV_new[lev], &rW_new[lev]},
                               base_state[lev], base_state[lev]);
        } else {
            cons.FillBoundary(geom[lev].periodicity());
        }
        erf_audit_cons_ghost(geom[lev], cons, "pre_micro_after_fillboundary_ghost",
                             lev, istep[lev], audit_time);
        micro->Update_Micro_Vars_Lev(lev, cons);
        AuditQCChanges(lev, cons, cons, "before_micro_Advance", audit_time);
        micro->Advance(lev, dt_advance, iteration, time, solverChoice, vars_new, z_phys_nd, phys_bc_type);
        AuditQCChanges(lev, cons, cons, "after_micro_Advance_before_Update_State_Vars_Lev", audit_time);
        micro->Update_State_Vars_Lev(lev, cons, *z_phys_nd[lev]);
        erf_audit_cons_ghost(geom[lev], cons, "post_micro_update_cons_ghost",
                             lev, istep[lev], audit_time);
        AuditQCChanges(lev, cons, cons, "after_micro_Update_State_Vars_Lev", audit_time);

        // Sync cons[lev-1] covered cells with the moist state just written
        // to cons[lev].  Without this, the next sub-cycle's FillPatchFineLevel
        // for level lev would pull stale (latent-heat-less) values from
        // cons[lev-1]'s coarse cells via cell-conservative interpolation,
        // causing an artificial outward heat/q flux across the lev/(lev-1)
        // boundary that drains the bubble interior.
        if (lev > 0 && solverChoice.coupling_type == CouplingType::TwoWay &&
            Microphysics::modelType(solverChoice.moisture_type) == MoistureModelType::Lagrangian) {
            AverageDownMoistStateTo(lev - 1);
        }
    }
}
