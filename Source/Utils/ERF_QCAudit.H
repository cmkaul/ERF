#ifndef ERF_QCAUDIT_H
#define ERF_QCAUDIT_H

#include <string>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

bool erf_qc_audit_enabled ();
bool erf_qc_audit_point_stencil_enabled ();
bool erf_qc_audit_should_log (int step);

void erf_set_qc_audit_context (int lev, int step, amrex::Real time);

int erf_qc_audit_level ();
int erf_qc_audit_step ();
amrex::Real erf_qc_audit_time ();

void erf_audit_ring_mf (const amrex::Geometry& geom,
                        const amrex::MultiFab& mf,
                        int comp,
                        const std::string& label,
                        int lev,
                        int step,
                        amrex::Real time,
                        int ring_width,
                        const std::string& quantity_name);

void erf_audit_ring_mf (const amrex::Geometry& geom,
                        const amrex::MultiFab& mf,
                        int comp,
                        const std::string& label,
                        int lev,
                        int step,
                        amrex::Real time,
                        int ring_width,
                        const std::string& quantity_name,
                        const std::string& audit_tag);

void erf_audit_mf_ghost (const amrex::Geometry& geom,
                         const amrex::MultiFab& mf,
                         int comp,
                         const std::string& label,
                         int lev,
                         int step,
                         amrex::Real time,
                         const std::string& quantity_name);

void erf_audit_cons_ghost (const amrex::Geometry& geom,
                           const amrex::MultiFab& cons,
                           const std::string& label,
                           int lev,
                           int step,
                           amrex::Real time);

void erf_audit_warn_if_gt (const amrex::Geometry& geom,
                           const amrex::MultiFab& mf,
                           int comp,
                           const std::string& label,
                           int lev,
                           int step,
                           amrex::Real time,
                           amrex::Real threshold);

#endif
