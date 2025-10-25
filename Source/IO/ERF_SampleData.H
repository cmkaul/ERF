#ifndef ERF_SAMPLEDATA_H
#define ERF_SAMPLEDATA_H

#include <memory>

#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>

#include <ERF_IndexDefines.H>
#include <ERF_EOS.H>
#include <ERF_Container.H>

#include <iostream>

/*
 * LineSampler Namelist Options (erf namespace)
 *
 * The following options control line sampling in ERF:
 *
 * 1. Integer index-based lines (backward compatible):
 *    - sample_line_lo:   int array, shape (N_lines*3), lower endpoint (i,j,k) for each line
 *    - sample_line_hi:   int array, shape (N_lines*3), upper endpoint (i,j,k) for each line
 *    - sample_line_dir:  int array, shape (N_lines), direction (0=x, 1=y, 2=z) for each line
 *    - sample_line_name: string array, optional, names for each line (default: plt_line_#####)
 *
 * 2. Physical coordinate-based lines (preferred for new use):
 *    - sample_line_phys_lo: real array, shape (N_lines*3), lower endpoint (lat, lon, z_phys) for each line
 *    - sample_line_phys_hi: real array, shape (N_lines*3), upper endpoint (lat, lon, z_phys) for each line
 *      (lat, lon in degrees, z_phys in meters)
 *    - sample_line_name: string array, optional, names for each line (default: plt_line_#####)
 *      (If both index and physical options are present, physical is used.)
 *
 * 3. Variable selection and output:
 *    - line_sampling_vars: string array, variables to sample (e.g., density, x_velocity, theta, ...)
 *    - line_sampling_text_output: bool, if true, write ASCII output for each variable/line
 *
 * Notes:
 *  - Both vertical and horizontal lines are supported.
 *  - For physical coordinates, the code finds the nearest (i,j,k) for each endpoint using lat/lon/z_phys MultiFabs.
 *  - Verbose output is printed during setup, showing requested and actual endpoints.
 *  - All options are under the 'erf' ParmParse namespace.
 */
struct LineSampler
{
    // Pass lat_m, lon_m, z_phys_cc as optional pointers (nullptr if not used)
    LineSampler (
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>* lat_m_in = nullptr,
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>* lon_m_in = nullptr,
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>* z_phys_cc_in = nullptr)
    {
        amrex::ParmParse pp("erf");

        // Count number of lo and hi points define the line
        int n_line_lo  = pp.countval("sample_line_lo") / AMREX_SPACEDIM;
        int n_line_hi  = pp.countval("sample_line_hi") / AMREX_SPACEDIM;
        int n_line_dir = pp.countval("sample_line_dir");

        // New: support for (lat, lon, z_phys) input
        int n_line_phys_lo = pp.countval("sample_line_phys_lo") / AMREX_SPACEDIM;
        int n_line_phys_hi = pp.countval("sample_line_phys_hi") / AMREX_SPACEDIM;
        bool use_phys = (n_line_phys_lo > 0 && n_line_phys_lo == n_line_phys_hi);

        // Validation: check consistency based on which mode is used
        if (use_phys) {
            AMREX_ALWAYS_ASSERT( n_line_phys_lo == n_line_dir );
        } else if (n_line_lo > 0) {
            AMREX_ALWAYS_ASSERT( (n_line_lo==n_line_hi ) && (n_line_lo==n_line_dir) );
        }

        // Helper lambda for nearest neighbor search in MultiFabs
        // Strategy: First find best horizontal (lat,lon) match, then find nearest z at that location
        auto find_ijk_from_latlonz = [](amrex::Real lat, amrex::Real lon, amrex::Real z_phys,
                                        const amrex::MultiFab* latmf, const amrex::MultiFab* lonmf, const amrex::MultiFab* zmf) -> amrex::IntVect {
            amrex::IntVect best_ijk(0,0,0);
            amrex::Real best_lat = 0, best_lon = 0, best_z = 0;
            amrex::Real min_z = 1e30, max_z = -1e30;
            
            // Step 1: Find best horizontal (i,j) location by minimizing lat/lon distance
            // Scan all FABs and use MPI reduction to find global best
            struct HorizResult { amrex::Real dist; int i; int j; };
            HorizResult local_best{1e30, 0, 0};
            
            for (amrex::MFIter mfi(*latmf); mfi.isValid(); ++mfi) {
                const auto& latarr = latmf->array(mfi);
                const auto& lonarr = lonmf->array(mfi);
                const amrex::Box& bx = mfi.validbox();
                
                // Scan k=0 to find horizontal position (assuming lat/lon don't vary with k)
                int k = bx.smallEnd(2);
                for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                    for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                        amrex::Real dlat = latarr(i,j,k) - lat;
                        amrex::Real dlon = lonarr(i,j,k) - lon;
                        amrex::Real horiz_dist = dlat*dlat + dlon*dlon;
                        if (horiz_dist < local_best.dist) {
                            local_best = {horiz_dist, i, j};
                        }
                    }
                }
            }
            
            // MPI reduction to find global best horizontal position
            amrex::Real global_min_dist = local_best.dist;
            amrex::ParallelDescriptor::ReduceRealMin(global_min_dist);
            int best_i = local_best.i;
            int best_j = local_best.j;
            if (local_best.dist > global_min_dist) {
                // This rank doesn't have the best match, use dummy values
                best_i = -1;
                best_j = -1;
            }
            // Broadcast best (i,j) from the rank that has it
            int rank_with_best = (local_best.dist == global_min_dist) ? amrex::ParallelDescriptor::MyProc() : -1;
            amrex::ParallelDescriptor::ReduceIntMax(rank_with_best);
            amrex::ParallelDescriptor::Bcast(&best_i, 1, rank_with_best);
            amrex::ParallelDescriptor::Bcast(&best_j, 1, rank_with_best);
            
            // Step 2: At the best (i,j), find the k-level with closest z_phys
            // Scan ALL FABs to find z range and best k
            struct VertResult { amrex::Real dist; int k; amrex::Real z; amrex::Real lat; amrex::Real lon; };
            VertResult local_vert{1e30, 0, 0.0, 0.0, 0.0};
            
            for (amrex::MFIter mfi(*zmf); mfi.isValid(); ++mfi) {
                const auto& latarr = latmf->array(mfi);
                const auto& lonarr = lonmf->array(mfi);
                const auto& zarr   = zmf->array(mfi);
                const amrex::Box& bx = mfi.validbox();
                
                // Check if this FAB contains our (best_i, best_j) in horizontal
                amrex::IntVect test_point(best_i, best_j, bx.smallEnd(2));
                if (bx.contains(test_point)) {
                    for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
                        amrex::Real z_here = zarr(best_i, best_j, k);
                        min_z = std::min(min_z, z_here);
                        max_z = std::max(max_z, z_here);
                        
                        amrex::Real dz = std::abs(z_here - z_phys);
                        if (dz < local_vert.dist) {
                            local_vert = {dz, k, z_here, latarr(best_i, best_j, k), lonarr(best_i, best_j, k)};
                        }
                    }
                }
            }
            
            // MPI reduction to find global z range and best k
            amrex::ParallelDescriptor::ReduceRealMin(min_z);
            amrex::ParallelDescriptor::ReduceRealMax(max_z);
            amrex::Real global_min_z_dist = local_vert.dist;
            amrex::ParallelDescriptor::ReduceRealMin(global_min_z_dist);
            
            int best_k = local_vert.k;
            best_z = local_vert.z;
            best_lat = local_vert.lat;
            best_lon = local_vert.lon;
            
            // Broadcast best k info from rank that has it
            int rank_with_best_k = (local_vert.dist == global_min_z_dist && local_vert.dist < 1e29) ? amrex::ParallelDescriptor::MyProc() : -1;
            amrex::ParallelDescriptor::ReduceIntMax(rank_with_best_k);
            if (rank_with_best_k >= 0) {
                amrex::ParallelDescriptor::Bcast(&best_k, 1, rank_with_best_k);
                amrex::ParallelDescriptor::Bcast(&best_z, 1, rank_with_best_k);
                amrex::ParallelDescriptor::Bcast(&best_lat, 1, rank_with_best_k);
                amrex::ParallelDescriptor::Bcast(&best_lon, 1, rank_with_best_k);
            }
            
            best_ijk = amrex::IntVect(best_i, best_j, best_k);
            amrex::Print() << "  [z_phys range in grid: " << min_z << " to " << max_z 
                          << ", matched cell has (lat,lon,z_phys)=(" 
                          << best_lat << "," << best_lon << "," << best_z << ")]" << std::endl;
            return best_ijk;
        };

        if (use_phys) {
            // Only proceed if all required MultiFabs are provided
            if (!lat_m_in || !lon_m_in || !z_phys_cc_in) {
                amrex::Abort("LineSampler: lat_m, lon_m, z_phys_cc MultiFabs must be provided for physical line sampling!");
            }
            // Parse direction array first (needed for consistency checks)
            m_dir.resize(n_line_dir);
            pp.queryarr("sample_line_dir",m_dir,0,n_line_dir);
            
            int lev = 0;
            amrex::Vector<amrex::Real> phys_lo; phys_lo.resize(n_line_phys_lo*AMREX_SPACEDIM);
            amrex::Vector<amrex::IntVect> iv_lo; iv_lo.resize(n_line_phys_lo);
            pp.queryarr("sample_line_phys_lo",phys_lo,0,n_line_phys_lo*AMREX_SPACEDIM);
            for (int i(0); i < n_line_phys_lo; i++) {
                amrex::Real lat = phys_lo[AMREX_SPACEDIM*i+0];
                amrex::Real lon = phys_lo[AMREX_SPACEDIM*i+1];
                amrex::Real z_phys = phys_lo[AMREX_SPACEDIM*i+2];
                amrex::IntVect iv = find_ijk_from_latlonz(lat, lon, z_phys,
                    (*lat_m_in)[lev].get(), (*lon_m_in)[lev].get(), (*z_phys_cc_in)[lev].get());
                iv_lo[i] = iv;
                amrex::Print() << "LineSampler: requested lo (lat,lon,z_phys)=(" << lat << "," << lon << "," << z_phys << ") -> (i,j,k)=" << iv << std::endl;
            }
            amrex::Vector<amrex::Real> phys_hi; phys_hi.resize(n_line_phys_hi*AMREX_SPACEDIM);
            amrex::Vector<amrex::IntVect> iv_hi; iv_hi.resize(n_line_phys_hi);
            pp.queryarr("sample_line_phys_hi",phys_hi,0,n_line_phys_hi*AMREX_SPACEDIM);
            for (int i(0); i < n_line_phys_hi; i++) {
                amrex::Real lat = phys_hi[AMREX_SPACEDIM*i+0];
                amrex::Real lon = phys_hi[AMREX_SPACEDIM*i+1];
                amrex::Real z_phys = phys_hi[AMREX_SPACEDIM*i+2];
                amrex::IntVect iv = find_ijk_from_latlonz(lat, lon, z_phys,
                    (*lat_m_in)[lev].get(), (*lon_m_in)[lev].get(), (*z_phys_cc_in)[lev].get());
                iv_hi[i] = iv;
                amrex::Print() << "LineSampler: requested hi (lat,lon,z_phys)=(" << lat << "," << lon << "," << z_phys << ") -> (i,j,k)=" << iv << std::endl;
            }
            m_bnd_bx.resize(n_line_phys_lo);
            for (int i = 0; i < n_line_phys_hi; i++){
                amrex::Box lbx(iv_lo[i],iv_hi[i]);
                m_bnd_bx[i] = lbx;
                amrex::Print() << "LineSampler: created line box from " << iv_lo[i] << " to " << iv_hi[i] << std::endl;
                // Consistency check: only one index should change along the specified direction
                int dir = m_dir.size() > i ? m_dir[i] : -1;
                if (dir >= 0 && dir < AMREX_SPACEDIM) {
                    bool consistent = true;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        if (d == dir) {
                            if (iv_lo[i][d] == iv_hi[i][d]) consistent = false;
                        } else {
                            if (iv_lo[i][d] != iv_hi[i][d]) consistent = false;
                        }
                    }
                    if (!consistent) {
                        amrex::Print() << "WARNING: sample_line_phys endpoints for line " << i << " are not consistent with sample_line_dir=" << dir << "." << std::endl;
                    }
                }
            }
            // Parse names
            std::string name_base = "plt_line_";
            m_name.resize(n_line_phys_lo);
            int n_names = pp.countval("sample_line_name");
            if (n_names > 0) {
                AMREX_ALWAYS_ASSERT( n_names==n_line_phys_lo );
                pp.queryarr("sample_line_name",m_name,0,n_names);
            } else {
                for (int iline(0); iline<n_line_phys_lo; ++iline) {
                    m_name[iline] = amrex::Concatenate(name_base, iline , 5);
                }
            }
            m_lev.resize(n_line_phys_lo,0);
            m_ls_mf.resize(n_line_phys_lo);
        } else if (n_line_lo > 0) {
            // Parse lo
            amrex::Vector<int> idx_lo; idx_lo.resize(n_line_lo*AMREX_SPACEDIM);
            amrex::Vector<amrex::IntVect> iv_lo; iv_lo.resize(n_line_lo);
            pp.queryarr("sample_line_lo",idx_lo,0,n_line_lo*AMREX_SPACEDIM);
            for (int i(0); i < n_line_lo; i++) {
                amrex::IntVect iv(idx_lo[AMREX_SPACEDIM*i+0],
                                  idx_lo[AMREX_SPACEDIM*i+1],
                                  idx_lo[AMREX_SPACEDIM*i+2]);
                iv_lo[i] = iv;
            }
            // Parse hi
            amrex::Vector<int> idx_hi; idx_hi.resize(n_line_hi*AMREX_SPACEDIM);
            amrex::Vector<amrex::IntVect> iv_hi; iv_hi.resize(n_line_hi);
            pp.queryarr("sample_line_hi",idx_hi,0,n_line_hi*AMREX_SPACEDIM);
            for (int i(0); i < n_line_hi; i++) {
                amrex::IntVect iv(idx_hi[AMREX_SPACEDIM*i+0],
                                  idx_hi[AMREX_SPACEDIM*i+1],
                                  idx_hi[AMREX_SPACEDIM*i+2]);
                iv_hi[i] = iv;
            }
            m_bnd_bx.resize(n_line_lo);
            for (int i = 0; i < n_line_hi; i++){
                amrex::Box lbx(iv_lo[i],iv_hi[i]);
                m_bnd_bx[i] = lbx;
            }
            m_dir.resize(n_line_dir);
            pp.queryarr("sample_line_dir",m_dir,0,n_line_dir);
            // Parse names
            std::string name_base = "plt_line_";
            m_name.resize(n_line_lo);
            int n_names = pp.countval("sample_line_name");
            if (n_names > 0) {
                AMREX_ALWAYS_ASSERT( n_names==n_line_lo );
                pp.queryarr("sample_line_name",m_name,0,n_names);
            } else {
                for (int iline(0); iline<n_line_lo; ++iline) {
                    m_name[iline] = amrex::Concatenate(name_base, iline , 5);
                }
            }
            m_lev.resize(n_line_dir,0);
            m_ls_mf.resize(n_line_lo);
            // Consistency check: only one index should change along the specified direction
            for (int i = 0; i < n_line_lo; ++i) {
                int dir = m_dir.size() > i ? m_dir[i] : -1;
                if (dir >= 0 && dir < AMREX_SPACEDIM) {
                    bool consistent = true;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        if (d == dir) {
                            if (iv_lo[i][d] == iv_hi[i][d]) consistent = false;
                        } else {
                            if (iv_lo[i][d] != iv_hi[i][d]) consistent = false;
                        }
                    }
                    if (!consistent) {
                        amrex::Print() << "WARNING: sample_line_lo/hi endpoints for line " << i << " are not consistent with sample_line_dir=" << dir << "." << std::endl;
                    }
                }
            }
        }

        // Get requested vars
        if (pp.countval("line_sampling_vars") > 0) {
            m_varnames.clear();
            amrex::Vector<std::string> requested_vars;
            pp.queryarr("line_sampling_vars",requested_vars);
            amrex::Print() << "Selected line sampling vars :";
            if (containerHasElement(requested_vars, "density")) {
                m_varnames.push_back("density");
                amrex::Print() << " " << "density";
            }
            if (containerHasElement(requested_vars, "x_velocity")) {
                m_varnames.push_back("x_velocity");
                amrex::Print() << " " << "x_velocity";
            }
            if (containerHasElement(requested_vars, "y_velocity")) {
                m_varnames.push_back("y_velocity");
                amrex::Print() << " " << "y_velocity";
            }
            if (containerHasElement(requested_vars, "z_velocity")) {
                m_varnames.push_back("z_velocity");
                amrex::Print() << " " << "z_velocity";
            }
            if (containerHasElement(requested_vars, "magvel")) {
                m_varnames.push_back("magvel");
                amrex::Print() << " " << "magvel";
            }
            if (containerHasElement(requested_vars, "theta")) {
                m_varnames.push_back("theta");
                amrex::Print() << " " << "theta";
            }
            if (containerHasElement(requested_vars, "qv")) {
                m_varnames.push_back("qv");
                amrex::Print() << " " << "qv";
            }
            if (containerHasElement(requested_vars, "qc")) {
                m_varnames.push_back("qc");
                amrex::Print() << " " << "qc";
            }
            if (containerHasElement(requested_vars, "pressure")) {
                m_varnames.push_back("pressure");
                amrex::Print() << " " << "pressure";
            }
            amrex::Print() << std::endl;
        }

        // Write outputs to text files, one file per variable, with all
        // times appended to the same file
        pp.query("line_sampling_text_output",m_write_ascii);
    int n_lines = use_phys ? n_line_phys_lo : n_line_lo;
        if (m_write_ascii && amrex::ParallelDescriptor::IOProcessor()) {
            int nvar = static_cast<int>(m_varnames.size());
            m_datastream.resize(n_lines * nvar);
            int i = 0;
            for (int iline(0); iline<n_lines; ++iline) {
                for (int ivar(0); ivar<nvar; ++ivar) {
                    std::string filename = m_name[iline] + "." + m_varnames[ivar];
                    m_datastream[i] = std::make_unique<std::fstream>();
                    m_datastream[i]->open(filename.c_str(),std::ios::out|std::ios::app);
                    if (!m_datastream[i]->good()) {
                        amrex::FileOpenFailed(filename);
                    }
                    i++;
                }
            }
        }
    }

    void
    write_coords (amrex::Vector<std::unique_ptr<amrex::MultiFab> >& z_phys_cc)
    {
        if (!m_write_ascii) return;

        amrex::Print() << "Writing out line coordinates to text" << std::endl;

        for (int lev(0); lev < z_phys_cc.size(); ++lev) {
            // Write one text file per level
            std::ofstream outfile;
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::string fname = amrex::Concatenate("plt_line_lev", lev, 1);
                fname += ".zcc";
                outfile.open(fname);

                if (!outfile.is_open()) {
                    amrex::AllPrint() << "Could not open " << fname << std::endl;
                }
            }

            // Loop over each line
            int nline = static_cast<int>(m_ls_mf.size());
            for (int iline(0); iline<nline; ++iline) {
                int dir = m_dir[iline];
                amrex::Box bnd_bx = m_bnd_bx[iline];
                amrex::IntVect first_cell = bnd_bx.smallEnd();

                // Create multifab with "sampled" z_phys values
                amrex::MultiFab line_coords_mf = get_line_data(
                    *z_phys_cc[lev], dir, first_cell, bnd_bx
                );

                // Convert multifab to vector
                amrex::Gpu::HostVector<amrex::Real> vec = sumToLine(
                    line_coords_mf, 0, 1, bnd_bx, dir
                );

                // Append to file
                if (amrex::ParallelDescriptor::IOProcessor()) {
                    for (const auto& zval : vec) {
                        outfile << " " << zval;
                    }
                    outfile << std::endl;
                }
            } // line loop

            outfile.close();
        } // level loop
    }

    void
    get_sample_data (amrex::Vector<amrex::Geometry>& /*geom*/,
                     amrex::Vector<amrex::Vector<amrex::MultiFab>>& vars_new)
    {
        int nlev  = static_cast<int>(vars_new.size());
        int nline = static_cast<int>(m_bnd_bx.size());
        int ncomp = static_cast<int>(m_varnames.size());

        int qv_comp = -1;

        // Loop over each line
        for (int iline(0); iline<nline; ++iline) {
            int dir = m_dir[iline];
            amrex::Box bnd_bx   = m_bnd_bx[iline];
            amrex::IntVect cell = bnd_bx.smallEnd();

            // Search each level to get the finest data possible
            for (int ilev(nlev-1); ilev>=0; --ilev) {

                // Construct CC velocities
                amrex::MultiFab mf_cc_vel;
                auto ba = vars_new[ilev][Vars::cons].boxArray();
                auto dm = vars_new[ilev][Vars::cons].DistributionMap();
                mf_cc_vel.define(ba, dm, AMREX_SPACEDIM, amrex::IntVect(1,1,1));
                average_face_to_cellcenter(mf_cc_vel,0,
                                           amrex::Array<const amrex::MultiFab*,3>{&vars_new[ilev][Vars::xvel],
                                                                                  &vars_new[ilev][Vars::yvel],
                                                                                  &vars_new[ilev][Vars::zvel]});

                // Construct vector of MFs holding T and WSP
                amrex::MultiFab mf_cc_data;
                mf_cc_data.define(ba, dm, ncomp, 1);

                int mf_comp = 0;

                if (containerHasElement(m_varnames, "density")) {
                    amrex::MultiFab::Copy(mf_cc_data, vars_new[ilev][Vars::cons], Rho_comp, mf_comp, 1, 0);
                    mf_comp += 1;
                }

                if (containerHasElement(m_varnames, "x_velocity")) {
                    amrex::MultiFab::Copy(mf_cc_data, mf_cc_vel, 0, mf_comp, 1, 0);
                    mf_comp += 1;
                }
                if (containerHasElement(m_varnames, "y_velocity")) {
                    amrex::MultiFab::Copy(mf_cc_data, mf_cc_vel, 1, mf_comp, 1, 0);
                    mf_comp += 1;
                }
                if (containerHasElement(m_varnames, "z_velocity")) {
                    amrex::MultiFab::Copy(mf_cc_data, mf_cc_vel, 2, mf_comp, 1, 0);
                    mf_comp += 1;
                }

                if (containerHasElement(m_varnames, "magvel")) {
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                    for (amrex::MFIter mfi(mf_cc_data, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                        const amrex::Box& tbx = mfi.tilebox();
                        auto const& dfab = mf_cc_data.array(mfi);
                        auto const& vfab = mf_cc_vel.array(mfi);

                        amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        {
                            dfab(i,j,k,mf_comp) = std::sqrt(vfab(i,j,k,0)*vfab(i,j,k,0)
                                                          + vfab(i,j,k,1)*vfab(i,j,k,1)
                                                          + vfab(i,j,k,2)*vfab(i,j,k,2)) ;
                        });
                    }
                    mf_comp += 1;
                }

                if (containerHasElement(m_varnames, "theta")) {
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                    for (amrex::MFIter mfi(mf_cc_data, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                        const amrex::Box& tbx = mfi.tilebox();
                        auto const& dfab = mf_cc_data.array(mfi);
                        auto const& cfab = vars_new[ilev][Vars::cons].array(mfi);

                        amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        {
                            dfab(i,j,k,mf_comp) = cfab(i,j,k,RhoTheta_comp) / cfab(i,j,k,Rho_comp);
                        });
                    }
                    mf_comp += 1;
                }

                if (containerHasElement(m_varnames, "qv")) {
                    // if qv is requested, assume that we have moisture
                    if (qv_comp >= 0) AMREX_ALWAYS_ASSERT(qv_comp == mf_comp);
                    qv_comp = mf_comp;
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                    for (amrex::MFIter mfi(mf_cc_data, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                        const amrex::Box& tbx = mfi.tilebox();
                        auto const& dfab = mf_cc_data.array(mfi);
                        auto const& cfab = vars_new[ilev][Vars::cons].array(mfi);

                        amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        {
                            dfab(i,j,k,mf_comp) = cfab(i,j,k,RhoQ1_comp) / cfab(i,j,k,Rho_comp);
                        });
                    }
                    mf_comp += 1;
                }
                if (containerHasElement(m_varnames, "qc")) {
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                    for (amrex::MFIter mfi(mf_cc_data, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                        const amrex::Box& tbx = mfi.tilebox();
                        auto const& dfab = mf_cc_data.array(mfi);
                        auto const& cfab = vars_new[ilev][Vars::cons].array(mfi);

                        amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        {
                            dfab(i,j,k,mf_comp) = cfab(i,j,k,RhoQ2_comp) / cfab(i,j,k,Rho_comp);
                        });
                    }
                    mf_comp += 1;
                }

                if (containerHasElement(m_varnames, "pressure") && (qv_comp >= 0)) {
                    // if qv is requested, assume that we have moisture
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                    for (amrex::MFIter mfi(mf_cc_data, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                        const amrex::Box& tbx = mfi.tilebox();
                        auto const& dfab = mf_cc_data.array(mfi);
                        auto const& cfab = vars_new[ilev][Vars::cons].array(mfi);

                        amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        {
                            dfab(i,j,k,mf_comp) = getPgivenRTh(cfab(i,j,k,RhoTheta_comp),
                                                               dfab(i,j,k,qv_comp));
                        });
                    }
                    mf_comp += 1;
                }
                else if (containerHasElement(m_varnames, "pressure")) {
                    // pressure requested w/o qv, assume dry
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                    for (amrex::MFIter mfi(mf_cc_data, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                        const amrex::Box& tbx = mfi.tilebox();
                        auto const& dfab = mf_cc_data.array(mfi);
                        auto const& cfab = vars_new[ilev][Vars::cons].array(mfi);

                        amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        {
                            dfab(i,j,k,mf_comp) = getPgivenRTh(cfab(i,j,k,RhoTheta_comp));
                        });
                    }
                    mf_comp += 1;
                }

                  m_lev[iline] = ilev;
                m_ls_mf[iline] = get_line_data(mf_cc_data, dir, cell, bnd_bx);

                // We can stop if we got the entire line
                auto min_bnd_bx = m_ls_mf[iline].boxArray().minimalBox();
                if (bnd_bx == min_bnd_bx) { break; }

            } // ilev
        }// iline
    }

    void
    write_sample_data (amrex::Vector<amrex::Real>& time,
                       amrex::Vector<int>& level_steps,
                       amrex::Vector<amrex::IntVect>& ref_ratio,
                       amrex::Vector<amrex::Geometry>& geom)
    {
        if (m_write_ascii) {
            write_line_ascii(time);
        } else {
            write_line_plotfile(time, level_steps, ref_ratio, geom);
        }
    }

    void
    write_line_ascii (amrex::Vector<amrex::Real>& time)
    {
        // same as definitions in ERF.H
        constexpr int datwidth = 14;
        constexpr int datprecision = 6;
        constexpr int timeprecision = 13;

        int nline = static_cast<int>(m_ls_mf.size());
        int nvar  = static_cast<int>(m_varnames.size());
        for (int iline(0); iline<nline; ++iline) {
            int dir = m_dir[iline];
            int lev = m_lev[iline];
            amrex::Real m_time = time[lev];
            amrex::Box m_dom = m_bnd_bx[iline];

            for (int ivar(0); ivar<nvar; ++ivar) {
                // Convert multifab to vector
                amrex::Gpu::HostVector<amrex::Real> vec = sumToLine(m_ls_mf[iline], ivar, 1, m_dom, dir);

                if (amrex::ParallelDescriptor::IOProcessor()) {
                    int ifile = iline*nvar + ivar;
                    std::ostream& fs = *m_datastream[ifile];
                    fs << std::setw(datwidth) << std::setprecision(timeprecision) << m_time
                       << std::setw(datwidth) << std::setprecision(datprecision);
                    for (const auto& val : vec) {
                        fs << " " << val;
                    }
                    fs << std::endl;
                }
            }
        }
    }

    void
    write_line_plotfile (amrex::Vector<amrex::Real>& time,
                         amrex::Vector<int>& level_steps,
                         amrex::Vector<amrex::IntVect>& ref_ratio,
                         amrex::Vector<amrex::Geometry>& geom)
    {
        int nline = static_cast<int>(m_ls_mf.size());
        for (int iline(0); iline<nline; ++iline) {
            // Data members that can be used as-is
            int dir = m_dir[iline];
            int lev = m_lev[iline];
            amrex::Real m_time = time[lev];
            amrex::Vector<int> m_level_steps = {level_steps[lev]};
            amrex::Vector<amrex::IntVect> m_ref_ratio  = {ref_ratio[lev]};

            // Create modified geometry object corresponding to the line
            auto plo = geom[lev].ProbLo();
            auto dx  = geom[lev].CellSize();
            amrex::Vector<amrex::Geometry> m_geom; m_geom.resize(1);
            amrex::Vector<int> is_per(AMREX_SPACEDIM,0);
            amrex::Box m_dom = m_bnd_bx[iline];
            amrex::RealBox m_rb;
            for (int d(0); d<AMREX_SPACEDIM; ++d) {
                amrex::Real offset = (d==dir) ? 0 : 0.5;
                amrex::Real lo = plo[d] + ( m_dom.smallEnd(d) - offset ) * dx[d];
                amrex::Real hi = plo[d] + ( m_dom.bigEnd(d)   + offset ) * dx[d];

                m_rb.setLo(d,lo);
                m_rb.setHi(d,hi);

                is_per[d] = geom[lev].isPeriodic(d);
            }
            m_geom[0].define(m_dom, &m_rb, geom[lev].Coord(), is_per.data());

            // Create plotfile name
            std::string name_line = m_name[iline];
            name_line += "_step_";
            std::string plotfilename = amrex::Concatenate(name_line, m_level_steps[0], 5);

            // Get the data
            amrex::Vector<const amrex::MultiFab*> mf = {&(m_ls_mf[iline])};

            // Write each line
            WriteMultiLevelPlotfile(plotfilename, 1, mf,
                                    m_varnames, m_geom, m_time,
                                    m_level_steps, m_ref_ratio);
        }
    }

    amrex::Vector<int> m_dir;
    amrex::Vector<int> m_lev;
    amrex::Vector<amrex::Box> m_bnd_bx;
    amrex::Vector<amrex::MultiFab> m_ls_mf;
    amrex::Vector<std::string> m_name;

    bool m_write_ascii{false};
    amrex::Vector<std::string> m_varnames {"magvel","theta"};
    amrex::Vector<std::unique_ptr<std::fstream> > m_datastream;
};


struct PlaneSampler
{
    PlaneSampler ()
    {
        amrex::ParmParse pp("erf");

        // Count number of lo and hi points define the plane
        int n_plane_lo  = pp.countval("sample_plane_lo") / AMREX_SPACEDIM;
        int n_plane_hi  = pp.countval("sample_plane_hi") / AMREX_SPACEDIM;
        int n_plane_dir = pp.countval("sample_plane_dir");
        AMREX_ALWAYS_ASSERT( (n_plane_lo==n_plane_hi ) &&
                             (n_plane_lo==n_plane_dir) );

        // Parse the data
        if (n_plane_lo > 0) {
            // Parse lo
            amrex::Vector<amrex::Real> r_lo; r_lo.resize(n_plane_lo*AMREX_SPACEDIM);
            amrex::Vector<amrex::Vector<amrex::Real>> rv_lo;
            pp.queryarr("sample_plane_lo",r_lo,0,n_plane_lo*AMREX_SPACEDIM);
            for (int i(0); i < n_plane_lo; i++) {
                amrex::Vector<amrex::Real> rv = {r_lo[AMREX_SPACEDIM*i+0],
                                                 r_lo[AMREX_SPACEDIM*i+1],
                                                 r_lo[AMREX_SPACEDIM*i+2]};
                rv_lo.push_back(rv);
            }

            // Parse hi
            amrex::Vector<amrex::Real> r_hi; r_hi.resize(n_plane_hi*AMREX_SPACEDIM);
            amrex::Vector<amrex::Vector<amrex::Real>> rv_hi;
            pp.queryarr("sample_plane_hi",r_hi,0,n_plane_hi*AMREX_SPACEDIM);
            for (int i(0); i < n_plane_hi; i++) {
                amrex::Vector<amrex::Real> rv = {r_hi[AMREX_SPACEDIM*i+0],
                                                 r_hi[AMREX_SPACEDIM*i+1],
                                                 r_hi[AMREX_SPACEDIM*i+2]};
                rv_hi.push_back(rv);
            }

            // Construct vector of bounding real boxes
            m_bnd_rbx.resize(n_plane_lo);
            for (int i(0); i < n_plane_hi; i++){
                amrex::RealBox rbx(rv_lo[i].data(),rv_hi[i].data());
                m_bnd_rbx[i] = rbx;
            }

            // Parse directionality
            m_dir.resize(n_plane_dir);
            pp.queryarr("sample_plane_dir",m_dir,0,n_plane_dir);

            // Parse names
            std::string name_base = "plt_plane_";
            m_name.resize(n_plane_lo);
            int n_names = pp.countval("sample_plane_name");
            if (n_names > 0) {
                AMREX_ALWAYS_ASSERT( n_names==n_plane_lo );
                pp.queryarr("sample_plane_name",m_name,0,n_names);
            } else {
                for (int iplane(0); iplane<n_plane_lo; ++iplane) {
                    m_name[iplane] = amrex::Concatenate(name_base, iplane , 5);
                }
            }

            // Allocate space for level indicator
            m_lev.resize(n_plane_dir,0);

            // Allocate space for MF pointers
            m_ps_mf.resize(n_plane_lo);
        }
    }

    // This must match what is in AMReX_MultiFabUtil.H
    amrex::Box
    getIndexBox (const amrex::RealBox& real_box,
                 const amrex::Geometry& geom) {
        amrex::IntVect slice_lo, slice_hi;

        AMREX_D_TERM(slice_lo[0]=static_cast<int>(std::floor((real_box.lo(0) - geom.ProbLo(0))/geom.CellSize(0)));,
                     slice_lo[1]=static_cast<int>(std::floor((real_box.lo(1) - geom.ProbLo(1))/geom.CellSize(1)));,
                     slice_lo[2]=static_cast<int>(std::floor((real_box.lo(2) - geom.ProbLo(2))/geom.CellSize(2))););

        AMREX_D_TERM(slice_hi[0]=static_cast<int>(std::floor((real_box.hi(0) - geom.ProbLo(0))/geom.CellSize(0)));,
                     slice_hi[1]=static_cast<int>(std::floor((real_box.hi(1) - geom.ProbLo(1))/geom.CellSize(1)));,
                     slice_hi[2]=static_cast<int>(std::floor((real_box.hi(2) - geom.ProbLo(2))/geom.CellSize(2))););

        return amrex::Box(slice_lo, slice_hi) & geom.Domain();
    }

    void
    get_sample_data (amrex::Vector<amrex::Geometry>& geom,
                     amrex::Vector<amrex::Vector<amrex::MultiFab>>& vars_new)
    {
        int nlev   = static_cast<int>(vars_new.size());
        int nplane = static_cast<int>(m_bnd_rbx.size());
        int ncomp  = 2;
        bool interpolate = true;

        // Loop over each plane
        for (int iplane(0); iplane<nplane; ++iplane) {
            int dir = m_dir[iplane];
            amrex::RealBox bnd_rbx  = m_bnd_rbx[iplane];
            amrex::Real point = bnd_rbx.lo(dir);

            // Search each level to get the finest data possible
            for (int ilev(nlev-1); ilev>=0; --ilev) {

                // Construct CC velocities
                amrex::MultiFab mf_cc_vel;
                auto ba = vars_new[ilev][Vars::cons].boxArray();
                auto dm = vars_new[ilev][Vars::cons].DistributionMap();
                mf_cc_vel.define(ba, dm, AMREX_SPACEDIM, amrex::IntVect(1,1,1));
                average_face_to_cellcenter(mf_cc_vel,0,
                                           amrex::Array<const amrex::MultiFab*,3>{&vars_new[ilev][Vars::xvel],
                                                                                  &vars_new[ilev][Vars::yvel],
                                                                                  &vars_new[ilev][Vars::zvel]});

                // Construct vector of MFs holding T and WSP
                amrex::MultiFab mf_cc_data;
                mf_cc_data.define(ba, dm, ncomp, 1);
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(mf_cc_data, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    const amrex::Box& tbx = mfi.tilebox();
                    auto const& dfab = mf_cc_data.array(mfi);
                    auto const& tfab = vars_new[ilev][Vars::cons].array(mfi);
                    auto const& wfab = mf_cc_vel.array(mfi);
                    amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        dfab(i,j,k,0) = tfab(i,j,k,1)/tfab(i,j,k,0);
                        dfab(i,j,k,1) = std::sqrt(wfab(i,j,k,0)*wfab(i,j,k,0)
                                                + wfab(i,j,k,1)*wfab(i,j,k,1)
                                                + wfab(i,j,k,2)*wfab(i,j,k,2)) ;
                    });

                }

                  m_lev[iplane] = ilev;
                m_ps_mf[iplane] = get_slice_data(dir, point, mf_cc_data, geom[ilev],
                                                 0, ncomp, interpolate, bnd_rbx);

                // We can stop if we got the entire plane
                auto min_bnd_bx   = m_ps_mf[iplane]->boxArray().minimalBox();
                amrex::Box bnd_bx = getIndexBox(bnd_rbx, geom[ilev]);
                if (bnd_bx == min_bnd_bx) { break; }

            } // ilev
        }// iplane
    }

    void
    write_sample_data (amrex::Vector<amrex::Real>& time,
                       amrex::Vector<int>& level_steps,
                       amrex::Vector<amrex::IntVect>& ref_ratio,
                       amrex::Vector<amrex::Geometry>& geom)
    {
        amrex::Vector<std::string> varnames = {"T", "Wsp"};

        int nplane = m_ps_mf.size();
        for (int iplane(0); iplane<nplane; ++iplane) {
            // Data members that can be used as-is
            int dir = m_dir[iplane];
            int lev = m_lev[iplane];
            amrex::Real m_time = time[lev];
            amrex::Vector<int> m_level_steps = {level_steps[lev]};
            amrex::Vector<amrex::IntVect> m_ref_ratio  = {ref_ratio[lev]};

            // Create modified geometry object corresponding to the plane
            amrex::RealBox m_rb = m_bnd_rbx[iplane];
            amrex::Box m_dom    = getIndexBox(m_rb, geom[lev]);
            amrex::Real point   = m_rb.hi(dir);
            amrex::Vector<int> is_per(AMREX_SPACEDIM,0);
            for (int d(0); d<AMREX_SPACEDIM; ++d) {
                if (d==dir) {
                    m_rb.setLo(d,point-0.5*geom[lev].CellSize(d));
                    m_rb.setHi(d,point+0.5*geom[lev].CellSize(d));
                }
                is_per[d] = geom[lev].isPeriodic(d);
            }
            amrex::Vector<amrex::Geometry> m_geom; m_geom.resize(1);
            m_geom[0].define(m_dom, &m_rb, geom[lev].Coord(), is_per.data());

            // Create plotfile name
            std::string name_plane = m_name[iplane];
            name_plane += "_step_";
            std::string plotfilename = amrex::Concatenate(name_plane, m_level_steps[0], 5);

            // Get the data
            amrex::Vector<const amrex::MultiFab*> mf = {m_ps_mf[iplane].get()};

            // Write each plane
            WriteMultiLevelPlotfile(plotfilename, 1, mf,
                                    varnames, m_geom, m_time,
                                    m_level_steps, m_ref_ratio);
        } // iplane
    }

    amrex::Vector<int> m_dir;
    amrex::Vector<int> m_lev;
    amrex::Vector<amrex::RealBox> m_bnd_rbx;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_ps_mf;
    amrex::Vector<std::string> m_name;
};
#endif
