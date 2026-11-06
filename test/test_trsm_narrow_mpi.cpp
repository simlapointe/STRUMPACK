/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals
 * from the U.S. Dept. of Energy).  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this
 * software, please contact Berkeley Lab's Technology Transfer
 * Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As
 * such, the U.S. Government has been granted for itself and others
 * acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative
 * works, and perform publicly and display publicly.  Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */

// Compares the distributed triangular solve for narrow right-hand
// sides (dense/DistTrsmNarrow.hpp, used in the solve phase of
// FrontDenseMPI) with PxTRSM, on the process grids of fronts with 1 to
// P processes, for lower and upper, unit and non-unit triangular
// matrices, several block sizes, matrix sizes and numbers of
// right-hand sides, in real and complex double precision.
#include <cmath>
#include <complex>
#include <iostream>
using namespace std;

#include "dense/DistributedMatrix.hpp"
#include "dense/DistTrsmNarrow.hpp"
using namespace strumpack;

#define SOLVE_TOLERANCE 1e-12

template<typename scalar_t> int run(const MPIComm& world) {
  const int P = world.size();
  long long cases = 0;
  int nfail = 0;
  for (int p=1; p<=P; p++) {
    // grid on the first p ranks, as for a front with p processes
    BLACSGrid g(world.sub(0, p), p);
    if (!g.active()) continue;
    trsmnarrow::Comms comms(g.Comm_active().comm(), g.nprows(),
                            g.npcols(), g.prow(), g.pcol());
    for (int nb : {4, 8, DistributedMatrix<scalar_t>::default_MB})
      for (int m : {1, 3, nb-1, nb, nb+1, 2*nb+3, 5*nb, 9*nb+7})
        for (int n : {1, 2, nb/2+1, nb}) {
          if (m < 1 || n > nb ||
              !trsmnarrow::applicable(m, n, nb, g.nprows(), g.npcols()))
            continue;
          DistributedMatrix<scalar_t> A(&g, m, m, nb, nb), B(&g, m, n, nb, nb);
          A.random();
          B.random();
          // diagonally dominant, so the solves are well conditioned
          for (int c=0; c<A.lcols(); c++)
            for (int r=0; r<A.lrows(); r++)
              if (A.rowl2g(r) == A.coll2g(c)) A(r, c) += scalar_t(2. * m);
          for (bool upper : {false, true})
            for (bool unit : {false, true}) {
              auto X = B;
              trsm(Side::L, upper ? UpLo::U : UpLo::L, Trans::N,
                   unit ? Diag::U : Diag::N, scalar_t(1.), A, X);
              auto Y = B;
              comms.solve(upper, unit, m, n, nb, A.data(), A.desc()[8],
                          Y.data(), Y.desc()[8]);
              double err[2] = {0., 0.};
              for (int c=0; c<X.lcols(); c++)
                for (int r=0; r<X.lrows(); r++) {
                  err[0] = std::max
                    (err[0], double(std::abs(Y(r, c) - X(r, c))));
                  err[1] = std::max(err[1], double(std::abs(X(r, c))));
                }
              g.Comm_active().all_reduce(err, 2, MPI_MAX);
              cases++;
              if (err[0] > SOLVE_TOLERANCE * err[1]) {
                nfail++;
                if (g.Comm_active().is_root())
                  cout << "# FAILED: grid " << g.nprows() << "x" << g.npcols()
                       << " nb " << nb << " m " << m << " n " << n
                       << (upper ? " upper" : " lower")
                       << (unit ? " unit" : " non-unit")
                       << ": max |X - X_pxtrsm| / max |X_pxtrsm| = "
                       << err[0] / err[1] << endl;
              }
            }
        }
  }
  // rank 0 is in all the grids, and counts all the cases
  nfail = world.all_reduce(nfail, MPI_MAX);
  if (world.is_root())
    cout << "# narrow triangular solves compared with PxTRSM: " << cases
         << " cases, " << nfail << " failed" << endl;
  return nfail ? 1 : 0;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int ierr = 0;
  {
    MPIComm world;
    ierr += run<double>(world);
    ierr += run<std::complex<double>>(world);
  }
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return ierr ? 1 : 0;
}
