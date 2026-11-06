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

// Distributed triangular solve A X = B for a narrow right-hand side, with
// A an m x m block-cyclic matrix (nb x nb blocks, source process (0,0))
// and B an m x n block-cyclic matrix with the same row blocking and
// n <= nb, so B lives in process column 0.
//
// For each block row k, the processes of the process row of the diagonal
// block sum their partial updates sum_j A(k,j) X(j) with B(k) onto the
// owner of the diagonal block (one MPI_Reduce over the process row), the
// owner solves with the diagonal block, and broadcasts X(k) to its
// process column (one MPI_Bcast), whose processes then update their
// partial sums for the remaining block rows. On grids with several
// process columns, the update of the next block row is done before the
// others, so that the process can take part in the next reduction
// sooner. The solution is sent to process column 0. This replaces the
// many small BLACS messages of PxTRSM by two collectives per block row.
// The result agrees with PxTRSM up to the summation order.
#ifndef STRUMPACK_DIST_TRSM_NARROW_HPP
#define STRUMPACK_DIST_TRSM_NARROW_HPP

#include <algorithm>
#include <vector>

#include "misc/MPIWrapper.hpp"
#include "BLASLAPACKWrapper.hpp"

namespace strumpack {
  namespace trsmnarrow {

    /**
     * Row and column communicators of a process grid, created once
     * per grid. comm has the ranks of the grid in column-major order.
     */
    class Comms {
    public:
      Comms(MPI_Comm comm, int nprow, int npcol, int myrow, int mycol)
        : comm_(comm), nprow_(nprow), npcol_(npcol),
          myrow_(myrow), mycol_(mycol) {
        MPI_Comm_split(comm, myrow, mycol, &row_);
        MPI_Comm_split(comm, mycol, myrow, &col_);
      }
      Comms(const Comms&) = delete;
      Comms& operator=(const Comms&) = delete;
      ~Comms() {
        if (row_ != MPI_COMM_NULL) MPI_Comm_free(&row_);
        if (col_ != MPI_COMM_NULL) MPI_Comm_free(&col_);
      }

      /**
       * Solve op(A) X = B in place in B, with op(A) = A, A lower or
       * upper triangular, with unit or non-unit diagonal. A has local
       * leading dimension lda, B has ldb. Collective on comm.
       */
      template<typename scalar_t> void
      solve(bool upper, bool unit, int m, int n, int nb,
            const scalar_t* A, int lda, scalar_t* B, int ldb) const;

    private:
      MPI_Comm comm_, row_ = MPI_COMM_NULL, col_ = MPI_COMM_NULL;
      int nprow_, npcol_, myrow_, mycol_;
    };

    // whether solve is used instead of PxTRSM
    inline bool applicable(int m, int n, int nb, int nprow, int npcol) {
      return m > 0 && n > 0 && n <= nb && nprow * npcol > 1;
    }

    template<typename scalar_t> void
    Comms::solve(bool upper, bool unit, int m, int n, int nb,
                 const scalar_t* A, int lda, scalar_t* B, int ldb) const {
      const int tag = 30218;
      const int nblk = (m + nb - 1) / nb;
      // local rows of the block rows < k, all of size nb
      auto nloc = [&](int k) {
        return k > myrow_ ? (k - myrow_ + nprow_ - 1) / nprow_ * nb : 0;
      };
      auto bsize = [&](int k) { return std::min(nb, m - k * nb); };
      // number of local rows (the last block row can be short)
      const int lm = nloc(nblk) -
        ((nblk - 1) % nprow_ == myrow_ ? nb - bsize(nblk - 1) : 0);
      const int ldw = std::max(1, lm);
      // partial sums of the updates of the local rows, solutions of
      // the diagonal blocks owned here, received solutions (column 0)
      std::vector<scalar_t> W(std::size_t(ldw) * n, scalar_t(0.)),
        X(std::size_t(ldw) * n), R(mycol_ == 0 ? std::size_t(ldw) * n : 0);
      std::vector<scalar_t> t(std::size_t(nb) * n), x(std::size_t(nb) * n),
        px(std::size_t(nb) * n);
      std::vector<MPI_Request> reqs;
      std::vector<int> rblocks;
      // update of rows [r0, r1) of W with block column k and solution
      // xk, deferred for all rows but the next block row
      int pk = -1, pr0 = 0, pr1 = 0;
      auto update = [&](int k, const scalar_t* xk, int r0, int r1) {
        if (r1 <= r0) return;
        const int kb = bsize(k), lc = (k / npcol_) * nb;
        blas::gemm('N', 'N', r1 - r0, n, kb, scalar_t(1.),
                   A + r0 + std::size_t(lc) * lda, lda, xk, kb,
                   scalar_t(1.), W.data() + r0, ldw);
      };
      for (int s=0; s<nblk; s++) {
        const int k = upper ? nblk - 1 - s : s;
        const int r = k % nprow_, c = k % npcol_, kb = bsize(k);
        const int lr = (k / nprow_) * nb, lc = (k / npcol_) * nb;
        if (myrow_ == r) {
          // B(k) - sum_j A(k,j) X(j), summed over the process row
          for (int j=0; j<n; j++)
            for (int i=0; i<kb; i++)
              t[i + j*kb] = (mycol_ == 0 ? B[lr + i + std::size_t(j)*ldb] :
                             scalar_t(0.)) - W[lr + i + std::size_t(j)*ldw];
          if (npcol_ > 1)
            MPI_Reduce(mycol_ == c ? MPI_IN_PLACE : t.data(), t.data(),
                       kb*n, mpi_type<scalar_t>(), MPI_SUM, c, row_);
          if (mycol_ == c) {
            blas::trsm('L', upper ? 'U' : 'L', 'N', unit ? 'U' : 'N', kb, n,
                       scalar_t(1.), A + lr + std::size_t(lc) * lda, lda,
                       t.data(), kb);
            std::copy(t.begin(), t.begin() + kb*n, x.begin());
          }
        }
        if (mycol_ == c) {
          if (nprow_ > 1)
            MPI_Bcast(x.data(), kb*n, mpi_type<scalar_t>(), r, col_);
          // the remaining updates of the previous block column
          if (pk >= 0) { update(pk, px.data(), pr0, pr1); pk = -1; }
          // rows of the block rows after k (before k for upper)
          int r0 = upper ? 0 : nloc(k+1), r1 = upper ? nloc(k) : lm;
          r0 = std::min(r0, lm); r1 = std::min(r1, lm);
          const int kn = upper ? k - 1 : k + 1;
          if (npcol_ > 1 && kn >= 0 && kn < nblk &&
              kn % nprow_ == myrow_ && r1 > r0) {
            // update the next block row now, and the others after its
            // reduction, to which this process contributes (with a
            // single process column, there is no reduction to wait
            // for, and the whole update is done at once)
            const int nr0 = (kn / nprow_) * nb, nr1 = nr0 + bsize(kn);
            update(k, x.data(), nr0, nr1);
            std::copy(x.begin(), x.begin() + kb*n, px.begin());
            pk = k;
            pr0 = upper ? r0 : nr1;
            pr1 = upper ? nr0 : r1;
          } else update(k, x.data(), r0, r1);
        } else if (pk >= 0 && myrow_ == r) {
          update(pk, px.data(), pr0, pr1);
          pk = -1;
        }
        // the solution goes to process column 0
        if (myrow_ == r && mycol_ == c) {
          if (c == 0) {
            for (int j=0; j<n; j++)
              std::copy(t.data() + j*kb, t.data() + (j+1)*kb,
                        B + lr + std::size_t(j)*ldb);
          } else {
            std::copy(t.begin(), t.begin() + kb*n,
                      X.begin() + std::size_t(lr) * n);
            reqs.emplace_back();
            MPI_Isend(X.data() + std::size_t(lr) * n, kb*n,
                      mpi_type<scalar_t>(), r, tag, comm_, &reqs.back());
          }
        } else if (myrow_ == r && mycol_ == 0) {
          reqs.emplace_back();
          MPI_Irecv(R.data() + std::size_t(lr) * n, kb*n,
                    mpi_type<scalar_t>(), r + c * nprow_, tag, comm_,
                    &reqs.back());
          rblocks.push_back(k);
        }
      }
      if (pk >= 0) update(pk, px.data(), pr0, pr1);
      MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
      for (auto k : rblocks) {
        const int kb = bsize(k), lr = (k / nprow_) * nb;
        for (int j=0; j<n; j++)
          std::copy(R.data() + std::size_t(lr) * n + j*kb,
                    R.data() + std::size_t(lr) * n + (j+1)*kb,
                    B + lr + std::size_t(j)*ldb);
      }
    }

  } // end namespace trsmnarrow
} // end namespace strumpack

#endif // STRUMPACK_DIST_TRSM_NARROW_HPP
