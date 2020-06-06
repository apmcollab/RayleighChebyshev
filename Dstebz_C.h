#include <cmath>
#include <cfloat>
#include <iostream>
#include <cstdlib>
    
#ifndef DSTEBZ_RC_
#define DSTEBZ_RC_
//
// Dstebz_C is a class that encapsulates the f2c translation of the LAPACK routine dstebz.f
// and it's dependent routines.
//
// Documentation for the calling parameters and return arguments for the member function
// dstebz(...) are identical to those for the Fortran routine with data types mapped
// appropriately.
//
// The dstebz(...) member function of this class differs from the dstebz routine 
// in the following two ways:
//
// (a) The call is dstebz(..) and not dstebz_(...)
// (b) The first two arguments are passed as char range and char order instead of 
//     char* range and char* order. 
// 
//
// If EXTERNAL_LAPACK is defined then the routine dstebz_(...) supplied by a lapack library
// is used to evaluate the dstebz member function of this class. The external library
// must be one associated with a 64 bit operating system where Fortran integers are
// mapped to C++ long.
//
// After translation all static declarations were removed as well any reference to defines
// within f2c.h so that there are no external dependencies.
//
// The f2c data types and data values were assigned as follows
//
// doublereal = double
// integer    = long
// ftnlen     = int
// logical    = bool
// FALSE_     = false
// TRUE_      = true
//
// Typically instances of this class are wrapped by another class whose purpose is to supply
// work arrays and parameters when dstebz is to be used for a particular purpose.
//
// Chris Anderson 5/9/2015 
/*
#############################################################################
#
# Modification and encapsulation of f2c translation of the lapack routine 
# dstebz.f, and it's dependencies.
#
# "http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dstebz.f"
#
# Copyright 2015 Chris Anderson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Lesser GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# For a copy of the GNU General Public License see
# <http://www.gnu.org/licenses/>.
#
#############################################################################
*/

#ifdef EXTERNAL_LAPACK
extern "C" int dstebz_(char* range, char* order, long *n, double *vl, double *vu, long *il, long *iu,
double *abstol, double *d__, double *e, long *m, long *nsplit, double *w, long *iblock,
long *isplit, double *work, long *iwork, long *info);
#endif

class Dstebz_C
{

public :

Dstebz_C()
{
    c__1 =  1;
    c_n1 = -1;
    c__3 =  3;
    c__2 =  2;
    c__0 =  0;
}


/* Table of constant values used by translation */

long c__1;
long c_n1;
long c__3;
long c__2;
long c__0;

//

int dstebz(char range, char order, long *n, double *vl, double *vu, long *il, long *iu,
double *abstol, double *d__, double *e, long *m, long *nsplit, double *w, long *iblock,
long *isplit, double *work, long *iwork, long *info)

/* *********************************************************************** */
/* > \brief \b DSTEBZ */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DSTEBZ + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dstebz.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dstebz.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dstebz.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DSTEBZ( RANGE, ORDER, N, VL, VU, IL, IU, ABSTOL, D, E, */
/*                          M, NSPLIT, W, IBLOCK, ISPLIT, WORK, IWORK, */
/*                          INFO ) */

/*       .. Scalar Arguments .. */
/*       CHARACTER          ORDER, RANGE */
/*       long            IL, INFO, IU, M, N, NSPLIT */
/*       DOUBLE PRECISION   ABSTOL, VL, VU */
/*       .. */
/*       .. Array Arguments .. */
/*       long            IBLOCK( * ), ISPLIT( * ), IWORK( * ) */
/*       DOUBLE PRECISION   D( * ), E( * ), W( * ), WORK( * ) */
/*       .. */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DSTEBZ computes the eigenvalues of a symmetric tridiagonal */
/* > matrix T.  The user may ask for all eigenvalues, all eigenvalues */
/* > in the half-open interval (VL, VU], or the IL-th through IU-th */
/* > eigenvalues. */
/* > */
/* > To avoid overflow, the matrix must be scaled so that its */
/* > largest element is no greater than overflow**(1/2) * underflow**(1/4) in absolute value, and for greatest
 */
/* > accuracy, it should not be much smaller than that. */
/* > */
/* > See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal */
/* > Matrix", Report CS41, Computer Science Dept., Stanford */
/* > University, July 21, 1966. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] RANGE */
/* > \verbatim */
/* >          RANGE is CHARACTER*1 */
/* >          = 'A': ("All")   all eigenvalues will be found. */
/* >          = 'V': ("Value") all eigenvalues in the half-open interval */
/* >                           (VL, VU] will be found. */
/* >          = 'I': ("Index") the IL-th through IU-th eigenvalues (of the */
/* >                           entire matrix) will be found. */
/* > \endverbatim */
/* > */
/* > \param[in] ORDER */
/* > \verbatim */
/* >          ORDER is CHARACTER*1 */
/* >          = 'B': ("By Block") the eigenvalues will be grouped by */
/* >                              split-off block (see IBLOCK, ISPLIT) and */
/* >                              ordered from smallest to largest within */
/* >                              the block. */
/* >          = 'E': ("Entire matrix") */
/* >                              the eigenvalues for the entire matrix */
/* >                              will be ordered from smallest to */
/* >                              largest. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is long */
/* >          The order of the tridiagonal matrix T.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] VL */
/* > \verbatim */
/* >          VL is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[in] VU */
/* > \verbatim */
/* >          VU is DOUBLE PRECISION */
/* > */
/* >          If RANGE='V', the lower and upper bounds of the interval to */
/* >          be searched for eigenvalues.  Eigenvalues less than or equal */
/* >          to VL, or greater than VU, will not be returned.  VL < VU. */
/* >          Not referenced if RANGE = 'A' or 'I'. */
/* > \endverbatim */
/* > */
/* > \param[in] IL */
/* > \verbatim */
/* >          IL is long */
/* > \endverbatim */
/* > */
/* > \param[in] IU */
/* > \verbatim */
/* >          IU is long */
/* > */
/* >          If RANGE='I', the indices (in ascending order) of the */
/* >          smallest and largest eigenvalues to be returned. */
/* >          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0. */
/* >          Not referenced if RANGE = 'A' or 'V'. */
/* > \endverbatim */
/* > */
/* > \param[in] ABSTOL */
/* > \verbatim */
/* >          ABSTOL is DOUBLE PRECISION */
/* >          The absolute tolerance for the eigenvalues.  An eigenvalue */
/* >          (or cluster) is considered to be located if it has been */
/* >          determined to lie in an interval whose width is ABSTOL or */
/* >          less.  If ABSTOL is less than or equal to zero, then ULP*|T| */
/* >          will be used, where |T| means the 1-norm of T. */
/* > */
/* >          Eigenvalues will be computed most accurately when ABSTOL is */
/* >          set to twice the underflow threshold 2*DLAMCH('S'), not zero. */
/* > \endverbatim */
/* > */
/* > \param[in] D */
/* > \verbatim */
/* >          D is DOUBLE PRECISION array, dimension (N) */
/* >          The n diagonal elements of the tridiagonal matrix T. */
/* > \endverbatim */
/* > */
/* > \param[in] E */
/* > \verbatim */
/* >          E is DOUBLE PRECISION array, dimension (N-1) */
/* >          The (n-1) off-diagonal elements of the tridiagonal matrix T. */
/* > \endverbatim */
/* > */
/* > \param[out] M */
/* > \verbatim */
/* >          M is long */
/* >          The actual number of eigenvalues found. 0 <= M <= N. */
/* >          (See also the description of INFO=2,3.) */
/* > \endverbatim */
/* > */
/* > \param[out] NSPLIT */
/* > \verbatim */
/* >          NSPLIT is long */
/* >          The number of diagonal blocks in the matrix T. */
/* >          1 <= NSPLIT <= N. */
/* > \endverbatim */
/* > */
/* > \param[out] W */
/* > \verbatim */
/* >          W is DOUBLE PRECISION array, dimension (N) */
/* >          On exit, the first M elements of W will contain the */
/* >          eigenvalues.  (DSTEBZ may use the remaining N-M elements as */
/* >          workspace.) */
/* > \endverbatim */
/* > */
/* > \param[out] IBLOCK */
/* > \verbatim */
/* >          IBLOCK is long array, dimension (N) */
/* >          At each row/column j where E(j) is zero or small, the */
/* >          matrix T is considered to split into a block diagonal */
/* >          matrix.  On exit, if INFO = 0, IBLOCK(i) specifies to which */
/* >          block (from 1 to the number of blocks) the eigenvalue W(i) */
/* >          belongs.  (DSTEBZ may use the remaining N-M elements as */
/* >          workspace.) */
/* > \endverbatim */
/* > */
/* > \param[out] ISPLIT */
/* > \verbatim */
/* >          ISPLIT is long array, dimension (N) */
/* >          The splitting points, at which T breaks up into submatrices. */
/* >          The first submatrix consists of rows/columns 1 to ISPLIT(1), */
/* >          the second of rows/columns ISPLIT(1)+1 through ISPLIT(2), */
/* >          etc., and the NSPLIT-th consists of rows/columns */
/* >          ISPLIT(NSPLIT-1)+1 through ISPLIT(NSPLIT)=N. */
/* >          (Only the first NSPLIT elements will actually be used, but */
/* >          since the user cannot know a priori what value NSPLIT will */
/* >          have, N words must be reserved for ISPLIT.) */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (4*N) */
/* > \endverbatim */
/* > */
/* > \param[out] IWORK */
/* > \verbatim */
/* >          IWORK is long array, dimension (3*N) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is long */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  some or all of the eigenvalues failed to converge or */
/* >                were not computed: */
/* >                =1 or 3: Bisection failed to converge for some */
/* >                        eigenvalues; these eigenvalues are flagged by a */
/* >                        negative block number.  The effect is that the */
/* >                        eigenvalues may not be as accurate as the */
/* >                        absolute and relative tolerances.  This is */
/* >                        generally caused by unexpectedly inaccurate */
/* >                        arithmetic. */
/* >                =2 or 3: RANGE='I' only: Not all of the eigenvalues */
/* >                        IL:IU were found. */
/* >                        Effect: M < IU+1-IL */
/* >                        Cause:  non-monotonic arithmetic, causing the */
/* >                                Sturm sequence to be non-monotonic. */
/* >                        Cure:   recalculate, using RANGE='A', and pick */
/* >                                out eigenvalues IL:IU.  In some cases, */
/* >                                increasing the PARAMETER "FUDGE" may */
/* >                                make things work. */
/* >                = 4:    RANGE='I', and the Gershgorin interval */
/* >                        initially used was too small.  No eigenvalues */
/* >                        were computed. */
/* >                        Probable cause: your machine has sloppy */
/* >                                        floating-point arithmetic. */
/* >                        Cure: Increase the PARAMETER "FUDGE", */
/* >                              recompile, and try again. */
/* > \endverbatim */

/* > \par Internal Parameters: */
/*  ========================= */
/* > */
/* > \verbatim */
/* >  RELFAC  DOUBLE PRECISION, default = 2.0e0 */
/* >          The relative tolerance.  An interval (a,b] lies within */
/* >          "relative tolerance" if  b-a < RELFAC*ulp*dstebz_max(|a|,|b|), */
/* >          where "ulp" is the machine precision (distance from 1 to */
/* >          the next larger floating point number.) */
/* > */
/* >  FUDGE   DOUBLE PRECISION, default = 2 */
/* >          A "fudge factor" to widen the Gershgorin intervals.  Ideally, */
/* >          a value of 1 should work, but on machines with sloppy */
/* >          arithmetic, this needs to be larger.  The default for */
/* >          publicly released versions should be large enough to handle */
/* >          the worst machine around.  Note that this has no effect */
/* >          on accuracy of the solution. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date November 2011 */

/* > \ingroup auxOTHERcomputational */

/*  ===================================================================== */
{
    // Conversion of range and order input character arguments to char*
    // CRA modification of f2c translation : 5/9/2015

    char *rangeArg = &range;
    char *orderArg = &order;

//
//  If using an external lapack implementation, then just call it directly
//
//  The integer data types are being specified as long, thus if this code is
//  used on a 32 bit OS, there may be problems and the argument type may not
//  be appropriate.
//
//
#ifdef EXTERNAL_LAPACK
return dstebz_(rangeArg,orderArg,n,vl, vu,il, iu,abstol, d__,e,m,nsplit,w, iblock, isplit, work, iwork, info);
#endif

    /* System generated locals */
    long i__1, i__2, i__3;
    double d__1, d__2, d__3, d__4, d__5;

    /* Builtin functions */
    //double sqrt(double), log(double);

    /* Local variables */
    long j, ib, jb, ie, je, nb;
    double gl;
    long im, in;
    double gu;
    long iw;
    double wl, wu;
    long nwl;
    double ulp, wlu, wul;
    long nwu;
    double tmp1, tmp2;
    long iend, ioff, iout, itmp1, jdisc;
    long iinfo;
    double atoli;
    long iwoff;
    double bnorm;
    long itmax;
    double wkill, rtoli, tnorm;
    long ibegin;
    long irange, idiscl;
    double safemn;
    long idumma[1];
    long idiscu, iorder;
    bool ncnvrg;
    double pivmin;
    bool toofew;


/*  -- LAPACK computational routine (version 3.4.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2011 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --iwork;
    --work;
    --isplit;
    --iblock;
    --w;
    --e;
    --d__;

    /* Function Body */
    *info = 0;

/*     Decode RANGE */

    if (lsame_(rangeArg, "A", (int)1, (int)1)) {
    irange = 1;
    } else if (lsame_(rangeArg, "V", (int)1, (int)1)) {
    irange = 2;
    } else if (lsame_(rangeArg, "I", (int)1, (int)1)) {
    irange = 3;
    } else {
    irange = 0;
    }

/*     Decode ORDER */

    if (lsame_(orderArg, "B", (int)1, (int)1)) {
    iorder = 2;
    } else if (lsame_(orderArg, "E", (int)1, (int)1)) {
    iorder = 1;
    } else {
    iorder = 0;
    }

/*     Check for Errors */

    if (irange <= 0) {
    *info = -1;
    } else if (iorder <= 0) {
    *info = -2;
    } else if (*n < 0) {
    *info = -3;
    } else if (irange == 2) {
    if (*vl >= *vu) {
        *info = -5;
    }
    } else if (irange == 3 && (*il < 1 || *il > dstebz_max(1,*n))) {
    *info = -6;
    } else if (irange == 3 && (*iu < dstebz_min(*n,*il) || *iu > *n)) {
    *info = -7;
    }

    if (*info != 0) {
    i__1 = -(*info);
    xerbla_("DSTEBZ", &i__1, (int)6);
    return 0;
    }

/*     Initialize error flags */

    *info = 0;
    ncnvrg = false;
    toofew = false;

/*     Quick return if possible */

    *m = 0;
    if (*n == 0) {
    return 0;
    }

/*     Simplifications: */

    if (irange == 3 && *il == 1 && *iu == *n) {
    irange = 1;
    }

/*     Get machine constants */
/*     NB is the minimum vector length for vector bisection, or 0 */
/*     if only scalar is to be done. */

    safemn = dlamch_("S", (int)1);
    ulp = dlamch_("P", (int)1);
    rtoli = ulp * 2.;
    nb = 1;
    //
    // Currently ilaenv always returns 1 for DSTEBZ so removed
    // call to ilaenv to reduce dependencies
    //
    // CRA modification of f2c translation : 5/9/2015
    //
    /*
    nb = ilaenv_(&c__1, "DSTEBZ", " ", n, &c_n1, &c_n1, &c_n1, (int)6, (
        int)1);
    */
    if (nb <= 1) {
    nb = 0;
    }

/*     Special Case when N=1 */

    if (*n == 1) {
    *nsplit = 1;
    isplit[1] = 1;
    if (irange == 2 && (*vl >= d__[1] || *vu < d__[1])) {
        *m = 0;
    } else {
        w[1] = d__[1];
        iblock[1] = 1;
        *m = 1;
    }
    return 0;
    }

/*     Compute Splitting Points */

    *nsplit = 1;
    work[*n] = 0.;
    pivmin = 1.;

    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
/* Computing 2nd power */
    d__1 = e[j - 1];
    tmp1 = d__1 * d__1;
/* Computing 2nd power */
    d__2 = ulp;
    if ((d__1 = d__[j] * d__[j - 1], std::abs(d__1)) * (d__2 * d__2) + safemn
        > tmp1) {
        isplit[*nsplit] = j - 1;
        ++(*nsplit);
        work[j - 1] = 0.;
    } else {
        work[j - 1] = tmp1;
        pivmin = dstebz_max(pivmin,tmp1);
    }
/* L10: */
    }
    isplit[*nsplit] = *n;
    pivmin *= safemn;

/*     Compute Interval and ATOLI */

    if (irange == 3) {

/*        RANGE='I': Compute the interval containing eigenvalues */
/*                   IL through IU. */

/*        Compute Gershgorin interval for entire (split) matrix */
/*        and use it as the initial interval */

    gu = d__[1];
    gl = d__[1];
    tmp1 = 0.;

    i__1 = *n - 1;
    for (j = 1; j <= i__1; ++j) {
        tmp2 = sqrt(work[j]);
/* Computing MAX */
        d__1 = gu, d__2 = d__[j] + tmp1 + tmp2;
        gu = dstebz_max(d__1,d__2);
/* Computing MIN */
        d__1 = gl, d__2 = d__[j] - tmp1 - tmp2;
        gl = dstebz_min(d__1,d__2);
        tmp1 = tmp2;
/* L20: */
    }

/* Computing MAX */
    d__1 = gu, d__2 = d__[*n] + tmp1;
    gu = dstebz_max(d__1,d__2);
/* Computing MIN */
    d__1 = gl, d__2 = d__[*n] - tmp1;
    gl = dstebz_min(d__1,d__2);
/* Computing MAX */
    d__1 = std::abs(gl), d__2 = std::abs(gu);
    tnorm = dstebz_max(d__1,d__2);
    gl = gl - tnorm * 2.1 * ulp * *n - pivmin * 4.2000000000000002;
    gu = gu + tnorm * 2.1 * ulp * *n + pivmin * 2.1;

/*        Compute Iteration parameters */

    itmax = (long) ((log(tnorm + pivmin) - log(pivmin)) / log(2.)) + 2;
    if (*abstol <= 0.) {
        atoli = ulp * tnorm;
    } else {
        atoli = *abstol;
    }

    work[*n + 1] = gl;
    work[*n + 2] = gl;
    work[*n + 3] = gu;
    work[*n + 4] = gu;
    work[*n + 5] = gl;
    work[*n + 6] = gu;
    iwork[1] = -1;
    iwork[2] = -1;
    iwork[3] = *n + 1;
    iwork[4] = *n + 1;
    iwork[5] = *il - 1;
    iwork[6] = *iu;

    dlaebz_(&c__3, &itmax, n, &c__2, &c__2, &nb, &atoli, &rtoli, &pivmin, 
        &d__[1], &e[1], &work[1], &iwork[5], &work[*n + 1], &work[*n 
        + 5], &iout, &iwork[1], &w[1], &iblock[1], &iinfo);

    if (iwork[6] == *iu) {
        wl = work[*n + 1];
        wlu = work[*n + 3];
        nwl = iwork[1];
        wu = work[*n + 4];
        wul = work[*n + 2];
        nwu = iwork[4];
    } else {
        wl = work[*n + 2];
        wlu = work[*n + 4];
        nwl = iwork[2];
        wu = work[*n + 3];
        wul = work[*n + 1];
        nwu = iwork[3];
    }

    if (nwl < 0 || nwl >= *n || nwu < 1 || nwu > *n) {
        *info = 4;
        return 0;
    }
    } else {

/*        RANGE='A' or 'V' -- Set ATOLI */

/* Computing MAX */
    d__3 = std::abs(d__[1]) + std::abs(e[1]), d__4 = (d__1 = d__[*n], std::abs(d__1)) + (
        d__2 = e[*n - 1], std::abs(d__2));
    tnorm = dstebz_max(d__3,d__4);

    i__1 = *n - 1;
    for (j = 2; j <= i__1; ++j) {
/* Computing MAX */
        d__4 = tnorm, d__5 = (d__1 = d__[j], std::abs(d__1)) + (d__2 = e[j - 1]
            , std::abs(d__2)) + (d__3 = e[j], std::abs(d__3));
        tnorm = dstebz_max(d__4,d__5);
/* L30: */
    }

    if (*abstol <= 0.) {
        atoli = ulp * tnorm;
    } else {
        atoli = *abstol;
    }

    if (irange == 2) {
        wl = *vl;
        wu = *vu;
    } else {
        wl = 0.;
        wu = 0.;
    }
    }

/*     Find Eigenvalues -- Loop Over Blocks and recompute NWL and NWU. */
/*     NWL accumulates the number of eigenvalues .le. WL, */
/*     NWU accumulates the number of eigenvalues .le. WU */

    *m = 0;
    iend = 0;
    *info = 0;
    nwl = 0;
    nwu = 0;

    i__1 = *nsplit;
    for (jb = 1; jb <= i__1; ++jb) {
    ioff = iend;
    ibegin = ioff + 1;
    iend = isplit[jb];
    in = iend - ioff;

    if (in == 1) {

/*           Special Case -- IN=1 */

        if (irange == 1 || wl >= d__[ibegin] - pivmin) {
        ++nwl;
        }
        if (irange == 1 || wu >= d__[ibegin] - pivmin) {
        ++nwu;
        } // N.B. Added ()'s around && evaluation CRA
        if (irange == 1 || (wl < d__[ibegin] - pivmin && wu >= d__[ibegin] 
            - pivmin)) {
        ++(*m);
        w[*m] = d__[ibegin];
        iblock[*m] = jb;
        }
    } else {

/*           General Case -- IN > 1 */

/*           Compute Gershgorin Interval */
/*           and use it as the initial interval */

        gu = d__[ibegin];
        gl = d__[ibegin];
        tmp1 = 0.;

        i__2 = iend - 1;
        for (j = ibegin; j <= i__2; ++j) {
        tmp2 = (d__1 = e[j], std::abs(d__1));
/* Computing MAX */
        d__1 = gu, d__2 = d__[j] + tmp1 + tmp2;
        gu = dstebz_max(d__1,d__2);
/* Computing MIN */
        d__1 = gl, d__2 = d__[j] - tmp1 - tmp2;
        gl = dstebz_min(d__1,d__2);
        tmp1 = tmp2;
/* L40: */
        }

/* Computing MAX */
        d__1 = gu, d__2 = d__[iend] + tmp1;
        gu = dstebz_max(d__1,d__2);
/* Computing MIN */
        d__1 = gl, d__2 = d__[iend] - tmp1;
        gl = dstebz_min(d__1,d__2);
/* Computing MAX */
        d__1 = std::abs(gl), d__2 = std::abs(gu);
        bnorm = dstebz_max(d__1,d__2);
        gl = gl - bnorm * 2.1 * ulp * in - pivmin * 2.1;
        gu = gu + bnorm * 2.1 * ulp * in + pivmin * 2.1;

/*           Compute ATOLI for the current submatrix */

        if (*abstol <= 0.) {
/* Computing MAX */
        d__1 = std::abs(gl), d__2 = std::abs(gu);
        atoli = ulp * dstebz_max(d__1,d__2);
        } else {
        atoli = *abstol;
        }

        if (irange > 1) {
        if (gu < wl) {
            nwl += in;
            nwu += in;
            goto L70;
        }
        gl = dstebz_max(gl,wl);
        gu = dstebz_min(gu,wu);
        if (gl >= gu) {
            goto L70;
        }
        }

/*           Set Up Initial Interval */

        work[*n + 1] = gl;
        work[*n + in + 1] = gu;
        dlaebz_(&c__1, &c__0, &in, &in, &c__1, &nb, &atoli, &rtoli, &
            pivmin, &d__[ibegin], &e[ibegin], &work[ibegin], idumma, &
            work[*n + 1], &work[*n + (in << 1) + 1], &im, &iwork[1], &
            w[*m + 1], &iblock[*m + 1], &iinfo);

        nwl += iwork[1];
        nwu += iwork[in + 1];
        iwoff = *m - iwork[1];

/*           Compute Eigenvalues */

        itmax = (long) ((log(gu - gl + pivmin) - log(pivmin)) / log(2.)
            ) + 2;
        dlaebz_(&c__2, &itmax, &in, &in, &c__1, &nb, &atoli, &rtoli, &
            pivmin, &d__[ibegin], &e[ibegin], &work[ibegin], idumma, &
            work[*n + 1], &work[*n + (in << 1) + 1], &iout, &iwork[1],
             &w[*m + 1], &iblock[*m + 1], &iinfo);

/*           Copy Eigenvalues Into W and IBLOCK */
/*           Use -JB for block number for unconverged eigenvalues. */

        i__2 = iout;
        for (j = 1; j <= i__2; ++j) {
        tmp1 = (work[j + *n] + work[j + in + *n]) * .5;

/*              Flag non-convergence. */

        if (j > iout - iinfo) {
            ncnvrg = true;
            ib = -jb;
        } else {
            ib = jb;
        }
        i__3 = iwork[j + in] + iwoff;
        for (je = iwork[j] + 1 + iwoff; je <= i__3; ++je) {
            w[je] = tmp1;
            iblock[je] = ib;
/* L50: */
        }
/* L60: */
        }

        *m += im;
    }
L70:
    ;
    }

/*     If RANGE='I', then (WL,WU) contains eigenvalues NWL+1,...,NWU */
/*     If NWL+1 < IL or NWU > IU, discard extra eigenvalues. */

    if (irange == 3) {
    im = 0;
    idiscl = *il - 1 - nwl;
    idiscu = nwu - *iu;

    if (idiscl > 0 || idiscu > 0) {
        i__1 = *m;
        for (je = 1; je <= i__1; ++je) {
        if (w[je] <= wlu && idiscl > 0) {
            --idiscl;
        } else if (w[je] >= wul && idiscu > 0) {
            --idiscu;
        } else {
            ++im;
            w[im] = w[je];
            iblock[im] = iblock[je];
        }
/* L80: */
        }
        *m = im;
    }
    if (idiscl > 0 || idiscu > 0) {

/*           Code to deal with effects of bad arithmetic: */
/*           Some low eigenvalues to be discarded are not in (WL,WLU], */
/*           or high eigenvalues to be discarded are not in (WUL,WU] */
/*           so just kill off the smallest IDISCL/largest IDISCU */
/*           eigenvalues, by simply finding the smallest/largest */
/*           eigenvalue(s). */

/*           (If N(w) is monotone non-decreasing, this should never */
/*               happen.) */

        if (idiscl > 0) {
        wkill = wu;
        i__1 = idiscl;
        for (jdisc = 1; jdisc <= i__1; ++jdisc) {
            iw = 0;
            i__2 = *m;
            for (je = 1; je <= i__2; ++je) {
            if (iblock[je] != 0 && (w[je] < wkill || iw == 0)) {
                iw = je;
                wkill = w[je];
            }
/* L90: */
            }
            iblock[iw] = 0;
/* L100: */
        }
        }
        if (idiscu > 0) {

        wkill = wl;
        i__1 = idiscu;
        for (jdisc = 1; jdisc <= i__1; ++jdisc) {
            iw = 0;
            i__2 = *m;
            for (je = 1; je <= i__2; ++je) {
            if (iblock[je] != 0 && (w[je] > wkill || iw == 0)) {
                iw = je;
                wkill = w[je];
            }
/* L110: */
            }
            iblock[iw] = 0;
/* L120: */
        }
        }
        im = 0;
        i__1 = *m;
        for (je = 1; je <= i__1; ++je) {
        if (iblock[je] != 0) {
            ++im;
            w[im] = w[je];
            iblock[im] = iblock[je];
        }
/* L130: */
        }
        *m = im;
    }
    if (idiscl < 0 || idiscu < 0) {
        toofew = true;
    }
    }

/*     If ORDER='B', do nothing -- the eigenvalues are already sorted */
/*        by block. */
/*     If ORDER='E', sort the eigenvalues from smallest to largest */

    if (iorder == 1 && *nsplit > 1) {
    i__1 = *m - 1;
    for (je = 1; je <= i__1; ++je) {
        ie = 0;
        tmp1 = w[je];
        i__2 = *m;
        for (j = je + 1; j <= i__2; ++j) {
        if (w[j] < tmp1) {
            ie = j;
            tmp1 = w[j];
        }
/* L140: */
        }

        if (ie != 0) {
        itmp1 = iblock[ie];
        w[ie] = w[je];
        iblock[ie] = iblock[je];
        w[je] = tmp1;
        iblock[je] = itmp1;
        }
/* L150: */
    }
    }

    *info = 0;
    if (ncnvrg) {
    ++(*info);
    }
    if (toofew) {
    *info += 2;
    }
    return 0;

/*     End of DSTEBZ */

} /* dstebz_ */

// Transformed c translations of XERBLA, DLAMCH and LSAME

int xerbla_(const char* A, long* I2, int)
{
    cerr << "** On entry to " <<  A << " parameter number " <<  I2 <<  " had an illegal value" << endl;
    cerr << "XXX Program Terminated XXX" << endl;
    exit(1);
    return 1;
}

double dlamch_(const char* CMACH, int len)
{
      double   ONE, ZERO;
      ONE  = 1.0;
      ZERO = 0.0;
      double   RND, EPS, SFMIN, SMALL, RMACH;

//    Assume rounding, not chopping. Always.

      RND = ONE;
      if( ONE == RND )
      {EPS = DBL_EPSILON * 0.5;}
      else
      {EPS = DBL_EPSILON;}

      if((CMACH[0] == 'E')||(CMACH[0] == 'e'))
      {
         RMACH = EPS;
      }
      else if((CMACH[0] == 'S' )||(CMACH[0] == 's' ))
      {
         SFMIN = DBL_MIN;
         SMALL = ONE / DBL_MAX;
         if(SMALL >= SFMIN)
         {
//
//        Use SMALL plus a bit, to avoid the possibility of rounding
//        causing overflow when computing  1/sfmin.
//
            SFMIN = SMALL*( ONE+EPS );
         }
         RMACH = SFMIN;
     }

     else if((CMACH[0] == 'B' )||(CMACH[0] == 'b' ))
     {
         RMACH = FLT_RADIX;
     }

     else if((CMACH[0] == 'P' )||(CMACH[0] == 'p' ))
     {RMACH = EPS * FLT_RADIX;}
     else if((CMACH[0] == 'N' )||(CMACH[0] == 'n' ))
     {RMACH = DBL_MANT_DIG;}
     else if((CMACH[0] == 'R' )||(CMACH[0] == 'r' ))
     {RMACH = RND;}
     else if((CMACH[0] == 'M' )||(CMACH[0] == 'm' ))
     {RMACH = DBL_MIN_EXP;}
     else if((CMACH[0] == 'U' )||(CMACH[0] == 'u' ))
     {RMACH = DBL_MIN;}
     else if((CMACH[0] == 'L' )||(CMACH[0] == 'l' ))
     {RMACH = DBL_MAX_EXP;}
     else if((CMACH[0] == 'O' )||(CMACH[0] == 'o'))
     {RMACH = DBL_MAX;}
     else
     {RMACH = ZERO;}

     return RMACH;
}

bool lsame_(const char *ca, const char *cb, int ca_len, int cb_len)
{
    bool ret_val;
    long inta, intb, zcode;


    ret_val = *(unsigned char *)ca == *(unsigned char *)cb;
    if (ret_val) {
    return ret_val;
    }

    zcode = 'Z';
    inta = *(unsigned char *)ca;
    intb = *(unsigned char *)cb;

    if (zcode == 90 || zcode == 122) {

/*        ASCII is assumed - ZCODE is the ASCII code of either lower or */
/*        upper case 'Z'. */

    // N.B. Added ()'s around && evaluation CRA
    if (inta >= 97 && inta <= 122) {
        inta += -32;
    }
    if (intb >= 97 && intb <= 122) {
        intb += -32;
    }

    } else if (zcode == 233 || zcode == 169) {

    if ((inta >= 129 && inta <= 137) || (inta >= 145 && inta <= 153) || (inta
        >= 162 && inta <= 169)) {
        inta += 64;
    }
    if ((intb >= 129 && intb <= 137) || (intb >= 145 && intb <= 153) || (intb
        >= 162 && intb <= 169)) {
        intb += 64;
    }

    } else if (zcode == 218 || zcode == 250) {

    if (inta >= 225 && inta <= 250) {
        inta += -32;
    }
    if (intb >= 225 && intb <= 250) {
        intb += -32;
    }
    }
    ret_val = inta == intb;

    return ret_val;
}



int dlaebz_(long *ijob, long *nitmax, long *n,
    long *mmax, long *minp, long *nbmin, double *abstol,
    double *reltol, double *pivmin, double *d__, double *
    e, double *e2, long *nval, double *ab, double *c__,
    long *mout, long *nab, double *work, long *iwork,
    long *info)
{
    /* System generated locals */
    long nab_dim1, nab_offset, ab_dim1, ab_offset, i__1, i__2, i__3, i__4,
        i__5, i__6;
    double d__1, d__2, d__3, d__4;

    /* Local variables */
    long j, kf, ji, kl, jp, jit;
    double tmp1, tmp2;
    long itmp1, itmp2, kfnew, klnew;

    /* Parameter adjustments */
    nab_dim1 = *mmax;
    nab_offset = 1 + nab_dim1;
    nab -= nab_offset;
    ab_dim1 = *mmax;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    --d__;
    --e;
    --e2;
    --nval;
    --c__;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;
    if (*ijob < 1 || *ijob > 3) {
    *info = -1;
    return 0;
    }

/*     Initialize NAB */

    if (*ijob == 1) {

/*        Compute the number of eigenvalues in the initial intervals. */

    *mout = 0;
    i__1 = *minp;
    for (ji = 1; ji <= i__1; ++ji) {
        for (jp = 1; jp <= 2; ++jp) {
        tmp1 = d__[1] - ab[ji + jp * ab_dim1];
        if (std::abs(tmp1) < *pivmin) {
            tmp1 = -(*pivmin);
        }
        nab[ji + jp * nab_dim1] = 0;
        if (tmp1 <= 0.) {
            nab[ji + jp * nab_dim1] = 1;
        }

        i__2 = *n;
        for (j = 2; j <= i__2; ++j) {
            tmp1 = d__[j] - e2[j - 1] / tmp1 - ab[ji + jp * ab_dim1];
            if (std::abs(tmp1) < *pivmin) {
            tmp1 = -(*pivmin);
            }
            if (tmp1 <= 0.) {
            ++nab[ji + jp * nab_dim1];
            }
/* L10: */
        }
/* L20: */
        }
        *mout = *mout + nab[ji + (nab_dim1 << 1)] - nab[ji + nab_dim1];
/* L30: */
    }
    return 0;
    }

/*     Initialize for loop */

/*     KF and KL have the following meaning: */
/*        Intervals 1,...,KF-1 have converged. */
/*        Intervals KF,...,KL  still need to be refined. */

    kf = 1;
    kl = *minp;

/*     If IJOB=2, initialize C. */
/*     If IJOB=3, use the user-supplied starting point. */

    if (*ijob == 2) {
    i__1 = *minp;
    for (ji = 1; ji <= i__1; ++ji) {
        c__[ji] = (ab[ji + ab_dim1] + ab[ji + (ab_dim1 << 1)]) * .5;
/* L40: */
    }
    }

/*     Iteration loop */

    i__1 = *nitmax;
    for (jit = 1; jit <= i__1; ++jit) {

/*        Loop over intervals */

    if (kl - kf + 1 >= *nbmin && *nbmin > 0) {

/*           Begin of Parallel Version of the loop */

        i__2 = kl;
        for (ji = kf; ji <= i__2; ++ji) {

/*              Compute N(c), the number of eigenvalues less than c */

        work[ji] = d__[1] - c__[ji];
        iwork[ji] = 0;
        if (work[ji] <= *pivmin) {
            iwork[ji] = 1;
/* Computing MIN */
            d__1 = work[ji], d__2 = -(*pivmin);
            work[ji] = dstebz_min(d__1,d__2);
        }

        i__3 = *n;
        for (j = 2; j <= i__3; ++j) {
            work[ji] = d__[j] - e2[j - 1] / work[ji] - c__[ji];
            if (work[ji] <= *pivmin) {
            ++iwork[ji];
/* Computing MIN */
            d__1 = work[ji], d__2 = -(*pivmin);
            work[ji] = dstebz_min(d__1,d__2);
            }
/* L50: */
        }
/* L60: */
        }

        if (*ijob <= 2) {

/*              IJOB=2: Choose all intervals containing eigenvalues. */

        klnew = kl;
        i__2 = kl;
        for (ji = kf; ji <= i__2; ++ji) {

/*                 Insure that N(w) is monotone */

/* Computing MIN */
/* Computing MAX */
            i__5 = nab[ji + nab_dim1], i__6 = iwork[ji];
            i__3 = nab[ji + (nab_dim1 << 1)], i__4 = dstebz_max(i__5,i__6);
            iwork[ji] = dstebz_min(i__3,i__4);

/*                 Update the Queue -- add intervals if both halves */
/*                 contain eigenvalues. */

            if (iwork[ji] == nab[ji + (nab_dim1 << 1)]) {

/*                    No eigenvalue in the upper interval: */
/*                    just use the lower interval. */

            ab[ji + (ab_dim1 << 1)] = c__[ji];

            } else if (iwork[ji] == nab[ji + nab_dim1]) {

/*                    No eigenvalue in the lower interval: */
/*                    just use the upper interval. */

            ab[ji + ab_dim1] = c__[ji];
            } else {
            ++klnew;
            if (klnew <= *mmax) {

/*                       Eigenvalue in both intervals -- add upper to */
/*                       queue. */

                ab[klnew + (ab_dim1 << 1)] = ab[ji + (ab_dim1 <<
                    1)];
                nab[klnew + (nab_dim1 << 1)] = nab[ji + (nab_dim1
                    << 1)];
                ab[klnew + ab_dim1] = c__[ji];
                nab[klnew + nab_dim1] = iwork[ji];
                ab[ji + (ab_dim1 << 1)] = c__[ji];
                nab[ji + (nab_dim1 << 1)] = iwork[ji];
            } else {
                *info = *mmax + 1;
            }
            }
/* L70: */
        }
        if (*info != 0) {
            return 0;
        }
        kl = klnew;
        } else {

/*              IJOB=3: Binary search.  Keep only the interval containing */
/*                      w   s.t. N(w) = NVAL */

        i__2 = kl;
        for (ji = kf; ji <= i__2; ++ji) {
            if (iwork[ji] <= nval[ji]) {
            ab[ji + ab_dim1] = c__[ji];
            nab[ji + nab_dim1] = iwork[ji];
            }
            if (iwork[ji] >= nval[ji]) {
            ab[ji + (ab_dim1 << 1)] = c__[ji];
            nab[ji + (nab_dim1 << 1)] = iwork[ji];
            }
/* L80: */
        }
        }

    } else {

/*           End of Parallel Version of the loop */

/*           Begin of Serial Version of the loop */

        klnew = kl;
        i__2 = kl;
        for (ji = kf; ji <= i__2; ++ji) {

/*              Compute N(w), the number of eigenvalues less than w */

        tmp1 = c__[ji];
        tmp2 = d__[1] - tmp1;
        itmp1 = 0;
        if (tmp2 <= *pivmin) {
            itmp1 = 1;
/* Computing MIN */
            d__1 = tmp2, d__2 = -(*pivmin);
            tmp2 = dstebz_min(d__1,d__2);
        }

        i__3 = *n;
        for (j = 2; j <= i__3; ++j) {
            tmp2 = d__[j] - e2[j - 1] / tmp2 - tmp1;
            if (tmp2 <= *pivmin) {
            ++itmp1;
/* Computing MIN */
            d__1 = tmp2, d__2 = -(*pivmin);
            tmp2 = dstebz_min(d__1,d__2);
            }
/* L90: */
        }

        if (*ijob <= 2) {

/*                 IJOB=2: Choose all intervals containing eigenvalues. */

/*                 Insure that N(w) is monotone */

/* Computing MIN */
/* Computing MAX */
            i__5 = nab[ji + nab_dim1];
            i__3 = nab[ji + (nab_dim1 << 1)], i__4 = dstebz_max(i__5,itmp1);
            itmp1 = dstebz_min(i__3,i__4);

/*                 Update the Queue -- add intervals if both halves */
/*                 contain eigenvalues. */

            if (itmp1 == nab[ji + (nab_dim1 << 1)]) {

/*                    No eigenvalue in the upper interval: */
/*                    just use the lower interval. */

            ab[ji + (ab_dim1 << 1)] = tmp1;

            } else if (itmp1 == nab[ji + nab_dim1]) {

/*                    No eigenvalue in the lower interval: */
/*                    just use the upper interval. */

            ab[ji + ab_dim1] = tmp1;
            } else if (klnew < *mmax) {

/*                    Eigenvalue in both intervals -- add upper to queue. */

            ++klnew;
            ab[klnew + (ab_dim1 << 1)] = ab[ji + (ab_dim1 << 1)];
            nab[klnew + (nab_dim1 << 1)] = nab[ji + (nab_dim1 <<
                1)];
            ab[klnew + ab_dim1] = tmp1;
            nab[klnew + nab_dim1] = itmp1;
            ab[ji + (ab_dim1 << 1)] = tmp1;
            nab[ji + (nab_dim1 << 1)] = itmp1;
            } else {
            *info = *mmax + 1;
            return 0;
            }
        } else {

/*                 IJOB=3: Binary search.  Keep only the interval */
/*                         containing  w  s.t. N(w) = NVAL */

            if (itmp1 <= nval[ji]) {
            ab[ji + ab_dim1] = tmp1;
            nab[ji + nab_dim1] = itmp1;
            }
            if (itmp1 >= nval[ji]) {
            ab[ji + (ab_dim1 << 1)] = tmp1;
            nab[ji + (nab_dim1 << 1)] = itmp1;
            }
        }
/* L100: */
        }
        kl = klnew;

    }

/*        Check for convergence */

    kfnew = kf;
    i__2 = kl;
    for (ji = kf; ji <= i__2; ++ji) {
        tmp1 = (d__1 = ab[ji + (ab_dim1 << 1)] - ab[ji + ab_dim1], std::abs(d__1));
/* Computing MAX */
        d__3 = (d__1 = ab[ji + (ab_dim1 << 1)], std::abs(d__1)), d__4 = (d__2 =
             ab[ji + ab_dim1], std::abs(d__2));
        tmp2 = dstebz_max(d__3,d__4);
/* Computing MAX */
        d__1 = dstebz_max(*abstol,*pivmin), d__2 = *reltol * tmp2;
        if (tmp1 < dstebz_max(d__1,d__2) || nab[ji + nab_dim1] >= nab[ji + (
            nab_dim1 << 1)]) {

/*              Converged -- Swap with position KFNEW, */
/*                           then increment KFNEW */

        if (ji > kfnew) {
            tmp1 = ab[ji + ab_dim1];
            tmp2 = ab[ji + (ab_dim1 << 1)];
            itmp1 = nab[ji + nab_dim1];
            itmp2 = nab[ji + (nab_dim1 << 1)];
            ab[ji + ab_dim1] = ab[kfnew + ab_dim1];
            ab[ji + (ab_dim1 << 1)] = ab[kfnew + (ab_dim1 << 1)];
            nab[ji + nab_dim1] = nab[kfnew + nab_dim1];
            nab[ji + (nab_dim1 << 1)] = nab[kfnew + (nab_dim1 << 1)];
            ab[kfnew + ab_dim1] = tmp1;
            ab[kfnew + (ab_dim1 << 1)] = tmp2;
            nab[kfnew + nab_dim1] = itmp1;
            nab[kfnew + (nab_dim1 << 1)] = itmp2;
            if (*ijob == 3) {
            itmp1 = nval[ji];
            nval[ji] = nval[kfnew];
            nval[kfnew] = itmp1;
            }
        }
        ++kfnew;
        }
/* L110: */
    }
    kf = kfnew;

/*        Choose Midpoints */

    i__2 = kl;
    for (ji = kf; ji <= i__2; ++ji) {
        c__[ji] = (ab[ji + ab_dim1] + ab[ji + (ab_dim1 << 1)]) * .5;
/* L120: */
    }

/*        If no more intervals to refine, quit. */

    if (kf > kl) {
        goto L140;
    }
/* L130: */
    }

/*     Converged */

L140:
/* Computing MAX */
    i__1 = kl + 1 - kf;
    *info = dstebz_max(i__1,0);
    *mout = kl;

    return 0;

/*     End of DLAEBZ */

} /* dlaebz_ */


double dlamc3_(double *a, double *b)
{
    double ret_val;
    ret_val = *a + *b;
    return ret_val;
} /* dlamc3_ */


double dstebz_max(double a, double b)
{
    if(a > b) return a;
    return b;
}
double dstebz_min(double a, double b)
{
    if(a < b) return a;
    return b;
}
long dstebz_min(long a, long b)
{
    if(a < b) return a;
    return b;
}
long dstebz_max(long a, long b)
{
    if(a > b) return a;
    return b;
}
 

};


#endif


