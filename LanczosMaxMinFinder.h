//
//############################################################################
//                             LanczosMaxMinFinder.h
//############################################################################
//
/*

   ############################################################################
   
   Otype
   ----------

   An opearator class with the following member function:

   void apply(Vtype& V)

   which applies the operator to the argument V and returns the result in V.

   ############################################################################

   VRandomizeOpType
   ----------

   An opearator class with the following member function:

   void randomize(Vtype& V)     

   which initializes the elements of the Vtype std::vector V to have random values.

   ############################################################################

   External dependencies:

   This routine currently uses a call to the LAPACK routine dstebz_ to find
   the eigenvalues of the tridiagonal system created with the Lanczos process.

   To use this code on a machine with a reasonably recent version of
   Ubuntu linux system that has lapack installed, use the compilation
   command g++ program.cpp -llapack-3

   For other Unix systems, one can try linking to -llapack -lg2c

   It is important to use the -lg2c library and not the -lf2c library.



	Author:  Chris Anderson
    Changes:

    Aug. 17, 2006: Removed the dependency on cammva

    March, 7, 2012 : Removed the dependency on std::vector<double>

*/
/*
#############################################################################
#
# Copyright 2006-2015 Chris Anderson
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

#ifndef LANCZOS_MAX_MIN_FINDER_
#define LANCZOS_MAX_MIN_FINDER_

#ifndef  LANCZOS_SMALL_TOL_
#define  LANCZOS_SMALL_TOL_ 1.0e-10
#endif

#ifndef  LANCZOS_MAX_MIN_FINDER_ITERATION_MAX
#define  LANCZOS_MAX_MIN_FINDER_ITERATION_MAX  1000
#endif

#include <iostream>
#include <cmath>
#include <cstdio>
#include "Dstebz_C.h"

/*
extern "C" int dstebz_(char *range, char *order, long *n, double 
*vl, double *vu, long *il, long *iu, double *abstol, 
double *d__, double *e, long *m, long *nsplit, double *w, long *iblock, 
long *isplit, double *work, long *iwork, long *info);
*/



template <class Vtype, class Otype, class VRandomizeOpType> 
class LanczosMaxMinFinder
{
    public :

LanczosMaxMinFinder()
{
	initialize();
}

LanczosMaxMinFinder(long maxIterCount) 
{
    initialize(maxIterCount);
}

~LanczosMaxMinFinder(){}

void initialize()
{
    this->iterationMaximumCount      =   LANCZOS_MAX_MIN_FINDER_ITERATION_MAX;
    this->verboseFlag                = 0;
    this->exactDiagnosticFlag        = 0;
    this->maxExact                   = 0.0;
    this->minExact                   = 0.0;
    this->iterCount                  = 0;

    alpha.resize(this->iterationMaximumCount,0.0);
    beta.resize(this->iterationMaximumCount,0.0);


    maxEigFlag = true;
    minEigFlag = true;
}

void initialize(long maxIterCount)
{
    this->iterationMaximumCount      =  maxIterCount;
    this->verboseFlag                = 0;
    this->exactDiagnosticFlag        = 0;
    this->maxExact                   = 0.0;
    this->minExact                   = 0.0;
    this->iterCount                  = 0;

    alpha.resize(this->iterationMaximumCount,0.0);
    beta.resize(this->iterationMaximumCount,0.0);

    maxEigFlag = true;
    minEigFlag = true;
}

double getRelErrorFactor(double val, double tol)
{
	double relErrFactor = 1.0;

    if(std::abs(val)*tol > LANCZOS_SMALL_TOL_ ){relErrFactor = std::abs(val);}
    else                                    {relErrFactor = LANCZOS_SMALL_TOL_/tol;}
    return relErrFactor;
}

void  setIterationMax(long iterMax)
{
	this->iterationMaximumCount = iterMax;
    alpha.resize(this->iterationMaximumCount,0.0);
    beta.resize(this->iterationMaximumCount,0.0);
};

void setVerboseFlag()
{verboseFlag = 1;}

void unsetVerboseFlag()
{verboseFlag = 0;}

void setMaxEigStopCondition()
{
	maxEigFlag = true;
	minEigFlag = false;
}

void setMinEigStopCondition()
{
	minEigFlag = true;
	maxEigFlag = false;
}

void setMinMaxEigStopCondition()
{
	minEigFlag = true;
	maxEigFlag = true;
}

void setDiagnosticExactValues(double exactMin,double exactMax)
{
    this->maxExact             = exactMax;
    this->minExact             = exactMin;
    exactDiagnosticFlag        = 1;
	setVerboseFlag();
}

//
// This routine uses the Lanczos procedure to obtain estimates of the algebraically
// largest and algebraically smallest eigenvalue.
//
// The default stopping condition is when estimates of both the largest and smallest
// eigenvalues have converged.
//
// If one specifies that the stopping condition be based on the largest
// eigenvalue by invoking setMaxEigStopCondition() then this routine
// returns a converged value for the largest eigenvalue and a
// possibly non-converged estimate of the smallest eigenvalue.
//
// Similarly, if one specifies the stopping condition be based on the smallest
// eigenvalue by invoking setMinEigStopCondition() then this routine
// returns converged value for the smallest eigenvalue and a possibly
// non-converged estimate of the largest eigenvalue.
//
void getMinMaxEigenvalues(double errorTolerance, Vtype& v, Vtype& w, Vtype& wTmp, Otype &oP, 
VRandomizeOpType& randOp, double& minEigValue, double& maxEigValue)
{
	double vNorm;

    eigMaxVector.resize(1);
    eigMinVector.resize(1);

    // impose a minimimal degree of accuracy

    if(errorTolerance > 0.01) errorTolerance = 0.01; 

    double tol       = errorTolerance;
    if(tol < LANCZOS_SMALL_TOL_ ) { tol = LANCZOS_SMALL_TOL_; }
    double relErrFactor;
//
//  Create initial vectors
//
	randOp.randomize(v);
    vNorm   = std::sqrt(v.dot(v));
    v      *= 1.0/vNorm;
    w       = v;
    wTmp    = v;
	
    v      *= 0.0;              // v = 0
    wTmp   *= 0.0;              // w = 0

    long k; long j;

    long nValues  = 1;

    double minDiffError  = 1.0e10; double maxDiffError  = 1.0e10;
    double minError      = 1.0e10; double maxError      = 1.0e10;
    double lanczosMin    = 0.0;    double lanczosMax    = 0.0;
    double rateFactorMax = 9.0;    double rateFactorMin = 9.0;
    double rateEigMax    = 1.0;    double rateEigMin    = 1.0;

    double eigMaxOld = 0.0;        double eigMinOld = 0.0;
    double diffMaxA  = 0.0;        double diffMinA  = 0.0;
    double diffMaxB  = 0.0;        double diffMinB  = 0.0;

    int diffCompareFlag  = 0;      int rateCompareFlag   = 0;

    k = 0;

    if((verboseFlag == 1)&&(exactDiagnosticFlag == 1))
    {
    printf("      Evalue        Exact Error   Est.  Error      Rate         Evalue        Exact Error   Est.  Error      Rate \n");
    }
    else if(verboseFlag == 1)
    {
    printf("      Evalue       Est.  Error    Rate       Evalue       Est.  Error      Rate \n");
    }


    bool exitFlag = false;

    while((k < iterationMaximumCount)&&(not exitFlag))
    {
//
//  Standard Lanczos Procedure
//
        if(k != 0)
        {
            if(std::abs(beta[k-1]) > 1.0e-14)
            {
                wTmp = w;
                w    = v;
                w   *= 1.0/beta[k-1];
                v    = wTmp;
                v   *= -beta[k-1];
            }
        }
        wTmp = w;                       // v = v + A(w)
        oP.apply(wTmp);
        v   += wTmp;

        alpha[k] = w.dot(v);            // alpha(k) = w.dot(v)

        wTmp  = w;                      // v = v - alpha(k)*w
        wTmp *= -alpha[k];
        v    += wTmp;

        beta[k]  = std::sqrt(std::abs(v.dot(v))); // beta(k) = norm_2(v);
        k   += 1;                       // k = k+1
//
//      Every five steps compute extremal eigenvalues 
//      and check for convergence. The stopping criterion is based
//      on the values of the differences between successive approximations
//      and the rate at which these differences are decreasing.
//
//      Specifically if d_(k) = std::abs( eVal_(k) - eVal_(k-1) )
//      and          rate_k   = d_(k)/d_(k-1)
//
//      I use a scaled relative error
//
//      when std::abs(eVal(k)) > valTol
//
//      error_k = (rate_k/(1. - rate_k))*d_(k)/std::abs(eVal(k)) < tol
//
//      else
//
//      error_k = (rate_k/(1. - rate_k))*d_(k)/std::abs(valTol) < tol
//
//      This formula comes about by knowing that the eigenvalues form 
//      an ever increasing sequence of values with differences changing
//      by a factor r, so the sequence of differences can be summed to
//      get an approximate value
//
//      eVal_est = eVal_k +/- (rate_k/(1. - rate_k))*d_(k) 
// 
//      which gives rise to the relative error estimate used in the stopping
//      criterion. 
// 
        if((k%5) == 0)
        {
        D.resize(k,0.0);

        for(j = 0; j < k; j++)
        {
        D[j] = alpha[j];
        }

        E.resize(k-1);
        for(j = 0; j < k-1; j++)
        {
        E[j] =  beta[j];
        }

        eigMaxVector = getLargestSymTriEigValues(nValues,D,E);  
        lanczosMax = eigMaxVector[0];

        eigMinVector =  getLowestSymTriEigValues(nValues, D, E);
        lanczosMin   = eigMinVector[0];

        if(diffCompareFlag == 0)
        {
        eigMaxOld = lanczosMax; eigMinOld = lanczosMin;
        diffCompareFlag = 1;
        }
        else
        {

        relErrFactor = getRelErrorFactor(lanczosMax,tol);
        maxDiffError = std::abs(lanczosMax-eigMaxOld)/(relErrFactor);

        relErrFactor = getRelErrorFactor(lanczosMin,tol);
        minDiffError = std::abs(lanczosMin-eigMinOld)/(relErrFactor);

        if(rateCompareFlag == 0)
        {
        diffMaxA = std::abs(lanczosMax-eigMaxOld);
        diffMinA = std::abs(lanczosMin-eigMinOld);
        rateCompareFlag = 1;
        }
        else // estimate rate if there are sufficient digits
        {

        if(maxDiffError > 1.0e-09)
        {
        diffMaxB   = std::abs(lanczosMax-eigMaxOld);
        rateEigMax = diffMaxB/diffMaxA;
        }
        else
        {
        rateEigMax = 0.95;
        }


        if(minDiffError > 1.0e-09)
        {
        diffMinB   = std::abs(lanczosMin-eigMinOld);
        rateEigMin = diffMinB/diffMinA;
        }
        else
        {
        rateEigMin = 0.95;
        }

        rateFactorMax = 9.0;
        rateFactorMin = 9.0;
        if(rateEigMax < 0.90) rateFactorMax = rateEigMax/(1.0-rateEigMax);
        if(rateEigMin < 0.90) rateFactorMin = rateEigMin/(1.0-rateEigMin);

        diffMaxA = diffMaxB;
        diffMinA = diffMinB;
        }

        eigMaxOld = lanczosMax;
        eigMinOld = lanczosMin;

        minDiffError *= rateFactorMin;
        maxDiffError *= rateFactorMax;
        }
//
//      Compute exact errors if exact solution is available
// 
        if(exactDiagnosticFlag == 1)
        {
        relErrFactor = getRelErrorFactor(maxError,tol);
        maxError = std::abs(maxExact - lanczosMax)/relErrFactor;

        relErrFactor = getRelErrorFactor(minError,tol);
        minError = std::abs(minExact - lanczosMin)/relErrFactor;
        }

        if((verboseFlag == 1)&&(exactDiagnosticFlag == 1))
        {
        printf("%4ld %10.6e %10.6e %10.6e %10.6e %10.6e  %10.6e %10.6e %10.6e\n",k,
        lanczosMax, maxError, maxDiffError,rateEigMax,
        lanczosMin, minError, minDiffError,rateEigMin);
        }
        else if(verboseFlag == 1)
        {
        printf("%4ld %10.6e %10.6e %10.6e %10.6e %10.6e  %10.6e \n",k,
        lanczosMax, maxDiffError,rateEigMax,
        lanczosMin,minDiffError,rateEigMin);
        }

        // Set exit flag here based on which eigenvalues we are
        // tracking.

	    if((maxEigFlag)&&(minEigFlag))
	    {
	    if( (minDiffError < tol) && (maxDiffError < tol) ) exitFlag = true;
	    }
	    else if(maxEigFlag)
	    {
	    if(maxDiffError < tol) exitFlag = true;
	    }
	    else if(minEigFlag)
	    {
	    if(minDiffError < tol) exitFlag = true;
	    }

        }
}
//
//  Insure max and min eigenvalue are ordered properly
//
    double lanczosTmp;
    if(lanczosMin > lanczosMax) 
    {
    lanczosTmp = lanczosMin;
    lanczosMin = lanczosMax;
    lanczosMax = lanczosTmp;
    }

    minEigValue = lanczosMin;
    maxEigValue = lanczosMax;
}



double getMinEigenvalue(double errorTolerance, Vtype& v, Vtype& w, Vtype& wTmp, Otype &oP, 
VRandomizeOpType& randOp)
{
     double minEigValue = 0;
     double maxEigValue = 0;

     bool   minFlagCache = minEigFlag;
     bool   maxFlagCache = maxEigFlag;

     setMinEigStopCondition();
     getMinMaxEigenvalues(errorTolerance, v,  w, wTmp,oP,  randOp, minEigValue, maxEigValue);

     minEigFlag = minFlagCache;
     maxEigFlag = maxFlagCache;

     return minEigValue;
}

double getMaxEigenvalue(double errorTolerance, Vtype& v, Vtype& w, Vtype& wTmp, Otype &oP, VRandomizeOpType& randOp)
{
     double minEigValue = 0;
     double maxEigValue = 0;

     bool   minFlagCache = minEigFlag;
     bool   maxFlagCache = maxEigFlag;

     setMaxEigStopCondition();
     getMinMaxEigenvalues(errorTolerance, v,  w, wTmp,oP,  randOp, minEigValue, maxEigValue);

     minEigFlag = minFlagCache;
     maxEigFlag = maxFlagCache;

     return maxEigValue;
}

    long   iterationMaximumCount;
    long   iterCount;

    int            verboseFlag;
    int    exactDiagnosticFlag;
    double         minExact;
    double         maxExact;

    bool         maxEigFlag;
    bool         minEigFlag;


    std::vector<double>  alpha;
    std::vector<double>   beta;

    std::vector<double>  D;
    std::vector<double>  E;
    std::vector<double>  eigMinVector;
    std::vector<double>  eigMaxVector;

    Dstebz_C        triEigRoutine;



std::vector<double>  getLargestSymTriEigValues(long nValues, std::vector<double> & Diag,  std::vector<double> & UpDiag)
{
	long i;

	long N = Diag.size();

    std::vector<double>  eVals(N);
	std::vector<double>  eValsReturn(nValues);
//
//  Input paramters 
//
    char range    =  'I';
    char order    =  'E';
    long n        =    N;
    double vLower =  0.0;
    double vUpper =  0.0;
    long   iLower =    N-(nValues-1);       // lower computed eigenvalue index
    long   iUpper =    N;                   // upper computed eigenvalue index

    double abstol = 1.0e-14;

    double* dPtr = &Diag[0];
    double* uPtr = &UpDiag[0];
//
//  Output parameters 
//
    long mFound;   // number of eigenvalues found
    long nsplit;   // number of diagonal blocks

    double* ePtr   = &eVals[0]; // array for the eigenvalues
    long*   iblock = new long[N];
    long*   isplit = new long[N];

    double* work   = new double[4*N];   // work array
    long*  iwork   = new long[3*N];     // work array 

    long   info;

    triEigRoutine.dstebz(range, order, &n, &vLower, &vUpper, &iLower, &iUpper,
    &abstol, dPtr, uPtr, &mFound, &nsplit, ePtr, iblock, isplit, work, iwork, 
    &info);

    /* extract return eigenvalues */

	for(i = 0; i < nValues; i++)
    {eValsReturn[i] = eVals[i];}

	/* clean up */

	delete [] iblock;
	delete [] isplit;
	delete [] work;
	delete [] iwork;

	return eValsReturn;
}

std::vector<double>  getLowestSymTriEigValues(long nValues, std::vector<double> & Diag,  std::vector<double> & UpDiag)
{
	long i;

	long N = Diag.size();

    std::vector<double>  eVals(N);
	std::vector<double>  eValsReturn(nValues);
//
//  Input paramters 
//
    char range    =  'I';
    char order    =  'E';
    long n        =    N;
    double vLower =  0.0;
    double vUpper =  0.0;
    long   iLower =    1;       // lower computed eigenvalue index
    long   iUpper =    nValues; // upper computed eigenvalue index

    double abstol = 1.0e-14;

    double* dPtr = &Diag[0];
    double* uPtr = &UpDiag[0];
//
//  Output parameters 
//
    long mFound;   // number of eigenvalues found
    long nsplit;   // number of diagonal blocks

    double* ePtr   = &eVals[0]; // array for the eigenvalues
    long*   iblock = new long[N];
    long*   isplit = new long[N];

    double* work   = new double[4*N];   // work array
    long*  iwork   = new long[3*N];     // work array 

    long   info;

    triEigRoutine.dstebz(range, order, &n, &vLower, &vUpper, &iLower, &iUpper,
    &abstol, dPtr, uPtr, &mFound, &nsplit, ePtr, iblock, isplit, work, iwork, 
    &info);

    /* extract return eigenvalues */

	for(i = 0; i < nValues; i++)
    {eValsReturn[i] = eVals[i];}

	/* clean up */

	delete [] iblock;
	delete [] isplit;
	delete [] work;
	delete [] iwork;

	return eValsReturn;
}

};

#undef LANCZOS_MAX_MIN_FINDER_ITERATION_MAX
#undef LANCZOS_SMALL_TOL_
#endif
 
