/**
                         RayleighChebyshev.h

   A templated class with member functions for computing eigenpairs 
   corresponding to the lowest eigenvalues of a linear operator. It is 
   assumed that all of the eigenvalues of the operator are real and 
   that there is a basis of orthogonal eigenvectors. The routine is designed
   for symmetric linear operators, but symmetry is not exploited in
   the implementation of the procedure.

   The eigenvalues are returned in a std::vector<double>  instance
   while the eigenvectors are internally allocated and returned in
   a std::vector<Vtype> class instance.

   OpenMP multi-thread usage is enabled by defining _OPENMP

   Note: _OPENMP is automatically defined if -fopenmp is specified 
   as part of the compilation command.

   See the samples for usage. 

   The minimal functionality required of the classes 
   that are used in this template are

   Vtype  
   ---------
   A std::vector class with the following member functions:

   Vtype()                            (null constructor)
   Vtype(const Vtype&)                (copy constructor)

   initialize()                       (null initializer)
   initialize(const Vtype&)           (copy initializer)
  
   operator =                         (duplicate assignemnt)
   operator +=                        (incremental addition)
   operator -=                        (incremental subtraction)
   operator *=(double alpha)          (scalar multiplication)

   double dot(const Vtype&)           (dot product)
   long getDimension() const          (returns dimension of vectors)

   if _VBLAS_ is defined, then the Vtype class must also possess member functions

   double nrm2()                                            (2-norm based on std::vector dot product)
   void   scal(double alpha)                                (scalar multiplication)
   void   axpy(double alpha,const Vtype& x)                 (this = this + alpha*x)
   void   axpby(double alpha,const Vtype& x, double beta)   (this = alpha*x + beta*this)

   If OpenMP is defined, then the std::vector class should NOT SET any class or static
   variables of the std::vector class arguments to copy, dot, or axpy. Also,
   no class variables or static variables should be set by nrm2().


   ############################################################################
   
   Otype
   ----------

   An operator class with the following member function:

   void apply(Vtype& V)


   which applies the operator to the argument V and returns the result in V.

   If OpenMP is used then Otype must have a copy constructor of the form

   Otype(const Otype& O);

   ############################################################################

   VRandomizeOpType
   ----------

   An opearator class with the following member function:

   void randomize(Vtype& V)     

   which initializes the elements of the Vtype std::vector V to have random values.

   ############################################################################


	###########################################################################

	!!!! Important restriction on the std::vector classes and operator classes
	used by the RayleighChebyshev template.

	###########################################################################

	When specifying a std::vector class to be used with a RayleighChebyshev instance, it is critical
	that the copy constructor handle null instances correctly (e.g. instances that were created
	with the null constructor).

	Specifically, one cannot assume that the input argument to the copy constructor is a
	non-null instance and one has to guard against copying over data that doesn't exist in
	the copy constructor code.

	This coding restriction arises because of the use of stl::vectors of the specified
	std::vector class; when intializing it creates a null instance and apparently calls the
	copy constructor to create the duplicates required of the array, rather than
	calling the null constructor multiple times.

	The symptoms of not doing this are segmentation faults that occur when one
	tries to copy data associated with a null instance.


	When using multi-threaded execution it is also necessary that the operator classes
	specified in the template also have copy constructors that handle null
	input instances correctly. This restriction arises because of the use of
	stl::vectors to create operator instances for each thread.

---

   External dependencies: None
 
   Reference:

   Christopher R. Anderson, "A Rayleigh-Chebyshev procedure for finding
   the smallest eigenvalues and associated eigenvectors of large sparse
   Hermitian matrices" Journal of Computational Physics,
   Volume 229 Issue 19, September, 2010.

   Author Chris Anderson July 12, 2005
   Version : June 19, 2015  
*/
/*
#############################################################################
#
# Copyright 2005-2015 Chris Anderson
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


#include <iostream>
#include <cstdlib>
#include <vector>


#ifndef RAYLEIGH_CHEBYSHEV_
#define RAYLEIGH_CHEBYSHEV_

#define JACOBI_TOL                   1.0e-12
#define DEFAULT_MAX_INNER_LOOP_COUNT 10000
#define DEFAULT_POLY_DEGREE_MAX      100

#ifndef  RAYLEIGH_CHEBYSHEV_SMALL_TOL_
#define  RAYLEIGH_CHEBYSHEV_SMALL_TOL_ 1.0e-10
#endif

#include "LanczosCpolyOperator.h"
#include "LanczosCpoly.h"
#include "LanczosMaxMinFinder.h"
#include "RC_Double2Darray.h"
#include "JacobiDiagonalizer.h"

#ifdef _TIMING_
#include "ClockIt.h"
#endif

#ifdef _OPENMP
#include "cOpThreadArray.h"
#include <omp.h>
#endif

//
//
// Jan. 16, 2014
//        ToDo: 
//
//        Rework the multi-threading of the computation of Vt*A*V. Use vTemp and
//        use the copies of the operator that have been created in the
//        cOpThreadArray, cOpArray for each thread.
//
//        Add the capability to call a dense eigensolver from lapack instead of
//        using Jacobi's method.

template <class Vtype, class Otype, class VRandomizeOpType >
class RayleighChebyshev
{

    public : 

    RayleighChebyshev()
    {
    VtAVdataPtr          = 0;
    VtAVeigValueDataPtr  = 0;
    VtAVeigVectorDataPtr = 0;
    verboseFlag          = false;
    eigDiagnosticsFlag   = false;
	verboseSubspaceFlag  = false;
    jacobiMethod.tol     = JACOBI_TOL;
    minIntervalPolyDegreeMax   = DEFAULT_POLY_DEGREE_MAX;
    minEigValueEst       = 0.0;
    maxEigValueEst       = 0.0;
    guardValue           = 0.0;
    intervalStopConditionFlag = false;
    hardIntervalStopFlag  = false;

    nonRandomStartFlag   = false;
    fixedIterationCount  = -1;
    maxInnerLoopCount    = DEFAULT_MAX_INNER_LOOP_COUNT;

    orthogSubspacePtr    = 0;
    }


    // This routine determines the factor used for estimating
    // relative errors.
    // 
    //  || Rel Error || = || Err ||/|| val || when ||val|| > (default small tolerance)/tol
    //  
    //  || Rel Error || = || Err ||/ ((default small tolerance)/tol)  otherwise
    //
    // 

	double getRelErrorFactor(double val, double tol)
	{
	double relErrFactor = 1.0;

    if(std::abs(val)*tol > RAYLEIGH_CHEBYSHEV_SMALL_TOL_ ){relErrFactor = std::abs(val);}
    else                                              {relErrFactor = RAYLEIGH_CHEBYSHEV_SMALL_TOL_/tol;}
    return relErrFactor;
	}

    void setOrthogonalSubspace(std::vector < Vtype >* orthogSubspacePtr)
    {
    this->orthogSubspacePtr = orthogSubspacePtr;
    }

    void clearOrthogonalSubspace()
    {
    this->orthogSubspacePtr = 0;
    }

    void setMaxInnerLoopCount(long val)
    {maxInnerLoopCount  = val;}

    void resetMaxInnerLoopCount()
    {maxInnerLoopCount  = DEFAULT_MAX_INNER_LOOP_COUNT;}

    void setFixedIterationCount(long val)
    {fixedIterationCount  = val;}

    void clearFixedIteratonCount()
    {fixedIterationCount  = -1;}

    void setNonRandomStartFlag(bool val = true)
    {nonRandomStartFlag  = val;}

    void clearNonRandomStartFlag()
    {nonRandomStartFlag  = false;}

    void setVerboseFlag(bool val = true)
    {verboseFlag = val;}

    void clearVerboseFlag()
    {verboseFlag = false;}

    void setVerboseSubspaceFlag(bool val = true)
    {verboseSubspaceFlag = val;}

    void clearVerboseSubspaceFlag()
    {verboseSubspaceFlag = false;}

    void setEigDiagnosticsFlag(bool val = true)
    {eigDiagnosticsFlag = val;}

    void clearEigDiagnosticsFlag()
    {eigDiagnosticsFlag = 0;}
    
      
    double getMinEigValueEst()
    {return minEigValueEst;}
    
    double getMaxEigValueEst()
    {return maxEigValueEst;}

    void setMinIntervalPolyDegreeMax(long polyDegMax)
    {
    	minIntervalPolyDegreeMax = polyDegMax;
    }
    
    double getGuardEigenvalue()
    {return guardValue;}

    void setIntervalStopCondition(bool val = true)
    {
    intervalStopConditionFlag = val;
    }

    void clearIntervalStopCondition()
    {
    intervalStopConditionFlag = false;
    }

    void setHardIntervalStop(bool val = true)
    {
    hardIntervalStopFlag = val;
    }

    void clearHardIntervalStop()
    {
    hardIntervalStopFlag = false;
    }

    void getMinEigAndMaxEig(double iterationTol,Vtype& vStart,Otype& oP, 
    VRandomizeOpType& randOp, double&  minEigValue, 
    double& maxEigValue)
    {
    //
    // Create temporaries based on vStart
    //
    Vtype w(vStart);
    Vtype wTmp(vStart);

    if(verboseFlag) 
    {lanczosMaxMinFinder.setVerboseFlag();}

    // Specify accurate estimates of largest and smallest

    lanczosMaxMinFinder.setMinMaxEigStopCondition();

    lanczosMaxMinFinder.getMinMaxEigenvalues(iterationTol,vStart,w,wTmp,oP,
    randOp,minEigValue,maxEigValue);

    if(verboseFlag) 
    {printf("Minimum_Eigenvalue : %10.5g  \nMaximum_Eigenvalue : %10.5g \n",minEigValue,
     maxEigValue);}

    minEigValueEst  = minEigValue;
    maxEigValueEst  = maxEigValue;
    }

    //
    // This routine obtains the spectral estimates required for the core Rayleigh-Chebyshev
    // routine, in particular, an accurate estimate of the largest eigenvalue and an
    // estimate of the smallest eigenvalue.
    //
    void getInitialRCspectralEstimates(double iterationTol,Vtype& vStart,Otype& oP,
    VRandomizeOpType& randOp, double&  minEigValue,
    double& maxEigValue)
    {
    //
    // Create temporaries based on vStart
    //
    Vtype w(vStart);
    Vtype wTmp(vStart);

    if(verboseFlag )
    {lanczosMaxMinFinder.setVerboseFlag();}

    // Specify accurate estimates of largest. Here we use the
    // fact that the core procedure only requires a good upper
    // bound and a reasonably good lower bound. 
    //
    // If this turns out to be problematic, then one can get  
    // accurate estimates of both the lower and the upper  
    // using getMinEigAndMaxEig(...) and then 
    // invoking a version of the eigen system routine that 
    // 

    lanczosMaxMinFinder.setMaxEigStopCondition();

    lanczosMaxMinFinder.getMinMaxEigenvalues(iterationTol,vStart,w,wTmp,oP,
    randOp,minEigValue,maxEigValue);

    if(verboseFlag)
    {printf("Minimum_Eigenvalue : %10.5g  \nMaximum_Eigenvalue : %10.5g \n",minEigValue,
     maxEigValue);}

    minEigValueEst  = minEigValue;
    maxEigValueEst  = maxEigValue;
    }


//
//  Computes the lowest eigCount eigenvalues and eigenvectors.
//
//  Input:
//
//  eigCount    : The desired number of eigenvalues
//  minEigValue : The minimum eigenvalue
//  maxEigValue : The maximum eigenvalue
//
// e.g. minEigValue <= lambda <= maxEigValue for all eigenvalues lambda.
//
// subspaceTol  : Stopping tolerance.
//
// An eigenvalue lambda is considered converged when
//
// | lambda^(n) - lambda^(n-1) | < subspaceTol*(1 + |lambda^(n)|)
//
// subspaceIncrementSize and bufferSize : sizes used to determine the
// dimension of the subspace used to determine eigenpars.
//
// The dimension of the subspace is subspaceIncrementSize + bufferSize,
// and the lowest subspaceIncrementSize are evaluated for convergence.
// The remaining subspace dimensions (of size bufferSize) are used to
// increase the gap between the desired states and other states in
// order to improve performance.
//
// vStart : A std::vector instance used as a template for for the
//          the construction of all eigenvectors computed.
//
//    oP  : The linear operator whose eigenpairs are sought
//
// randOp : An operator that assigns random values to the elements
//          of a std::vector. Used for initial guesses.
//
// Input/Output
//
// eigValues  : std::vector of doubles to capture the eigenvalues
//
// eigVectors : std::vector of vectors to capture the eigenvectors.
//              If this std::vector is non-empty, then the non-null
//              vectors are used as starting vectors for the
//              subspace iteration.
//
//
    //
    // Restrictions on the spectral ranges estimates:
    //
    // maxEigValueBound > maxEigValue
    // minEigValueEst   > minEigValue
    //
    long getMinEigenSystem(long eigCount, double minEigValueEst, double maxEigValueBound,
    double subspaceTol, long subspaceIncrementSize, long bufferSize, 
    Vtype& vStart, Otype& oP, VRandomizeOpType& randOp, std::vector<double>& eigValues,
    std::vector < Vtype > & eigVectors)
    {

    long maxEigensystemDim = eigCount;
    double lambdaMax       = maxEigValueBound;

    this->clearIntervalStopCondition();

    return getMinIntervalEigenSystem_Base(minEigValueEst, lambdaMax, maxEigValueBound,
    subspaceTol, subspaceIncrementSize, bufferSize, maxEigensystemDim, 
    vStart, oP, randOp, eigValues, eigVectors);
    }
//
//  Computes the lowest eigCount eigenvalues and eigenvectors
//
    long getMinEigenSystem(long eigCount, double subspaceTol, long subspaceIncrementSize, 
    long bufferSize, Vtype& vStart, Otype& oP, VRandomizeOpType& randOp, 
    std::vector<double>&  eigValues, std::vector < Vtype > & eigVectors)
    {

    double minFinderTol              = subspaceTol;
    
    double minEigValue;
    double maxEigValue;

    getInitialRCspectralEstimates(minFinderTol,vStart, oP, randOp,minEigValue, maxEigValue);

    //
    // Increase maxEigValue slightly to be on the safe side if we don't have
    // the identity.
    //
    if(std::abs(maxEigValue - minEigValue) > 1.0e-12)
    {
    maxEigValue += 0.001*std::abs(maxEigValue - minEigValue);
    }

    this->clearIntervalStopCondition();

    return getMinEigenSystem(eigCount, minEigValue, maxEigValue,
    subspaceTol, subspaceIncrementSize, bufferSize, 
    vStart, oP, randOp, eigValues, eigVectors);
    }

    //
    // return value >= 0 returns the number of eigenpairs found
    // return value <  0 returns error code
    // 

    long getMinIntervalEigenSystem(double lambdaMax, double subspaceTol, 
    long subspaceIncrementSize, long bufferSize, long maxEigensystemDim, 
    Vtype& vStart,Otype& oP, VRandomizeOpType& randOp, std::vector<double>&  eigValues,
    std::vector < Vtype > & eigVectors)
    {
    double minFinderTol              = subspaceTol;
    
    double minEigValue;
    double maxEigValue;

    // Get accurate estimates of both the largest and smallest eigenvalues 

    getMinEigAndMaxEig(minFinderTol,vStart,oP, randOp, minEigValue, maxEigValue);
    
    // Quick return if the upper bound is smaller than the smallest eigenvalue 

    if(lambdaMax < minEigValue) 
	{
    return 0;
    eigValues.clear();
    eigVectors.clear();
    }

    //
    // Increase maxEigValue slightly to be on the safe side if we don't have
    // the identity.
    //
    if(std::abs(maxEigValue - minEigValue) > 1.0e-12)
    {
    maxEigValue += 0.001*std::abs(maxEigValue - minEigValue);
    }

    this->setIntervalStopCondition();

    return getMinIntervalEigenSystem_Base(minEigValue, lambdaMax, maxEigValue,
    subspaceTol, subspaceIncrementSize, bufferSize, maxEigensystemDim,
    vStart, oP, randOp, eigValues, eigVectors);
    }


    //
    // Restrictions on the spectral ranges estimates:
    //
    // maxEigValueBound > maxEigValue
    // minEigValueEst   > minEigValue
    //
    // While it is technically ok to have lambdaMax < minEigValueEst, in such 
    // cases it's probably better to use the version that estimates the 
    // the minimal eigenvalue for you. 
    // 

    long getMinIntervalEigenSystem(double minEigValueEst, double lambdaMax, double maxEigValueBound,
    double subspaceTol, long subspaceIncrementSize, long bufferSize, long maxEigensystemDim,
    Vtype& vStart, Otype& oP, VRandomizeOpType& randOp, std::vector<double>&  eigValues,
    std::vector < Vtype > & eigVectors)
    {
    this->setIntervalStopCondition();

    return getMinIntervalEigenSystem_Base(minEigValueEst, lambdaMax, maxEigValueBound,
    subspaceTol, subspaceIncrementSize, bufferSize, maxEigensystemDim, 
    vStart, oP, randOp, eigValues, eigVectors);
    }

//
//  Base routine
//
//  If the nonRandomStart flag is set, the code will use all available or up to
//  subspaceSize = subspaceIncrementSize + bufferSize vectors that are
//  specified in the eigVectors input argument.
//
protected:

    long getMinIntervalEigenSystem_Base(double minEigValue, double lambdaMax, double maxEigValue,
    double subspaceTol, long subspaceIncrementSize, long bufferSize, long maxEigensystemDim, 
    Vtype& vStart, Otype& oP, VRandomizeOpType& randOp, std::vector<double>&  eigValues,
    std::vector < Vtype > & eigVectors)
    {

    // Insure that subspaceTol isn't too small

    if(subspaceTol < RAYLEIGH_CHEBYSHEV_SMALL_TOL_ ) {subspaceTol = RAYLEIGH_CHEBYSHEV_SMALL_TOL_; }
    double relErrFactor;

    //
    // Delete any old eigenvalues and eigenvectors if not random start, otherwise
    // use input eigenvectors for as much of the initial subspace as possible

    eigValues.clear();

    if(not nonRandomStartFlag)
    {
    eigVectors.clear();
    }

    long returnFlag = 0;

#ifdef _TIMING_
    ClockIt     timer;
#endif
    //
    //#######################################################
    //  Find eigenpairs in [lambda_min, lambdaMax] 
    //#######################################################
    //
    long   minIntervalRepetitionCount =   1;
  
 
    double   lambdaStar;
    long   subspaceSize;   
    long      foundSize;

    std::vector<double>     oldEigs;
    std::vector<double>    eigDiffs;
    std::vector<double> oldEigDiffs;
    double             eigDiff;

    lambdaStar        = maxEigValue;
    subspaceSize      = subspaceIncrementSize +  bufferSize;
    foundSize         = 0;

    //
    // Reset subspace sizes (if necessary) to make sure 
    // it's not larger than the subspace dimension.
    //

    long vectorDimension = vStart.getDimension();
    if(subspaceSize > vectorDimension)
    {
     subspaceSize = vectorDimension;
     if(bufferSize > vectorDimension)
     {
        bufferSize = vectorDimension;
        subspaceIncrementSize = 0;
     }
     else
     {
        subspaceIncrementSize = vectorDimension - bufferSize;
     }
    }
    
    oldEigs.resize(subspaceSize,0.0);
    eigDiffs.resize(subspaceIncrementSize+1,1.0);
    oldEigDiffs.resize(subspaceIncrementSize+1,1.0);

    //
    // vArray contains the current subspace, and is of size
    // equal to the sum of number of desired states and
    // the number of  buffer vectors
    //
    //
    // eigVectors is the array of eigenvectors that have
    // been found
    //

    vArray.resize(subspaceSize);
    vArrayTmp.resize(subspaceSize);

    VtAV.initialize(subspaceSize,subspaceSize);
    VtAVeigVector.initialize(subspaceSize,subspaceSize);
    VtAVeigValue.resize(subspaceSize,0.0);

    double* VtAVdataPtr;
    double* VtAVeigValueDataPtr;
    double* VtAVeigVectorDataPtr;

    long i; long j; long k;

    double rkk; 
    double rkj;

    long   starDegree     = 0;
    long   starDegreeSave = 0;
    double starBoundSave  = 0.0;
    double shift          = 0.0;
    double starBound      = 0.0;
    double maxEigDiff     = 0.0;
    double eigDiffRatio   = 0.0;

    long   innerLoopCount  = 0;

    double vtvEig;
    double vtvEigCheck;

    long indexA_start; long indexA_end;
    long indexB_start; long indexB_end;

    //
    // Initialize subspace vectors using random vectors, or input
    // starting vectors if the latter is specified.
    //

    if(not nonRandomStartFlag)
    {
    	for(k = 0; k < subspaceSize; k++)
    	{
    		vArray[k].initialize(vStart);
    		randOp.randomize(vArray[k]);
    		vArrayTmp[k].initialize(vStart);
    	}
    }
    else
    {
    	if(subspaceSize > (long)eigVectors.size())
    	{
    		for(k = 0; k < (long)eigVectors.size(); k++)
    		{
    			vArray[k].initialize(eigVectors[k]);
    			vArrayTmp[k].initialize(vStart);
    		}
    		for(k = (long)eigVectors.size(); k < subspaceSize; k++)
    		{
    		vArray[k].initialize(vStart);
    		randOp.randomize(vArray[k]);
    		vArrayTmp[k].initialize(vStart);
    		}
    	}
    	else
    	{
    		for(k = 0; k <  subspaceSize; k++)
    		{
    			vArray[k].initialize(eigVectors[k]);
    			vArrayTmp[k].initialize(vStart);
    		}
    	}

    	// Clear the pre-existing eigenvalues

    	eigVectors.clear();
    }

//  Orthogonalize working subspace (vArray) to orthogSubspace if it's been specified

    if(orthogSubspacePtr != 0)
    {
    indexA_start = 0;
    indexA_end   = subspaceSize-1;
    indexB_start = 0;
    indexB_end   = (long)orthogSubspacePtr->size() -1;

    OrthogonalizeAtoB(vArray, indexA_start,indexA_end, *orthogSubspacePtr, indexB_start, indexB_end);
    }

    vTemp.initialize(vStart);
    
#ifdef _OPENMP
    cOpThreadArray < Vtype, Otype > cOpArray(oP);
#endif
//
//  ################## Main Loop ######################
//
    if(fixedIterationCount > 0)
    {
    maxInnerLoopCount  = fixedIterationCount;
    }

    int  exitFlag = 0;
    long foundCount;
    long guardStopValue;

#ifdef _TIMING_
    double orthoTime   = 0.0;
    long   orthoCount  = 0;

    double applyTime   = 0.0;


    double eigTime     = 0.0;
    long   eigCount    = 0;

    double orthoTimeCumulative   = 0.0;
    double applyTimeCumulative   = 0.0;
    double eigTimeCumulative     = 0.0;

    long   orthoCountCumulative  = 0;

    double eigCountCumulative    = 0;

#endif

    long   applyCount  = 0;
    long   applyCountCumulative  = 0;

    while(exitFlag == 0)
    {
#ifdef _TIMING_
    orthoTime   = 0.0;
    orthoCount  = 0;
    applyTime   = 0.0;
    applyCount  = 0;
    eigTime     = 0.0;
    eigCount    = 0;
#endif
//
//  Initialize old eigenvalue array using buffer values.
//  This step is done for cases when the routine
//  is called to continue an existing computation.
//
    for(k = bufferSize; k < subspaceSize; k++)
    {
         oldEigs[k] = oldEigs[bufferSize];
    }
//
//  Randomize buffer vectors in the case of random start, or
//  after first increment.
//
    if((applyCountCumulative > 0)||(not nonRandomStartFlag))
    {
    for(k = bufferSize; k < subspaceSize; k++)
    {
        randOp.randomize(vArray[k]);
    }}

#ifdef _TIMING_
    timer.start();
#endif

//  Orthogonalize working subspace (vArray) to subspace of found eigenvectors (eigVectors)
//
//    indexA_start = 0;
//    indexA_end   = subspaceSize-1;
//    indexB_start = 0;
//    indexB_end   = foundSize-1;
//
//    OrthogonalizeAtoB(vArray, indexA_start,indexA_end, eigVectors, indexB_start, indexB_end);

//  Orthogonalize working subspace (vArray)

    orthogonalizeVarray(subspaceSize);

#ifdef _TIMING_
    timer.stop();
    orthoTime  += timer.getSecElapsedTime();
    orthoCount += 1;
#endif 


    maxEigDiff     = subspaceTol + 1.0;
    lambdaStar     = maxEigValue;
    eigDiffRatio   = 1.0;
    innerLoopCount = 0;
    starDegreeSave = 0;
    starBoundSave  = 0.0;

    double eMin; double eMax;

    while((maxEigDiff > subspaceTol)&&(innerLoopCount < maxInnerLoopCount))
    {
//  
//  Compute filter polynomial parameters 
//
    shift          = -minEigValue;
    starDegreeSave =  starDegree;
    starBoundSave  =  starBound;
    starDegree = 0;
//
//  Conditions to check for the identity matrix, and for multiplicity > subspace dimension,
//  which can lead to an erroneous increase in polynomial filtering degree.
//

    relErrFactor = getRelErrorFactor(minEigValue,subspaceTol);
    eigDiff = std::abs(minEigValue - lambdaStar)/relErrFactor;
    eMin = eigDiff;

    relErrFactor = getRelErrorFactor(maxEigValue,subspaceTol);
    eMax = std::abs(maxEigValue - lambdaStar)/relErrFactor;

    if // Identity matrix
    ((eMin < subspaceTol)&&(eMax < subspaceTol))
    {
    	   starDegree = 1;
    	   starBound  = maxEigValue;
    }                                                   // For multiplicity > subspace dimension
    else if((innerLoopCount > 3)&&(eigDiffRatio < .2)) // .2 is slightly less than the secondary
    {                                                   // maximum of the Lanczos C polynmoial
    	  starDegree = starDegreeSave;
    	  starBound  = starBoundSave;
    }
//
//#############################################################################
//
//
//  Find the polynomial that captures eigenvalues between current minEigValue
//  and lambdaStar. If the required polynomial is greater than minIntervalPolyDegreeMax,
//  set starDegree = minIntervalPolyDegreeMax. This is a safe modification, as
//  it always safe to use a polynomial of lower degree than required.
//
    if(starDegree == 0)
    {
    cPoly.getStarDegreeAndSpectralRadius(shift,maxEigValue,
    lambdaStar,minIntervalPolyDegreeMax, starDegree,starBound);
    }

//
// 	Only allow for increasing degrees. A decrease in degree
// 	arises when orthogonalization with respect to previously
// 	found eigenvectors isn't sufficient to insure
// 	their negligible contribution to the working subspace
// 	after the polynomial in the operator is applied. The
// 	fix here is to revert to the previous polynomial.
//  Often when the eigensystems associated with  
//  working subspaces that overlap, the eigenvalues
//  created won't be monotonically increasing. When this
//  occurs the problems are corrected by doing a final
//  projection of the subspace and an additional eigensystem
//  computation.
//
//  The monotonically increasing degree is only enforced
//  for a given subspace. When the subspace changes because 
//  because of a shift, the degree and bound are set to 1 
//  and the maximal eigenvalue bound respectively. 
// 
//
    if(starDegree < starDegreeSave)
    {
    starDegree    = starDegreeSave;
    starBound     = starBoundSave;
    }

#ifdef _OPENMP
    cOpArray.initialize(starDegree,minIntervalRepetitionCount,starBound,shift);
#else
    cOp.initialize(starDegree,minIntervalRepetitionCount,starBound,shift,oP);
#endif

#ifdef _TIMING_
    timer.start();
#endif 


#ifdef _OPENMP
	#pragma omp parallel for \
	private(k) \
	firstprivate(cOpArray) \
	schedule(static,1)
    for(k = 0; k < subspaceSize; k++)
    {
    cOpArray.apply(vArray[k]);
    }
#else
    for(k = 0; k < subspaceSize; k++)
    {
    cOp.apply(vArray[k]);
    }
#endif

    applyCount += 1;

#ifdef _TIMING_
    timer.stop();
    applyTime  += timer.getSecElapsedTime();
#endif 

#ifdef _TIMING_
    timer.start();
#endif 


//  Orthogonalize working subspace (vArray) to subspace of found eigenvectors (eigVectors)
//  It is important to do this before orthgonalizing the new vectors with respect to each other.

    indexA_start = 0;
    indexA_end   = subspaceSize-1;
    indexB_start = 0;
    indexB_end   = foundSize-1;

    OrthogonalizeAtoB(vArray, indexA_start,indexA_end, eigVectors, indexB_start, indexB_end);


    //  Orthogonalize working subspace (vArray) to orthogSubspace if it's been specified

    if(orthogSubspacePtr != 0)
    {
    indexA_start = 0;
    indexA_end   = subspaceSize-1;
    indexB_start = 0;
    indexB_end   = (long)orthogSubspacePtr->size() -1;

    OrthogonalizeAtoB(vArray, indexA_start,indexA_end, *orthogSubspacePtr, indexB_start, indexB_end);
    }


#ifndef _VBLAS_

//  Orthogonalize the subspace vectors using Modified Gram-Schmidt

    for(k = 1; k <= subspaceSize; k++)
    {
        rkk     = std::sqrt(vArray[k-1].dot(vArray[k-1]));
        vArray[k-1] *= 1.0/rkk;
        for(j = k+1; j <= subspaceSize; j++)
        {
            rkj           =   vArray[k-1].dot(vArray[j-1]);
            vTemp         =   vArray[k-1];
            vTemp        *=  -rkj;
            vArray[j-1]  += vTemp;
        }
    }
#endif
#ifdef _VBLAS_
//  Orthogonalize the subspace vectors using   Modified Gram-Schmidt

    for(k = 1; k <= subspaceSize; k++)
    {
        rkk     = vArray[k-1].nrm2(); 
        vArray[k-1].scal(1.0/rkk);
#ifdef _OPENMP
		#pragma omp parallel for \
		private(j,rkj) \
		schedule(static,1)
#endif
        for(j = k+1; j <= subspaceSize; j++)
        {
            rkj  =   vArray[j-1].dot(vArray[k-1]);
            vArray[j-1].axpy(-rkj,vArray[k-1]);
        }
    }
#endif
//
//#############################################################################
//#############################################################################
//
#ifdef _TIMING_
    timer.stop();
    orthoTime  += timer.getSecElapsedTime();
    orthoCount += 1;
#endif 


#ifdef _TIMING_
    timer.start();
#endif 
//
//  Form Vt*A*V 
//
    for(i = 0; i < subspaceSize; i++)
    {
        vTemp     = vArray[i];
        oP.apply(vTemp);
#ifdef _OPENMP
		#pragma omp parallel for \
		private(j) \
		schedule(static,1)
#endif
        for(j = i; j < subspaceSize; j++)
        {
        VtAV(j,i) = vArray[j].dot(vTemp);
        VtAV(i,j) = VtAV(j,i);
        }
    }

    /*
    cout << "Matrix " << endl;
    for(i = 0; i < subspaceSize; i++)
    {
    for(j = 0; j < subspaceSize; j++)
    {
    printf("%3.2e ",VtAV(i,j));
    }
    printf("\n");
    }
    cout << endl;
    */

//
//  Compute eigenvalues of  Vt*A*V

//
//  The jacobiMethod procedure returns the eigenvalues
//  and eigenvectors ordered from largest to smallest, e.g.
//
//  VtAVeigValue[0] >= VtAVeigValue[1] >= VtAVeigValue[2] ...
//
//
    VtAVdataPtr           = VtAV.getDataPointer();
    VtAVeigValueDataPtr   = &VtAVeigValue[0];
    VtAVeigVectorDataPtr  = VtAVeigVector.getDataPointer();

    jacobiMethod.getEigenSystem(VtAVdataPtr, subspaceSize, 
    VtAVeigValueDataPtr, VtAVeigVectorDataPtr);


    if(verboseSubspaceFlag)
    {
    printf("XXXX Subspace Eigs XXXX \n");
    for(i = 0; i < subspaceSize; i++)
    {
    printf("%3ld : %+10.5e \n",i,VtAVeigValue[subspaceSize - i - 1 ]);
    }
    printf("\n");
    printf("Shift      : %10.5e MaxEigValue : %10.5e \n ",shift,maxEigValue);
    printf("LambdaStar : %10.5e StarBound   : %10.5e StarDegree : %3ld \n",lambdaStar,starBound,starDegree); 
    printf("XXXXXXXXXXXXXXXXXXXXXXX \n");
    }
 
#ifdef _TIMING_
    timer.stop();
    eigTime  += timer.getSecElapsedTime();
    eigCount += 1;
#endif 


    maxEigDiff  = 0.0;
//
//  Check the smallest subspaceIncrementSize values for convergence.
//  In the case of finding the eigenvalues less than some specified
//  bound, we check the guard eigenvalue for convergence as well,
//  because this is the value that determines when
//  we've captured all less than the specified value.
//
    if(intervalStopConditionFlag) {guardStopValue = 1;}
    else                          {guardStopValue = 0;}

    if(subspaceIncrementSize == 0) // If the subspace fills out the remaining dimensions; no need to check
    {
    guardStopValue = 0; 
    } 


    for(i = 0; i < subspaceIncrementSize + guardStopValue; i++)
    {
    	oldEigDiffs[i] = eigDiffs[i];
    }
    
    for(i = 0; i < subspaceIncrementSize + guardStopValue; i++)
    {
    eigDiff = std::abs(VtAVeigValue[subspaceSize - i - 1] - oldEigs[subspaceSize - i - 1]);


    relErrFactor = getRelErrorFactor(oldEigs[subspaceSize - i - 1],subspaceTol);
    eigDiff = eigDiff/relErrFactor;
   
    eigDiffs[i] = eigDiff;
    maxEigDiff = (eigDiff > maxEigDiff)? eigDiff : maxEigDiff;
    }


    for(i = 0; i < subspaceSize; i++)
    {
    oldEigs[i] =VtAVeigValue[i];
    }
    
    //
    // Compute an average estimated convergence rate based upon components
    // for which the convergence tolerance has not been achieved. 
    //
    long diffCount = 0;
    eigDiffRatio  = 0.0;
    if(maxEigDiff > subspaceTol)
    {
    	for(i = 0; i < subspaceIncrementSize+1; i++)
    	{
          if(std::abs(oldEigDiffs[i]) > subspaceTol/10.0)
          {
          eigDiffRatio += std::abs(eigDiffs[i]/oldEigDiffs[i]);
          diffCount++;
          }
    	}
    	if(diffCount > 0) {eigDiffRatio /= (double)diffCount;}
    	else              {eigDiffRatio = 1.0;}
    } 

    if(verboseFlag == 1)
    {
    printf(" Degree %-5ld Eig Diff Max: %-10.5g   Eig Conv Factor: %-10.5g \n",starDegree,maxEigDiff,eigDiffRatio);
    }

    //
    // Force termination if we've filled out the subspace
    //
    if(subspaceIncrementSize == 0) maxEigDiff = 0.0;

    //
    // Update cPoly parameters based upon the eigensystem computation.
    //
    //
    // lambdaStar  : is reset to the largest eigenvalue currently computed
    // minEigValue : is reset when the subspace computation yields a 
    // minimum eigenvalue smaller than minEigValue. 
    //
    lambdaStar     = VtAVeigValue[0];
    minEigValue    = (minEigValue < VtAVeigValue[subspaceSize-1]) ? minEigValue : VtAVeigValue[subspaceSize-1];

    innerLoopCount++;
    }

    if(verboseFlag == 1)
    {
    if(( fixedIterationCount < 0) && (innerLoopCount >= maxInnerLoopCount))
    {
    printf(" Warning             : Maximal number of iterations taken before tolerance reached \n");
    printf(" Iterations taken    : %ld \n",innerLoopCount);
    printf(" Eig Diff Max        : %-10.5g \n",maxEigDiff);
    printf(" Requested Tolerance : %-10.5g \n",subspaceTol);
    }
    }


#ifndef _VBLAS_
    //
    // We have subspace convergence so we now create eigenvectors
    // from the current subspace
    //
    for(k = 0; k < subspaceSize; k++)
    {
        vArrayTmp[k]  = vArray[0];
        vArrayTmp[k] *= VtAVeigVector(0,k);
        for(i = 1; i < subspaceSize; i++)
        {
        vTemp =  vArray[i];
        vTemp *= VtAVeigVector(i,k);
        vArrayTmp[k] += vTemp;
        }

        rkk = vArrayTmp[k].dot(vArrayTmp[k]);
        vArrayTmp[k] *= 1.0/rkk;
    }
#endif
#ifdef _VBLAS_

   //
    // We have subspace convergence so we now create eigenvectors
    // from the current subspace
    //
#ifdef _OPENMP
		#pragma omp parallel for \
		private(i,k,rkk) \
		schedule(static,1)
#endif
    for(k = 0; k < subspaceSize; k++)
    {
        vArrayTmp[k] = vArray[0];
        vArrayTmp[k].scal(VtAVeigVector(0,k));


        for(i = 1; i < subspaceSize; i++)
        {
        vArrayTmp[k].axpy(VtAVeigVector(i,k),vArray[i]);
        }

        rkk = vArrayTmp[k].nrm2();
        vArrayTmp[k].scal(1.0/rkk);
    }
#endif

    foundCount = 0;
//
//  Capture the found eigenvalues
//
    long checkIndexCount;

    if(foundSize + subspaceSize < vectorDimension)
    {checkIndexCount = subspaceIncrementSize;}
    else 
    {checkIndexCount = subspaceSize;}

    for(i = 0; i < checkIndexCount; i++) 
    {
    vtvEig     = VtAVeigValue[subspaceSize - i - 1];
    relErrFactor = getRelErrorFactor(lambdaMax,subspaceTol);
    vtvEigCheck = (vtvEig - lambdaMax)/relErrFactor;

    if(vtvEigCheck < subspaceTol) foundCount++;
    }

    if((foundCount + foundSize) >= maxEigensystemDim)
    {
    foundCount =  maxEigensystemDim - foundSize;
    exitFlag   =  1;
    }

    if((foundCount + foundSize) >= vectorDimension)
    {
    foundCount =  vectorDimension - foundSize;
    exitFlag   =  1;
    }

    if(foundCount > 0)
    {
        expandVector(eigVectors,foundCount);
        expandArray (eigValues, foundCount);

        for(i = 0; i < foundCount; i++) 
        {
        eigVectors[foundSize + i] = vArrayTmp[subspaceSize - i - 1];
        eigValues[foundSize + i]  = VtAVeigValue[subspaceSize - i - 1];
        }
    
        foundSize += foundCount; 
        if(verboseFlag == 1)
        {
            printf("Found Count: %3ld Largest Eig: %-9.5g Lambda Bound: %-9.5g \n",foundSize, eigValues[foundSize-1], lambdaMax);
        }
    }


//
//  See if "guard" eigenvalue is greater than lambdaMax. 
//

    if(bufferSize > 0)
    {
    	vtvEig     = VtAVeigValue[bufferSize-1];
    }
    else
    {
    	vtvEig     = lambdaMax;
    }

    guardValue = vtvEig;
    relErrFactor = getRelErrorFactor(lambdaMax,subspaceTol);
    vtvEigCheck = (vtvEig - lambdaMax)/relErrFactor;


//  We assume that states are degenerate if |lambda[i] - lambda[i+1]|/relErrFactor < 10.0*subspaceTol
//  and hence require a relative gap of size 10.0*subspaceTol between lambdaMax and the guard std::vector
//  to insure that all vectors in the subspace associated with an eigenvalue with multiplicity > 1
//  are captured.

    if(hardIntervalStopFlag)
	{
    if(guardValue > lambdaMax)     {exitFlag = 1;}
	}
    else
    {
    if(vtvEigCheck >= 10.0*subspaceTol)     {exitFlag = 1;}
    }

    //
    // Shifting minEigenValue 
    //
    if((subspaceIncrementSize > 0)||(bufferSize == 0))     // New minEigValue = largest of found eigenvalues
    {                                                      // They are reversed ordered, so [bufferSize] is the largest
    minEigValue = VtAVeigValue[bufferSize];
    }
    else
    {
    minEigValue = VtAVeigValue[bufferSize-1]; 
    }

    // Reset star degree and bound

    starDegree = 1;
    starBound = maxEigValue;
//
//  check for exceeding std::vector space dimension
//

    if(not exitFlag)
    {
    if((foundSize + bufferSize) >= vectorDimension)
    {
     bufferSize            = vectorDimension - foundSize;
     subspaceIncrementSize = 0;
     subspaceSize          = bufferSize;

     VtAV.initialize(subspaceSize,subspaceSize);
     VtAVeigVector.initialize(subspaceSize,subspaceSize);
     VtAVeigValue.resize(subspaceSize,0.0);
    }
    else if((foundSize + subspaceSize) >= vectorDimension)
    {
     subspaceIncrementSize = vectorDimension - (foundSize + bufferSize);
     subspaceSize          = subspaceIncrementSize + bufferSize;
     VtAV.initialize(subspaceSize,subspaceSize);
     VtAVeigVector.initialize(subspaceSize,subspaceSize);
     VtAVeigValue.resize(subspaceSize,0.0);
    }


    for(k = 0; k < bufferSize; k++)
    {
        vArray[k] = vArrayTmp[k];
    }

    applyCountCumulative  += applyCount;
    }

#ifdef _TIMING_
    orthoTimeCumulative   += orthoTime;
    applyTimeCumulative   += applyTime;
    eigTimeCumulative     += eigTime;

    orthoCountCumulative  += orthoCount;
    eigCountCumulative    += eigCount;
#endif


#ifdef _TIMING_
    if(eigDiagnosticsFlag == 1)
    {
        printf("OrthoTime_Sec : %10.5f \n",orthoTime);
        printf("ApplyTime_Sec : %10.5f \n",applyTime);
        printf("EigTime_Sec   : %10.5f \n",eigTime);
    }
#endif 

   }// end of main loop
//
// Validate that the eigenvectors computed are associated with a monotonically increasing
// sequence of eigenvalues. If not, this problem is corrected by recomputing the eigenvalues
// and eigenvectors of the projected operator.
//
// This task is typically only required when when the desired eigenvalues are associated
// with multiple clusters where
// (a) the eigenvalues in the clusters are nearly degenerate
// (b) the subspaceIncrement size is insufficient to contain a complete collection of
//     clustered states, e.g. when the approximating subspace "splits" a cluster.
//
// This task will never be required if the subspaceIncrementSize > number of eigenvalues
// that are found.
//
//
    bool nonMonotoneFlag = false;

    for(long i = 0; i < foundSize-1; i++)
    {
    if(eigValues[i] > eigValues[i+1]) {nonMonotoneFlag = true;}
    }

    if(nonMonotoneFlag)
    {
    if(foundSize > subspaceSize)
    {
    vArrayTmp.resize(foundSize);
    for(long k = 0; k < foundSize; k++)
    {
    vArrayTmp[k].initialize(vStart);
    }
    }

    VtAV.initialize(foundSize,foundSize);
    VtAVeigVector.initialize(foundSize,foundSize);
    VtAVeigValue.resize(foundSize,0.0);

    VtAVdataPtr           = VtAV.getDataPointer();
    VtAVeigValueDataPtr   = &VtAVeigValue[0];
    VtAVeigVectorDataPtr  = VtAVeigVector.getDataPointer();

    #ifdef _TIMING_
    timer.start();
#endif

//  Form Ut*A*U

    for(i = 0; i < foundSize; i++)
    {
        vTemp     = eigVectors[i];
        oP.apply(vTemp);
#ifdef _OPENMP
		#pragma omp parallel for \
		private(j) \
		schedule(static,1)
#endif
        for(j = i; j < foundSize; j++)
        {
        VtAV(j,i) = eigVectors[j].dot(vTemp);
        VtAV(i,j) = VtAV(j,i);
        }
    }

    jacobiMethod.getEigenSystem(VtAVdataPtr, foundSize,
    VtAVeigValueDataPtr, VtAVeigVectorDataPtr);

#ifndef _VBLAS_
    //
    // Create eigenvectors
    //
    for(k = 0; k < foundSize; k++)
    {
        vArrayTmp[k]  = eigVectors[0];
        vArrayTmp[k] *= VtAVeigVector(0,k);
        for(i = 1; i < foundSize; i++)
        {
        vTemp =  eigVectors[i];
        vTemp *= VtAVeigVector(i,k);
        vArrayTmp[k] += vTemp;
        }

        rkk = vArrayTmp[k].dot(vArrayTmp[k]);
        vArrayTmp[k] *= 1.0/rkk;
    }
#endif
#ifdef _VBLAS_

    //
    // Create eigenvectors
    //
#ifdef _OPENMP
		#pragma omp parallel for \
		private(i,k,rkk) \
		schedule(static,1)
#endif
    for(k = 0; k < foundSize; k++)
    {
        vArrayTmp[k] = eigVectors[0];
        vArrayTmp[k].scal(VtAVeigVector(0,k));


        for(i = 1; i < foundSize; i++)
        {
        vArrayTmp[k].axpy(VtAVeigVector(i,k),eigVectors[i]);
        }

        rkk = vArrayTmp[k].nrm2();
        vArrayTmp[k].scal(1.0/rkk);
    }
#endif
        for(i = 0; i < foundSize; i++)
        {
        eigVectors[i] = vArrayTmp[foundSize - i - 1];
        eigValues[i]  = VtAVeigValue[foundSize - i - 1];
        }

#ifdef _TIMING_
    timer.stop();
    eigTimeCumulative  += timer.getSecElapsedTime();
#endif
    } // End non-monotone eigensystem correction


//
//   In the case of computing all eigenvalues less than a specified bound,
//   trim the resulting vectors to those that are within tolerance of
//   being less than lambdaMax
//   
//
	long finalFoundCount = 0;
  
    if(intervalStopConditionFlag)
    {
    relErrFactor = getRelErrorFactor(lambdaMax,subspaceTol);
    for(i = 0; i < (long)eigValues.size(); i++)
    {
    if((eigValues[i] - lambdaMax)/relErrFactor < subspaceTol) {finalFoundCount++;}
    }

    if(finalFoundCount < (long)eigValues.size())
    {
    eigValues.resize(finalFoundCount);
    eigVectors.resize(finalFoundCount);
    foundSize = finalFoundCount;
    }
    }
    

#ifdef _TIMING_
    if(eigDiagnosticsFlag == 1)
    {
    printf("Cumulative_OrthoTime_Sec  : %10.5f \n",orthoTimeCumulative);
    printf("Cumulative_ApplyTime_Sec  : %10.5f \n",applyTimeCumulative);
    printf("Cumulative_EigTime_Sec    : %10.5f \n",eigTimeCumulative);
#ifdef _OPENMP
    printf("XXX --- Using OpenMP Constructs  --- XXX\n");
#endif

    }
#endif 


    if(fixedIterationCount > 0)
    {this->resetMaxInnerLoopCount();}


    if(foundSize >= 0) returnFlag =  foundSize;
    return returnFlag;
    }

// Internal utility routines and class data

    void expandArray(std::vector<double>& v, long expandSize)
    {
    long origSize = (long)v.size();
    v.resize(origSize+expandSize,0.0);
    }

    void expandVector(std::vector< Vtype >& v, long expandSize)
    {
    long origSize = (long)v.size();
    v.resize(origSize+expandSize);
    }

//
//  OrthogonalizeAtoB projects the subspace defined by the
//  specified Avectors to be orthogonal to the subspace defined by the
//  specified Bvectors. After orthogonalization, the Avectors are normalized
//  to unit length.
//
//  This routine assumes that vTemp has been initialized
//
    void OrthogonalizeAtoB(std::vector< Vtype >& Avectors, long indexA_start, long indexA_end,
    std::vector< Vtype >&  Bvectors, long indexB_start, long indexB_end)
    {
    double rkj;
    double rkk;
    long j; long k;
    //
    //  Orthogonalize vArray to eigVectors (found eigenvectors)
    //
#ifndef _VBLAS_
    for(j = indexB_start; j <= indexB_end ; j++)
    {
    for(k = indexA_start; k <= indexA_end; k++)
    {
        vTemp           =   Bvectors[j];
        rkj             =   Avectors[k].dot(Bvectors[j]);
        vTemp          *=  -rkj;
        Avectors[k]    +=   vTemp;
    }
    }
    for(k = indexA_start; k <= indexA_end; k++)
    {
    rkk           = std::sqrt(Avectors[k].dot(Avectors[k]));
    Avectors[k]  *= 1.0/rkk;
    }
#endif
#ifdef _VBLAS_
for(j = indexB_start; j <= indexB_end; j++)
{
#ifdef _OPENMP
	   #pragma omp parallel for \
	   private(k,rkj) \
	   schedule(static,1)
#endif
       for(k = indexA_start; k <= indexA_end; k++)
       {
        rkj  =   Avectors[k].dot(Bvectors[j]);
        Avectors[k].axpy(-rkj,Bvectors[j]);
       }
}
#ifdef _OPENMP
#pragma omp parallel for \
private(k,rkk) \
schedule(static,1)
#endif
for(k = indexA_start; k <= indexA_end; k++)
{
    rkk  =   Avectors[k].nrm2();
    Avectors[k].scal(1.0/rkk);
}
#endif
}

//
//  This routine assumes that vTemp has been initialized
//

void orthogonalizeVarray(long subspaceSize)
{
	double rkk;
	double rkj;

#ifndef _VBLAS_

//  Orthogonalize the subspace vectors using Modified Gram-Schmidt

    for(long k = 1; k <= subspaceSize; k++)
    {
        rkk     = std::sqrt(vArray[k-1].dot(vArray[k-1]));
        vArray[k-1] *= 1.0/rkk;
        for(long j = k+1; j <= subspaceSize; j++)
        {
            rkj           =   vArray[k-1].dot(vArray[j-1]);
            vTemp         =   vArray[k-1];
            vTemp        *=  -rkj;
            vArray[j-1]  += vTemp;
        }
    }
#endif
#ifdef _VBLAS_

    long j;

//  Orthogonalize the subspace vectors using modified Gram-Schmidt

    for(long k = 1; k <= subspaceSize; k++)
    {
        rkk     = vArray[k-1].nrm2();
        vArray[k-1].scal(1.0/rkk);
#ifdef _OPENMP
		#pragma omp parallel for \
		private(j,rkj) \
		schedule(static,1)
#endif
        for(j = k+1; j <= subspaceSize; j++)
        {
            rkj  =   vArray[j-1].dot(vArray[k-1]);
            vArray[j-1].axpy(-rkj,vArray[k-1]);
        }
    }
#endif

}
    bool verboseSubspaceFlag;
    bool verboseFlag;
    bool eigDiagnosticsFlag;

    std::vector< Vtype >        vArray;
    std::vector< Vtype >     vArrayTmp;

    std::vector< Vtype >* orthogSubspacePtr; // A pointer to an array of vectors defining
                                        // a subspace to which the eigensystem
                                        // determination will be carried out
                                        // orthogonal to.

    Vtype vTemp;

    RC_Double2Darray       VtAV;
    RC_Double2Darray       VtAVeigVector;
    std::vector<double>    VtAVeigValue;

    double* VtAVdataPtr;
    double* VtAVeigValueDataPtr;
    double* VtAVeigVectorDataPtr;

    long minIntervalPolyDegreeMax;

    LanczosMaxMinFinder < Vtype, Otype, VRandomizeOpType > lanczosMaxMinFinder;
    LanczosCpoly cPoly;
    LanczosCpolyOperator< Vtype , Otype > cOp;

    JacobiDiagonalizer jacobiMethod;

    double    guardValue;                // Value of the guard eigenvalue.
    bool      intervalStopConditionFlag; // Converge based on value of guard eigenvalue
    bool      hardIntervalStopFlag;
    double    minEigValueEst;
    double    maxEigValueEst;

    long    maxInnerLoopCount;
    bool   nonRandomStartFlag;
    long  fixedIterationCount;
};

#undef   JACOBI_TOL
#undef   DEFAULT_MAX_INNER_LOOP_COUNT
#undef   RAYLEIGH_CHEBYSHEV_SMALL_TOL_

#endif

 
