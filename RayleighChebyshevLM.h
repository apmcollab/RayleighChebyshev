/**
                         RayleighChebyshevLM.h

   
   A templated class with member functions for computing eigenpairs 
   corresponding to the lowest eigenvalues of a linear operator.

   The routine is designed for both real symmetric and
   complex Hermitian operators.

   The eigenvalues are returned in a std::vector<double> instance
   while the eigenvectors are internally allocated and returned in
   a std::vector<Vtype> class instance.

   To use with complex Hermitian operators the template parameter
   specification

   typename Dtype  =  std::complex<double>.

   must be used. For complex Hermitian operators it is assumed that
   the inner product of the vector type Vtype is the
   complex inner product.

   OpenMP multi-thread usage is enabled by defining _OPENMP

   Note: _OPENMP is automatically defined if -fopenmp is specified 
   as part of the compilation command.

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
   long getDimension() const          (returns dimension of the Vtype subspace)

   if VBLAS_ is defined, then the Vtype class must also possess member functions

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
	used by the RayleighChebyshevLM template.

	###########################################################################

	When specifying a std::vector class to be used with a RayleighChebyshevLM instance, it is critical
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

   External dependencies: LAPACK
 
   Reference:

   Christopher R. Anderson, "A Rayleigh-Chebyshev procedure for finding
   the smallest eigenvalues and associated eigenvectors of large sparse
   Hermitian matrices" Journal of Computational Physics,
   Volume 229 Issue 19, September, 2010.


   Author Chris Anderson July 12, 2005
   Version : May 24, 2023
*/
/*
#############################################################################
#
# Copyright 2005-2023 Chris Anderson
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
#include <map>
#include <algorithm>

#include "LapackInterface/SCC_LapackMatrix.h"
#include "LapackInterface/SCC_LapackMatrixRoutines.h"

#include "LapackInterface/SCC_LapackMatrixCmplx16.h"
#include "LapackInterface/SCC_LapackMatrixRoutinesCmplx16.h"


#include "RCarray2d.h"
#include "RC_Types.h"

#include "LanczosCpoly.h"             // Chebyshev polynomial based filter polynomial
#include "LanczosCpolyOperatorLM.h"   // Chebyshev polynomial based filter polynomial operator
#include "LanczosMaxMinFinder.h"

#include "JacobiDiagonalizer.h"

#ifndef RAYLEIGH_CHEBYSHEV_LM_
#define RAYLEIGH_CHEBYSHEV_LM_

#define DEFAULT_MAX_MIN_TOL                1.0e-06
#define JACOBI_TOL                         1.0e-12
#define DEFAULT_MAX_INNER_LOOP_COUNT         10000
#define DEFAULT_POLY_DEGREE_MAX                200
#define DEFAULT_FILTER_REPETITION_COUNT          1
#define DEFAULT_USE_JACOBI_FLAG              false
#define RAYLEIGH_CHEBYSHEV_SMALL_TOL_      1.0e-11

#ifdef TIMING_
#include "ClockIt.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif


template <class Vtype, class Otype, class VRandomizeOpType, typename Dtype = double >
class RayleighChebyshevLM
{
    public : 

    RayleighChebyshevLM()
    {
    initialize();
    }

    void initialize()
    {
    OpPtr                      = nullptr;
    verboseFlag                = false;
    eigDiagnosticsFlag         = false;
	verboseSubspaceFlag        = false;
    jacobiMethod.tol           = JACOBI_TOL;
    useJacobiFlag              = DEFAULT_USE_JACOBI_FLAG;
    minIntervalPolyDegreeMax   = DEFAULT_POLY_DEGREE_MAX;
    filterRepetitionCount      = DEFAULT_FILTER_REPETITION_COUNT;
    minEigValueEst             = 0.0;
    maxEigValueEst             = 0.0;
    guardValue                 = 0.0;
    intervalStopConditionFlag  = false;
    hardIntervalStopFlag       = false;
    stopCondition              = RC_Types::StopCondition::COMBINATION;
    maxMinTol                  = DEFAULT_MAX_MIN_TOL;

    nonRandomStartFlag         = false;
    fixedIterationCount        = false;
    maxInnerLoopCount          = DEFAULT_MAX_INNER_LOOP_COUNT;

    eigVecResiduals.clear();

    finalData.clear();
    countData.clear();

    #ifdef TIMING_
    timeValue.clear();
    #endif

    #ifdef _OPENMP
    MtVarray.clear();
	#endif
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


	void setStopCondition(std::string stopConditionStr)
	{
		std::transform(stopConditionStr.begin(), stopConditionStr.end(), stopConditionStr.begin(),[](unsigned char c)
		{return  static_cast<char>(std::toupper(c));});

		if(stopConditionStr == "COMBINATION")
		{
         setStopCondition(RC_Types::StopCondition::COMBINATION);
		}
		else if(stopConditionStr == "EIGENVALUE_ONLY")
		{
         setStopCondition(RC_Types::StopCondition::EIGENVALUE_ONLY);
		}
	    else if(stopConditionStr == "RESIDUAL_ONLY")
		{
         setStopCondition(RC_Types::StopCondition::RESIDUAL_ONLY);
		}
	    else if(stopConditionStr == "DEFAULT")
		{
         setStopCondition(RC_Types::StopCondition::COMBINATION);
		}
	    else
	    {
	    std::string errMsg  = "\nRayleighChebyshev Error : Stopping condition type specified not";
	                errMsg +=  "\none of DEFAULT, COMBINATION, EIGENVALUE_ONLY,or RESIDUAL_ONLY.";
	                errMsg +=  "\nOffending specification : " + stopConditionStr + "\n";
	    throw std::runtime_error(errMsg);
	    }
	}

    void setStopCondition(RC_Types::StopCondition stopCondition)
	{
		switch (stopCondition)
		{
			case RC_Types::StopCondition::COMBINATION :
			{
				this->stopCondition = RC_Types::StopCondition::COMBINATION;
			} break;
			case RC_Types::StopCondition::EIGENVALUE_ONLY :
			{
				this->stopCondition = RC_Types::StopCondition::EIGENVALUE_ONLY;
			} break;

		    case RC_Types::StopCondition::RESIDUAL_ONLY :
			{
				this->stopCondition = RC_Types::StopCondition::RESIDUAL_ONLY;
			} break;

		    case RC_Types::StopCondition::DEFAULT:
			{
				this->stopCondition = RC_Types::StopCondition::COMBINATION;
			} break;

		    default :
		   	{
		   		this->stopCondition = RC_Types::StopCondition::COMBINATION;
			}
		}
	}

    void setMaxMinTolerance(double val = DEFAULT_MAX_MIN_TOL)
    {
    maxMinTol = val;
    }
    
    void setMaxInnerLoopCount(long val)
    {maxInnerLoopCount  = val;}

    void resetMaxInnerLoopCount()
    {maxInnerLoopCount  = DEFAULT_MAX_INNER_LOOP_COUNT;}

    void setFixedIterationCount(bool val)
    {fixedIterationCount  = val;}

    void clearFixedIteratonCount()
    {fixedIterationCount  = false;}

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
    
    void setFilterRepetitionCount(long repetitionCount)
    {
    	filterRepetitionCount = repetitionCount;
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

    void setMaxSpectralBoundsIter(long iterCount)
    {
    	lanczosMaxMinFinder.setIterationMax(iterCount);
    }

    long getSpectralBoundsIterCount()
    {
    	return lanczosMaxMinFinder.getIterationCount();
    }

    void setUseJacobi(bool val)
    {
    	useJacobiFlag = val;
    }

    void clearUseJacobi()
    {
    	useJacobiFlag = false;
    }

    std::vector<double> getEigVectorResiduals() const
    {
    	return  eigVecResiduals;
    }

    std::map<std::string,double> getFinalData() const
    {
    	return finalData;
    }

    std::map<std::string,long> getCountData() const
    {
    	return countData;
    }

#ifdef TIMING_
    std::map<std::string,double> getTimingData() const
    {
    	return timeValue;
    }
#endif

//
//  Member functions called to return eigensystem of operator projected onto
//  working subspace.
//
    void computeVtVeigensystem(RCarray2d<double>& VtAV, std::vector<double>& VtAVeigValue,
    RCarray2d<double>& VtAVeigVector)
    {
    	long rowSize = VtAV.getRowSize();
    	long colSize = VtAV.getColSize();

    	if(useJacobiFlag)
    	{
    		jacobiMethod.setSortIncreasing(true);
    		jacobiMethod.setIOdataRowStorage(false);
    		jacobiMethod.getEigenSystem(VtAV.getDataPointer(), rowSize, &VtAVeigValue[0], VtAVeigVector.getDataPointer());
    	}
        else
        {

    	/////////////////////////////////////////////////////////////////////////////
    	//     Calculation using LAPACK
    	////////////////////////////////////////////////////////////////////////////

    	SCC::LapackMatrix VtAVmatrix;
    	SCC::LapackMatrix VtAVeigVectorMatrix;

    	VtAVmatrix.initialize(rowSize, colSize);
    	VtAVeigVectorMatrix.initialize(rowSize,colSize);

    	for(long i = 0; i < rowSize; i++)
    	{
    	for(long j = 0; j < colSize; j++)
    	{
    	VtAVmatrix(i,j) = VtAV(i,j);
    	VtAVeigVectorMatrix(i,j) = VtAVeigVector(i,j);
    	}}

        dsyev.computeEigensystem(VtAVmatrix, VtAVeigValue, VtAVeigVectorMatrix);

        for(long i = 0; i < rowSize; i++)
    	{
    	for(long j = 0; j < colSize; j++)
    	{
    	VtAVeigVectorMatrix(i,j) = VtAVeigVector(i,j) = VtAVeigVectorMatrix(i,j);
    	}}
    	}
    }

    void computeVtVeigensystem(RCarray2d< std::complex<double> > & VtAV, std::vector<double>& VtAVeigValue,
    RCarray2d<  std::complex<double>  >& VtAVeigVector)
    {
    	long rowSize = VtAV.getRowSize();
    	long colSize = VtAV.getColSize();

    	/////////////////////////////////////////////////////////////////////////////
    	//     Calculation using LAPACK
    	/////////////////////////////////////////////////////////////////////////////

    	SCC::LapackMatrixCmplx16  VtAVmatrix(rowSize,colSize);
    	SCC::LapackMatrixCmplx16  VtAVeigVectorMatrix(rowSize,colSize);

    	for(long i = 0; i < rowSize; i++)
    	{
    	for(long j = 0; j < colSize; j++)
    	{
    	VtAVmatrix(i,j) = VtAV(i,j);
    	VtAVeigVectorMatrix(i,j) = VtAVeigVector(i,j);
    	}}

    	zhpevx.createEigensystem(VtAVmatrix, VtAVeigValue,VtAVeigVectorMatrix);

    	for(long i = 0; i < rowSize; i++)
    	{
    	for(long j = 0; j < colSize; j++)
    	{
    	VtAVeigVectorMatrix(i,j) = VtAVeigVector(i,j) = VtAVeigVectorMatrix(i,j);
    	}}

    	/////////////////////////////////////////////////////////////////////////////
    	//     Calculation using LAPACK
    	/////////////////////////////////////////////////////////////////////////////

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
    VRandomizeOpType& randOp, double&  minEigValue, double& maxEigValue)
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
    this->setHardIntervalStop();

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

    double minFinderTol  = maxMinTol;

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
    double minFinderTol  = maxMinTol;
    
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

    OpPtr = &oP; // Pointer to input operator for use by supporting member functions

#ifdef _OPENMP
     int threadCount = omp_get_max_threads();
#endif


    this->setupCountData();
    this->setupFinalData();

    /////////////////////////////////////////////////////////////////////
    // Compile with _TIMING defined for capturing timing data
    // otherwise timeing routines are noOps
    /////////////////////////////////////////////////////////////////////

    this->setupTimeData();
    this->startGlobalTimer();

    std::vector<double> residualHistory;

    // Insure that subspaceTol isn't too small

    if(subspaceTol < RAYLEIGH_CHEBYSHEV_SMALL_TOL_ ) {subspaceTol = RAYLEIGH_CHEBYSHEV_SMALL_TOL_; }
    double relErrFactor;
    //
    // Delete any old eigenvalues and eigenvectors if not random start, otherwise
    // use input eigenvectors for as much of the initial subspace as possible

    eigValues.clear();
    eigVecResiduals.clear();

    if(not nonRandomStartFlag)
    {
    eigVectors.clear();
    }

    long returnFlag = 0;

    double   lambdaStar;
    long   subspaceSize;   
    long      foundSize;
    long     foundCount;

    bool  completedBasisFlag = false;

    std::vector<double>           oldEigs;
    std::vector<double>          eigDiffs;
    std::vector<double>       oldEigDiffs;
    std::vector<double> subspaceResiduals;

    double            eigDiff;

    lambdaStar        = maxEigValue;
    subspaceSize      = subspaceIncrementSize +  bufferSize;
    foundSize         = 0;

    //
    // Reset sizes if subspaceSize is larger
    // than dimension of system

    long vectorDimension = vStart.getDimension();

    if(subspaceSize > vectorDimension)
    {
    	if(subspaceIncrementSize < vectorDimension)
    	{
    		bufferSize  = vectorDimension - subspaceIncrementSize;
    		subspaceSize = vectorDimension;
    	}
    	else
    	{
    		subspaceSize          = vectorDimension;
    		subspaceIncrementSize = vectorDimension;
    		bufferSize            = 0;
    	}

    	maxEigensystemDim     = vectorDimension;
    }

    oldEigs.resize(subspaceSize,0.0);
    eigDiffs.resize(subspaceSize,1.0);
    oldEigDiffs.resize(subspaceSize,1.0);

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
    VtAVeigValue.resize(subspaceSize,0.0);

    VtAV.initialize(subspaceSize,subspaceSize);
    VtAVeigVector.initialize(subspaceSize,subspaceSize);

    long   starDegree     = 0;
    long   starDegreeSave = 0;
    double starBoundSave  = 0.0;
    double shift          = 0.0;
    double starBound      = 0.0;
    double maxEigDiff     = 0.0;
    double maxResidual    = 0.0;
    double maxGap         = 0.0;
    double stopCheckValue = 0.0;
    double eigDiffRatio   = 0.0;

    long residualCheckCount = 0;
    long   innerLoopCount   = 0;

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
    	for(long k = 0; k < subspaceSize; k++)
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
    		for(long k = 0; k < (long)eigVectors.size(); k++)
    		{
    			vArray[k].initialize(eigVectors[k]);
    			vArrayTmp[k].initialize(vStart);
    		}
    		for(long k = (long)eigVectors.size(); k < subspaceSize; k++)
    		{
    		vArray[k].initialize(vStart);
    		randOp.randomize(vArray[k]);
    		vArrayTmp[k].initialize(vStart);
    		}
    	}
    	else
    	{
    		for(long k = 0; k <  subspaceSize; k++)
    		{
    			vArray[k].initialize(eigVectors[k]);
    			vArrayTmp[k].initialize(vStart);
    		}
    	}

    	// Clear the pre-existing eigenvectors

    	eigVectors.clear();
    }


    // Initialize temporaries

    vTemp.initialize(vStart);

    #ifdef _OPENMP
    MtVarray.clear();
    MtVarray.resize(threadCount);

    for(long k = 0; k < threadCount; k++)
    {
    	MtVarray[k].initialize(vStart);
    }
    #endif


    // Quick return if subspaceSize >= vector dimension

    if(vectorDimension == subspaceSize)
    {
    long maxOrthoCheck   = 10;
    long orthoCheckCount = 1;
    orthogonalize(vArray);

    // Due to instability of modified Gram-Schmidt for creating an
    // orthonormal basis for a high dimensional vector space, multiple
    // orthogonalization passes may be needed.

    while((OrthogonalityCheck(vArray, false) > 1.0e-12)&&(orthoCheckCount <= maxOrthoCheck))
    {
    	orthoCheckCount += 1;
    	orthogonalize(vArray);
    }

    if(orthoCheckCount > maxOrthoCheck)
    {
    	std::string errMsg = "\nXXXX RayleighChebyshev Error XXXX";
    	errMsg +=            "\nUnable to create basis for complete vector space.\n";
    	errMsg +=            "\nReduce size of buffer and/or subspaceIncrement \n";
    	throw std::runtime_error(errMsg);
    }

    formVtAV(vArray, VtAV);
    computeVtVeigensystem(VtAV, VtAVeigValue, VtAVeigVector);
    createEigenVectorsAndResiduals(VtAVeigVector,vArray,subspaceSize,eigVecResiduals);
    eigVectors.resize(subspaceSize,vStart);
    eigVectors = vArray;
    eigValues  = VtAVeigValue;
    return subspaceSize;
    }


    // Initialize filter polynomial operators

    cOp.initialize(oP);


//
//  ################## Main Loop ######################
//
    if(fixedIterationCount)
    {
    maxInnerLoopCount  = fixedIterationCount;
    }

    int  exitFlag = 0;

    long   applyCount            = 0;
    long   applyCountCumulative  = 0;

//
////////////////////////////////////////////////////////////
//                Main loop
////////////////////////////////////////////////////////////
//
    while(exitFlag == 0)
    {
//
//  Initialize old eigenvalue array using buffer values.
//  This step is done for cases when the routine
//  is called to continue an existing computation.
//
    for(long k = bufferSize; k < subspaceSize; k++)
    {
         oldEigs[k] = oldEigs[bufferSize];
    }
//
//  Randomize buffer vectors after first increment.
//
    if(applyCountCumulative > 0)
    {
    for(long k = bufferSize; k < subspaceSize; k++)
    {
        randOp.randomize(vArray[k]);
    }
    }


    startTimer();

    // Orthogonalize working subspace (vArray)
    // Repeat orthogonalization initially to compensate for possible
    // inaccuracies using modified-Gram Schmidt. During iteration,
    // subspace orthogonality is preserved due to use of eigenvector
    // basis of projected operator.
    //

    orthogonalize(vArray);
    orthogonalize(vArray);

    incrementTime("ortho");
    incrementCount("ortho",2);


    lambdaStar     = maxEigValue;
    eigDiffRatio   = 1.0;
    innerLoopCount = 0;
    starDegreeSave = 0;
    starBoundSave  = 0.0;

    double eMin; double eMax;

    stopCheckValue = subspaceTol + 1.0e10;

    applyCount = 0;
    residualHistory.clear();

    maxGap      = 0.0;
    maxResidual = 0.0;
    maxEigDiff  = 0.0;

    long oscillationCount = 0;

    while((stopCheckValue  > subspaceTol)&&(innerLoopCount < maxInnerLoopCount))
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
    }

    ////////ZZZZZZZZZZZZZZZZZZZZZZZ
    // For multiplicity > subspace dimension
    else if((not (maxEigDiff > subspaceTol))&&(innerLoopCount > 3)&&(eigDiffRatio < .2)) // .2 is slightly less than the secondary
    {                                                                                     // maximum of the Lanczos C polynmoial
    	  starDegree = starDegreeSave;
    	  starBound  = starBoundSave;
    }

//
/////////////////////////////////////////////////////////////////////////////
//         Determining filter polynomial parameters
/////////////////////////////////////////////////////////////////////////////
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

/////////////////////////////////////////////////////////////////////////////
//      Applying filter polynomial to working subspace (vArray)
/////////////////////////////////////////////////////////////////////////////

    cOp.setLanczosCpolyParameters(starDegree,filterRepetitionCount,starBound,shift);


    startTimer();

    if(not completedBasisFlag)
    {
    cOp.apply(vArray);
    }

    applyCount += 1;

    this->incrementTime("OpApply");
    this->incrementCount("OpApply", starDegree*subspaceSize);

/////////////////////////////////////////////////////////////////////////////
//                Orthogonalizing working subspace (vArray)
/////////////////////////////////////////////////////////////////////////////

    startTimer();

//  Orthogonalize working subspace (vArray) to subspace of found eigenvectors (eigVectors)
//  It is important to do this before orthgonalizing the new vectors with respect to each other.

    indexA_start = 0;
    indexA_end   = subspaceSize-1;
    indexB_start = 0;
    indexB_end   = foundSize-1;

    OrthogonalizeAtoB(vArray, indexA_start,indexA_end, eigVectors, indexB_start, indexB_end);

//  Orthogonalize the subspace vectors using Modified Gram-Schmidt

    orthogonalize(vArray);

    incrementTime("ortho");
    incrementCount("ortho");
//
//#############################################################################
// 			Forming projection of operator onto working subspace (VtAV)
//#############################################################################
//
    startTimer();

    formVtAV(vArray, VtAV);

    incrementCount("OpApply", subspaceSize);

/////////////////////////////////////////////////////////////////////////////
//         Compute eigenvalues of  Vt*A*V
/////////////////////////////////////////////////////////////////////////////

    computeVtVeigensystem(VtAV, VtAVeigValue, VtAVeigVector);

/////////////////////////////////////////////////////////////////////////////
// Compute new approximations to eigenvectors and evaluate selected residuals
/////////////////////////////////////////////////////////////////////////////

    // Only check residuals of subspace eigenvectors for
    // the eigenvectors one is determining in this subspace,
    // e.g. do not check residuals of buffer vectors.

    if(foundSize + subspaceSize < vectorDimension)
    {residualCheckCount = subspaceIncrementSize;}
    else
    {residualCheckCount = subspaceSize;}

    subspaceResiduals.clear();

    createEigenVectorsAndResiduals(VtAVeigVector,vArray,residualCheckCount,subspaceResiduals);

    incrementCount("OpApply", residualCheckCount);

    maxResidual = 0.0;
    for(size_t k = 0; k < subspaceResiduals.size(); k++)
    {
    relErrFactor = getRelErrorFactor(VtAVeigValue[k],subspaceTol);
    maxResidual  = std::max(maxResidual,subspaceResiduals[k]/relErrFactor);
    }

    residualHistory.push_back(maxResidual);

    incrementTime("eigenvalue");
    incrementCount("eigenvalue");
/////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////

    if(verboseSubspaceFlag)
    {
    printf("XXXX Subspace Eigs XXXX \n");
    for(long i = 0; i < subspaceSize; i++)
    {
    printf("%3ld : %+10.5e \n",i,VtAVeigValue[i]);
    }
    printf("\n");
    printf("Shift      : %10.5e MaxEigValue : %10.5e \n ",shift,maxEigValue);
    printf("LambdaStar : %10.5e StarBound   : %10.5e StarDegree : %3ld \n",lambdaStar,starBound,starDegree); 
    printf("XXXXXXXXXXXXXXXXXXXXXXX \n");
    }
 
//
//  Determining the subspace size to check for eigenvalue convergence.
//  Ignore the eigenvalues associated with the buffer vectors, except
//  buffer vector with smallest eigenvalue when determining eigenvalues
//  over an interval.
//

    long eigSubspaceCheckSize;

    if(intervalStopConditionFlag) {eigSubspaceCheckSize  = subspaceIncrementSize + 1;}
    else                          {eigSubspaceCheckSize  = subspaceIncrementSize;}

    for(long i = 0; i < eigSubspaceCheckSize; i++)
    {
    	oldEigDiffs[i] = eigDiffs[i];
    }

    maxEigDiff  = 0.0;

    for(long i = 0; i < eigSubspaceCheckSize; i++)
    {
    eigDiff = std::abs(VtAVeigValue[i] - oldEigs[i]);
    relErrFactor = getRelErrorFactor(oldEigs[i],subspaceTol);
	eigDiff = eigDiff/relErrFactor;
    eigDiffs[i] = eigDiff;
    maxEigDiff = (eigDiff > maxEigDiff)? eigDiff : maxEigDiff;
    }

    for(long i = 0; i < subspaceSize; i++)
    {
    oldEigs[i] = VtAVeigValue[i];
    }
    
    //
    // Compute an average estimated convergence rate based upon components
    // for which the convergence tolerance has not been achieved. 
    //

    long diffCount = 0;
    eigDiffRatio  = 0.0;
    if(maxEigDiff > subspaceTol)
    {
        for(long i = 0; i < eigSubspaceCheckSize; i++)
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

    double spectralRange = std::abs((lambdaMax-minEigValue));

    maxGap = 0.0;
    for(long i = 1; i < eigSubspaceCheckSize; i++)
    {
    maxGap = std::max(maxGap,std::abs(VtAVeigValue[i]-VtAVeigValue[i-1])/spectralRange);
    }

    if(verboseFlag == 1)
    {
    printf("%-5ld : Degree %-3ld  Residual Max: %-10.5g  Eig Diff Max: %-10.5g  Eig Conv Factor: %-10.5g Max Gap %-10.5g \n",
    innerLoopCount,starDegree,maxResidual,maxEigDiff,eigDiffRatio,maxGap);
    }


    // Create value to determine when iteration should terminate


    if(stopCondition == RC_Types::StopCondition::RESIDUAL_ONLY)
    {
    	stopCheckValue = maxResidual;
    }
    else if(stopCondition == RC_Types::StopCondition::EIGENVALUE_ONLY)
    {
    	stopCheckValue = maxEigDiff;
    }
    else // Stop based up convergence of eigenvalues and residuals  < sqrt(subspaceTol)
    {
    	stopCheckValue = maxEigDiff;

    	if(maxResidual > std::sqrt(subspaceTol))
    	{
    	stopCheckValue = maxResidual;
    	}
    }

    //
    // Force termination if we've filled out the subspace
    //

    if(subspaceIncrementSize == 0) {stopCheckValue = 0.0;}

    // When using residual stop tolearnce force termination
    // if residual is oscillating and has value < sqrt(subspaceTol)
    //
    long   rIndex; double residual2ndDiffA; double residual2ndDiffB;

    if((stopCondition == RC_Types::StopCondition::RESIDUAL_ONLY)&&(maxResidual < std::sqrt(subspaceTol)))
    {
    rIndex   = residualHistory.size() - 1;

    if(rIndex > 3)
    {
      residual2ndDiffA = (residualHistory[rIndex-3] - 2.0*residualHistory[rIndex-2] + residualHistory[rIndex-1])/(std::abs(maxResidual));
      residual2ndDiffB = (residualHistory[rIndex-2] - 2.0*residualHistory[rIndex-1] + residualHistory[rIndex])  /(std::abs(maxResidual));
      if(residual2ndDiffA*residual2ndDiffB < 0.0)
      {
      oscillationCount += 1;
      if(oscillationCount > 5)
      {
      stopCheckValue = 0.0;
      if(verboseFlag)
      {
      std::cout << "Warning : Oscillatory residuals observed when max residual less than square root of subspace tolerance. " << std::endl;
      std::cout << "          Rayleigh-Chebyshev subspace iteration stopped before residual terminaion criterion met. " << std::endl;
      std::cout << "          Subspace tolerance specified :  " << subspaceTol << std::endl;
      std::cout << "          Resididual obtained          :  " << maxResidual << std::endl;
      std::cout << "Typical remediation involves either increasing subspace tolerance or buffer size." << std::endl;
      std::cout <<  std::endl;
      }
      }
      }
    }}

    //
    // Update cPoly parameters based upon the eigensystem computation.
    //
    //
    // lambdaStar  : is reset to the largest eigenvalue currently computed
    // minEigValue : is reset when the subspace computation yields a 
    // minimum eigenvalue smaller than minEigValue. 
    //

    lambdaStar     = VtAVeigValue[subspaceSize-1];
    minEigValue    = (minEigValue < VtAVeigValue[0]) ? minEigValue : VtAVeigValue[0];

    innerLoopCount++;
    }

    // Capture inner loop parameters

    finalData["maxResidual"]           = std::max(maxResidual,finalData["maxResidual"]);
	finalData["maxEigValueDifference"] = std::max(maxEigDiff,finalData["maxEigValueDifference"]);
	finalData["maxRelEigValueGap"]     = std::max(maxGap,finalData["maxRelEigValueGap"]);


    applyCountCumulative  += applyCount;

    if(verboseFlag == 1)
    {
    if((not fixedIterationCount) && (innerLoopCount >= maxInnerLoopCount))
    {
    printf(" Warning             : Maximal number of iterations taken before tolerance reached \n");
    printf(" Iterations taken    : %ld \n",innerLoopCount);
    printf(" Eig Diff Max        : %-10.5g \n",maxEigDiff);
    printf(" Residual Max        : %-10.5g \n",maxResidual);
    printf(" Requested Tolerance : %-10.5g \n",subspaceTol);
    }
    }

    foundCount = 0;
//
//  Capture the found eigenpairs
//
    long checkIndexCount;

    if(foundSize + subspaceSize < vectorDimension)
    {checkIndexCount = subspaceIncrementSize;}
    else 
    {checkIndexCount = subspaceSize;}

    // Check for eigenvalues less than maximal eigenvalue

    for(long i = 0; i < checkIndexCount; i++)
    {
    vtvEig  = VtAVeigValue[i];
    relErrFactor = getRelErrorFactor(lambdaMax,subspaceTol);

    vtvEigCheck = (vtvEig - lambdaMax)/relErrFactor; // Signed value is important here

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

    // Capture found eigenvalues and eigenvectors

    if(foundCount > 0)
    {
    	eigVectors.resize(foundSize+foundCount,vStart);
    	eigValues.resize(foundSize+foundCount,0.0);

        for(long i = 0; i < foundCount; i++)
        {
        eigVectors[foundSize + i] =      vArray[i];
        eigValues[foundSize + i]  = VtAVeigValue[i];
        eigVecResiduals.push_back(subspaceResiduals[i]);
        }
    
        foundSize += foundCount; 
        if(verboseFlag == 1)
        {
        printf("Found Count: %3ld Largest Eig: %-9.5g Lambda Bound: %-9.5g \n",foundSize, eigValues[foundSize-1], lambdaMax);
        }
    }


    //
    // Shuffle all computed vectors to head of vArray
    //
    if(not exitFlag)
    {
    	for(long k = 0; k+foundCount < subspaceSize; k++)
    	{
    	vArray[k] = vArray[k+foundCount];
    	}
    }
//
//  See if "guard" eigenvalue is greater than lambdaMax. 
//
    if(bufferSize > 0)
    {
    	vtvEig     = VtAVeigValue[subspaceSize - bufferSize];
    }
    else
    {
    	vtvEig     = lambdaMax;
    }

    guardValue = vtvEig;
    relErrFactor = getRelErrorFactor(lambdaMax,subspaceTol);
    vtvEigCheck = (vtvEig - lambdaMax)/relErrFactor;

//
//  Using a hard interval stop when computing a fixed number of eigenpairs.
//
//  We assume that states are degenerate if |lambda[i] - lambda[i+1]|/relErrFactor < 10.0*subspaceTol
//  and hence require a relative gap of size 10.0*subspaceTol between lambdaMax and the guard vector
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

    minEigValue = eigValues[foundSize-1]; // New minEigValue = largest of found eigenvalues

    // Reset star degree and bound and addjust

    starDegree = 1;
    starBound = maxEigValue;
//
//  Check for exceeding vector space dimension
//  this step always reduces the subspace size, so the
//  the resize of the vArray does not alter the initial
//  elements of the vArray.
//
    if(not exitFlag)
    {
    if((foundSize + subspaceSize) >= vectorDimension)
    {
    	// The computational subspace fills out the dimension of the
    	// vector space so the last iteration will just be
    	// projection onto a collection of random vectors that
    	// are orthogonal to the current collection of eigenvectors.
    	//
    	if(foundSize + subspaceIncrementSize >= vectorDimension)
    	{
    		bufferSize            = 0;
    		subspaceIncrementSize = vectorDimension - foundSize;
    		subspaceSize          = subspaceIncrementSize;
    		completedBasisFlag    = true;
    		vArray.resize(subspaceSize);
    		for(long k = 0; k < subspaceSize; k++)
    		{
    		randOp.randomize(vArray[k]);
    		}
    	}
    	else
    	{
    		bufferSize = vectorDimension - (foundSize + subspaceIncrementSize);
    		subspaceSize = subspaceIncrementSize + bufferSize;
    		vArray.resize(subspaceSize);
    	}

    	vArrayTmp.resize(subspaceSize);

    	VtAV.initialize(subspaceSize,subspaceSize);
    	VtAVeigVector.initialize(subspaceSize,subspaceSize);
    	VtAVeigValue.resize(subspaceSize,0.0);
    }

    }





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
    foundSize = eigVectors.size();
    vArrayTmp.resize(foundSize,vStart);
    VtAV.initialize(foundSize,foundSize);
    VtAVeigVector.initialize(foundSize,foundSize);
    VtAVeigValue.resize(foundSize,0.0);

    startTimer();

/////////////////////////////////////////////////////////////////////////////
//     Form projection of operation on subspace of found eigenvectors
/////////////////////////////////////////////////////////////////////////////

    formVtAV(eigVectors, VtAV);

    incrementCount("OpApply",foundSize);

/////////////////////////////////////////////////////////////////////////////
//             Compute eigenvalues of  Vt*A*V
/////////////////////////////////////////////////////////////////////////////

    computeVtVeigensystem(VtAV, VtAVeigValue, VtAVeigVector);

/////////////////////////////////////////////////////////////////////////////
//                   Create eigenvectors
/////////////////////////////////////////////////////////////////////////////

    eigVecResiduals.clear();
    createEigenVectorsAndResiduals(VtAVeigVector,eigVectors,foundSize,eigVecResiduals);

    incrementCount("OpApply",foundSize);
    incrementTime("eigenvalue");
    incrementCount("eigenvalue");

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
    for(long i = 0; i < (long)eigValues.size(); i++)
    {
    if((eigValues[i] - lambdaMax)/relErrFactor < subspaceTol) {finalFoundCount++;}
    }

    if(finalFoundCount < (long)eigValues.size())
    {
    eigValues.resize(finalFoundCount);
    eigVectors.resize(finalFoundCount);
    eigVecResiduals.resize(finalFoundCount);
    foundSize = finalFoundCount;
    }
    }


    this->incrementTotalTime();
    if(eigDiagnosticsFlag == 1)
    {
    	printf("\nXXXX Rayleigh-Chebyshev Diagnostics XXXXX \n");

        #ifdef _OPENMP
        printf("XXXX --- Using OpenMP Constructs  --- XXXX\n");
		#endif

        printf("\n");
        printf("Total_Iterations        : %-ld   \n",applyCountCumulative);
        printf("Total_OpApply           : %-ld   \n",countData["OpApply"]);
        printf("Total_SubspaceEig       : %-ld   \n",countData["eigenvalue"]);
        printf("Total_Orthogonalization : %-ld   \n",countData["ortho"]);

    	#ifdef TIMING_
        printf("TotalTime_Sec : %10.5f \n",timeValue["totalTime"]);
        printf("OrthoTime_Sec : %10.5f \n",timeValue["ortho"]);
        printf("ApplyTime_Sec : %10.5f \n",timeValue["OpApply"]);
        printf("EigTime_Sec   : %10.5f \n",timeValue["eigenvalue"]);
        #endif

    }


    if(fixedIterationCount > 0)
    {this->resetMaxInnerLoopCount();}


    if(foundSize >= 0) returnFlag =  foundSize;
    return returnFlag;
    }

/////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////

void orthogonalize(std::vector< Vtype >& V)
{
	long subspaceSize = (long)V.size();
#ifndef VBLAS_
#ifdef _OPENMP
	int threadNum;
	for(long k = 1; k <= subspaceSize; k++)
    {
        auto rkk          = std::sqrt(std::abs(V[k-1].dot(V[k-1])));
        V[k-1] *= 1.0/rkk;

		#pragma omp parallel for \
		private(threadNum) \
		schedule(static,1)
        for(long j = k+1; j <= subspaceSize; j++)
        {
        	threadNum = omp_get_thread_num();
            auto rkj              =   V[j-1].dot(V[k-1]);
            MtVarray[threadNum]   =   V[k-1];
            MtVarray[threadNum]   *=  -rkj;
            V[j-1]           +=   MtVarray[threadNum];
        }
    }
#else
    for(long k = 1; k <= subspaceSize; k++)
    {
        auto rkk          = std::sqrt(std::abs(V[k-1].dot(V[k-1])));
        V[k-1] *= 1.0/rkk;
        for(long j = k+1; j <= subspaceSize; j++)
        {
            auto rkj      =   V[j-1].dot(V[k-1]);
            vTemp         =   V[k-1];
            vTemp        *=  -rkj;
            V[j-1]  += vTemp;
        }
    }
#endif
#endif

#ifdef VBLAS_
    for(long k = 1; k <= subspaceSize; k++)
    {
        auto rkk  = V[k-1].nrm2();
        V[k-1].scal(1.0/rkk);
#ifdef _OPENMP
		#pragma omp parallel for \
		schedule(static,1)
#endif
        for(long j = k+1; j <= subspaceSize; j++)
        {
            auto rkj  =   V[j-1].dot(V[k-1]);
            V[j-1].axpy(-rkj,V[k-1]);
        }
    }
#endif
}
//
//  OrthogonalizeAtoB projects the subspace defined by the
//  specified Avectors to be orthogonal to the subspace defined by the
//  specified Bvectors. After orthogonalization, the Avectors are normalized
//  to unit length.
//
void OrthogonalizeAtoB(std::vector< Vtype >& Avectors, long indexA_start, long indexA_end,
std::vector< Vtype >&  Bvectors, long indexB_start, long indexB_end)
{
#ifndef VBLAS_

#ifdef _OPENMP
    int threadNum;
    for(long j = indexB_start; j <= indexB_end ; j++)
    {
 	#pragma omp parallel for \
 	private(threadNum) \
	schedule(static,1)
    for(long k = indexA_start; k <= indexA_end; k++)
    {
    	threadNum = omp_get_thread_num();
        MtVarray[threadNum]  =   Bvectors[j];
        auto rkj             =   Avectors[k].dot(Bvectors[j]);
        MtVarray[threadNum] *=  -rkj;
        Avectors[k]         +=   MtVarray[threadNum];
    }
    }

	#pragma omp parallel for \
	schedule(static,1)
    for(long k = indexA_start; k <= indexA_end; k++)
    {
    auto rkk      = std::sqrt(std::abs(Avectors[k].dot(Avectors[k])));
    Avectors[k]  *= 1.0/rkk;
    }
#else

    for(long j = indexB_start; j <= indexB_end ; j++)
    {
    for(long k = indexA_start; k <= indexA_end; k++)
    {
        vTemp           =   Bvectors[j];
        auto rkj        =   Avectors[k].dot(Bvectors[j]);
        vTemp          *=  -rkj;
        Avectors[k]    +=   vTemp;
    }
    }
    for(long k = indexA_start; k <= indexA_end; k++)
    {
    auto rkk      = std::sqrt(std::abs(Avectors[k].dot(Avectors[k])));
    Avectors[k]  *= 1.0/rkk;
    }
#endif
#endif


#ifdef VBLAS_
for(long j = indexB_start; j <= indexB_end; j++)
{
#ifdef _OPENMP
 	   #pragma omp parallel for \
	   schedule(static,1)
#endif
       for(long k = indexA_start; k <= indexA_end; k++)
       {
        auto rkj  =   Avectors[k].dot(Bvectors[j]);
        Avectors[k].axpy(-rkj,Bvectors[j]);
       }
}
#ifdef _OPENMP
#pragma omp parallel for \
schedule(static,1)
#endif
for(long k = indexA_start; k <= indexA_end; k++)
{
    auto rkk  =   Avectors[k].nrm2();
    Avectors[k].scal(1.0/rkk);
}
#endif
}




// Assumes the operator is symmetric (or complex Hermitian).
//
// Only the upper triangular part of the projection of the
// operator is formed and the lower part is obtained by
// reflection.

void formVtAV(std::vector< Vtype >& V, RCarray2d<Dtype>& H)
{
	long subspaceSize = (long)V.size();

	for(size_t p = 0; p < vArray.size(); p++)
	{
    	vArrayTmp[p] = V[p];
	}

    OpPtr->apply(vArrayTmp);

#ifdef _OPENMP
#pragma omp parallel for \
schedule(static,1)
    for(long i = 0; i < subspaceSize; i++)
    {
        for(long j = i; j < subspaceSize; j++)
        {
        H(j,i) = V[j].dot(vArrayTmp[i]);
        }
    }
    for(long i = 0; i < subspaceSize; i++)
    {
    	for(long j = i+1; j < subspaceSize; j++)
    	{
    	     H(i,j) = H(j,i);
    	}
    }

#else
    for(long i = 0; i < subspaceSize; i++)
    {
        vTemp     = V[i];
        OpPtr->apply(vTemp);
        for(long j = i; j < subspaceSize; j++)
        {
        VtAV(j,i) = V[j].dot(vTemp);
        VtAV(i,j) = VtAV(j,i);
        }
    }
#endif
}

//
// Input
//
// V              : the collection of vectors in the subspace
// VtAVeigVector  : the eigenvectors of the projected operator
//
// Output
// V              : Approximate eigenvectors of the operator based
//                  upon the eigenvectors of the projected operator
//
// eigVresiduals  : Residuals of the first residualCheckCount approximate eigenvectors
//                  ordered by eigenvalues, algebraically smallest to largest.
//
void createEigenVectorsAndResiduals(RCarray2d<Dtype>& VtAVeigVector, std::vector< Vtype >& V,
long residualCheckCount, std::vector<double>& eigVresiduals)
{
	long subspaceSize = (long) V.size();

#ifdef _OPENMP
	int threadNum;
#endif

#ifndef VBLAS_
#ifdef _OPENMP
    #pragma omp parallel for \
	private(threadNum) \
	schedule(static,1)
    for(long k = 0; k < subspaceSize; k++)
    {
    	threadNum = omp_get_thread_num();
        vArrayTmp[k]  = V[0];
        vArrayTmp[k] *= VtAVeigVector(0,k);
        for(long i = 1; i < subspaceSize; i++)
        {
        MtVarray[threadNum]  = V[i];
        MtVarray[threadNum] *= VtAVeigVector(i,k);
        vArrayTmp[k]        += MtVarray[threadNum];
        }

        auto rkk = std::sqrt(std::abs(vArrayTmp[k].dot(vArrayTmp[k])));
        vArrayTmp[k] *= 1.0/rkk;
    }
#else
    for(long k = 0; k < subspaceSize; k++)
    {
        vArrayTmp[k]  = V[0];
        vArrayTmp[k] *= VtAVeigVector(0,k);
        for(long i = 1; i < subspaceSize; i++)
        {
        vTemp =  V[i];
        vTemp *= VtAVeigVector(i,k);
        vArrayTmp[k] += vTemp;
        }

        auto rkk = std::sqrt(std::abs(vArrayTmp[k].dot(vArrayTmp[k])));
        vArrayTmp[k] *= 1.0/rkk;
    }
#endif
#endif


#ifdef VBLAS_
	#ifdef _OPENMP
	#pragma omp parallel for \
	schedule(static,1)
    #endif
    for(long k = 0; k < subspaceSize; k++)
    {
        vArrayTmp[k] = V[0];
        vArrayTmp[k].scal(VtAVeigVector(0,k));

        for(long i = 1; i < subspaceSize; i++)
        {
        vArrayTmp[k].axpy(VtAVeigVector(i,k),V[i]);
        }

        auto rkk = vArrayTmp[k].nrm2();
        vArrayTmp[k].scal(1.0/rkk);
    }
#endif

    for(long k = 0; k < subspaceSize; k++)
    {
        V[k]  = vArrayTmp[k];
    }

    eigVresiduals.resize(residualCheckCount,0.0);

    OpPtr->apply(vArrayTmp);



#ifdef _OPENMP
    #pragma omp parallel for \
	private(threadNum) \
	schedule(static,1)
    for(long k = 0; k < residualCheckCount; k++)
    {
    	threadNum = omp_get_thread_num();
        MtVarray[threadNum]   = V[k];
        MtVarray[threadNum]  *= VtAVeigValue[k];
        MtVarray[threadNum]  -= vArrayTmp[k];
        eigVresiduals[k]      = MtVarray[threadNum].norm2();
    }
#else
    for(long k = 0; k < residualCheckCount; k++)
    {
        // Compute residuals

        vTemp               = V[k];
        OpPtr->apply(vTemp);
        vArrayTmp[k]        *= VtAVeigValue[k];
        vTemp               -= vArrayTmp[k];
        eigVresiduals[k]     = vTemp.norm2();
    }
#endif
}


double OrthogonalityCheck(std::vector< Vtype >& Avectors, bool printOrthoCheck = false)
{
	double orthoErrorMax = 0.0;

    for(size_t i = 0; i < Avectors.size(); i++)
	{
    	for(size_t j = 0; j < Avectors.size(); j++)
    	{
    	if(printOrthoCheck) {std::cout << std::abs(Avectors[i].dot(Avectors[j])) << " ";}
    	if(i != j)
    	{
    		orthoErrorMax = std::max(orthoErrorMax,std::abs(Avectors[i].dot(Avectors[j])));
    	}
    	else
    	{
    		orthoErrorMax = std::max(orthoErrorMax,std::abs(Avectors[i].dot(Avectors[j]) - 1.0));
    	}
    	}
	    if(printOrthoCheck){std::cout << std::endl;}
	}

    return orthoErrorMax;
}

void setupFinalData()
{
	finalData["maxResidual"]           = 0.0;
	finalData["maxEigValueDifference"] = 0.0;
	finalData["maxRelEigValueGap"]     = 0.0;
    #ifdef TIMING_
	finalData["totalTime"]               = 0.0;
	#endif
}


void setupCountData()
{
    countData["ortho"]       = 0;
    countData["OpApply"]     = 0;
    countData["eigenvalue"]  = 0;
}

void incrementCount(const std::string& countValue,long increment = 1)
{
	 countData[countValue] += increment;
}

void setupTimeData()
{
	#ifdef TIMING_
    timeValue["ortho"]       = 0.0;
    timeValue["OpApply"]     = 0.0;
    timeValue["eigenvalue"]  = 0.0;
    timeValue["totalTime"]   = 0.0;
	#endif
}
void startTimer()
{
    #ifdef TIMING_
    timer.start();
	#endif
}



void incrementTime(const std::string& timedValue)
{
	 #ifdef TIMING_
	 timer.stop();
	 timeValue[timedValue]  = timer.getSecElapsedTime();
     #endif
}

void startGlobalTimer()
{
    #ifdef TIMING_
    globalTimer.start();
	#endif
}
void incrementTotalTime()
{
	 #ifdef TIMING_
	 globalTimer.stop();
	 timeValue["totalTime"]  = globalTimer.getSecElapsedTime();
	 finalData["totalTime"]  = timeValue["totalTime"];
     #endif
}



    Otype* OpPtr;

    bool verboseSubspaceFlag;
    bool verboseFlag;
    bool eigDiagnosticsFlag;

    std::vector< Vtype >        vArray;
    std::vector< Vtype >     vArrayTmp;

    Vtype vTemp;

    // For storage of matrices and eigenvectors of projected system

    RCarray2d<Dtype>       VtAV;
    RCarray2d<Dtype>       VtAVeigVector;

    std::vector<double>      VtAVeigValue;
    std::vector<double>   eigVecResiduals;

    LanczosMaxMinFinder < Vtype, Otype, VRandomizeOpType > lanczosMaxMinFinder;

    LanczosCpoly cPoly;
    LanczosCpolyOperatorLM< Vtype , Otype > cOp;

    JacobiDiagonalizer jacobiMethod;
    bool              useJacobiFlag;

    SCC::DSYEV                dsyev;
    SCC::ZHPEVX              zhpevx;

    double    guardValue;                // Value of the guard eigenvalue.
    bool      intervalStopConditionFlag; // Converge based on value of guard eigenvalue
    bool      hardIntervalStopFlag;
    double    minEigValueEst;
    double    maxEigValueEst;
    double    maxMinTol;

    long     minIntervalPolyDegreeMax;
    long            maxInnerLoopCount;
    bool           nonRandomStartFlag;
    bool         fixedIterationCount;
    long        filterRepetitionCount;

    RC_Types::StopCondition stopCondition;

    std::map<std::string,long>   countData;

    std::map<std::string,double> finalData;


	#ifdef TIMING_
    ClockIt        timer;
    ClockIt  globalTimer;
    std::map<std::string,double> timeValue;
    std::map<std::string,long>   timeCount;
	#endif

    // Temporaries for multi-threading

	#ifdef _OPENMP
    std::vector<Vtype>                                   MtVarray;
	#endif
};

#undef JACOBI_TOL
#undef DEFAULT_MAX_INNER_LOOP_COUNT
#undef RAYLEIGH_CHEBYSHEV_SMALL_TOL_
#undef DEFAULT_MAX_MIN_TOL
#undef DEFAULT_POLY_DEGREE_MAX
#undef DEFAULT_FILTER_REPETITION_COUNT
#undef DEFAULT_USE_JACOBI_FLAG
#undef DEFAULT_USE_RESIDUAL_STOP_CONDITION
#endif

 
