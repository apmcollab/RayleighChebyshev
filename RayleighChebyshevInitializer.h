#include <vector>

#include "SubspaceIteration.h"
#include "OpStar.h"

#ifndef RAYLEIGHT_CHEBYSHEV_INITIALIZER_
#define RAYLEIGHT_CHEBYSHEV_INITIALIZER_
//
// This class is used to create estimates of the algebraically smallest and  largest eigenvalues of
// a self-adjoint operator, A, and optionally, a subspace to initialize the subspace used for
// creating approximations to the algebraically smallest eigenvalues using the Rayleigh-Chebyshev
// procedure.
//
// When an upper bound for lambda_max is specified, then this routine just creates an approximation
// to the algebraically lowest eigenvalue and, optionally, an associated subspace.
//
// Since the Rayleigh-Chebyshev procedure requires an estimate of lambda_max such that
// lambda_max_est >= lambda_max then, in the case when lambda_max is not specified,
// this routine creates an estimate that is within the specified tolerance.
//
// The Rayleigh-Chebyshev procedure does not need a particularly accurate estimate of lambda_min,
// only one such that lambda_min <= lambda_min_est. Therefore, the estimated lambda_min
// is computed by carrying out a fixed number of subspace iterations on the operator
// [-A + lambda_max_est], and, if the result is denoted lambda_star, then
//
// lambda_min_est = lambda_max_est-lambda_star
//
// The subspace is that resulting from the subspace iteration(*).
//
// Alternate member functions are used to enable the invocation for special types of
// problems, e.g. positive or negative definite operators.
//
// (*) In one case, that of an indefinite operator such that it's spectral radius
// is equal to ||lamda_min||, this routine computes an approximation to lambda_min
// that is accurate to within tol and it's associated eigenvector.
//
//
// The algorithm used is a combination of subspace iteration and invocations of the
// Rayleigh-Chebyshev procedure on operators that are transforms of the
// original operator. When the Rayleigh-Chebyshev procedure is invoked, the
// subspace size used is 2, unless otherwise specified, and so it is reasonably
// efficient. The use of this class avoids the use of the Lanczo's procedure to create the
// estimates of the extremal eigenvalues, and hence is stable with respect to
// unsymmetric perturbations of the operator. Currently, symmetry is "enforced"
// by evaluating the "upper half" of the projection of the operator on the
// working subspace and obtaining the "lower half" by reflection.
//
// If the input matrix is a multiple of the identity operator, then this routine will
// return values of lambda_min and lambda_max that agree to within machine precision.
// The norm of the difference between these values can be used to flag identity operators.
//
/*
#############################################################################
#
# Copyright 2009-2015 Chris Anderson
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
template < class Vtype, class Otype, class VRandomizeOpType > class RayleighChebyshevInitializer
{
public:

	RayleighChebyshevInitializer()
	{
	initialize();
	}


    void initialize()
    {
    subspaceIterationCount   = 3;
    nonRandomStartFlag       = false;
    minIntervalPolyDegreeMax = 100;
    }

    void setNonRandomStart()
    {
    nonRandomStartFlag = true;
    }

    void clearNonRandomStart()
    {
    nonRandomStartFlag = false;
    minIntervalPolyDegreeMax = 100;
    }

    void setMinIntervalPolyDegreeMax(long polyDegMax)
    {
    	minIntervalPolyDegreeMax = polyDegMax;
    }

	void createEigSystemEstimates(long subspaceSize, double subspaceTol, Vtype& vStart, Otype& oP,
	VRandomizeOpType& randOp, double& lambdaMin, double& lambdaMax, std::vector < Vtype > & subspace)
	{
	lambdaMin = 0.0;
	lambdaMax = 0.0;

	long lambdaMinFoundFlag = false;
	//
	// Indefinite or definite nature not known
	//
	// Estimate the largest eigenpair associated with A^2 using a combination of
	// subspace iteration followed by an application of RC.
	//
	// Use a subspace of size maxEstimateSubspaceSize = 1
	//

    long maxEstimateSubspaceSize = 1;

	int signOp                 = +1;
    double shift               = 0.0;
    bool squareFlag            = true;
    bool computeEigVectorsFlag = true;

	std::vector <double> eigValues(1,0.0);
	std::vector <Vtype>  eigVectors;

#ifdef _OPEN_MP_
    opStar.initialize(oP,signOp,shift,squareFlag);
    subspaceIter_OpStar.applySubspaceIteration(subspaceIterationCount,maxEstimateSubspaceSize, vStart, opStar, randOp, eigValues, eigVectors, computeEigVectorsFlag);
#else
    opStarRef.initialize(oP,signOp,shift,squareFlag);
    subspaceIter_OpStarRef.applySubspaceIteration(subspaceIterationCount,maxEstimateSubspaceSize, vStart, opStarRef, randOp, eigValues, eigVectors, computeEigVectorsFlag);
#endif

	// Now evaluate the maximal eigenpair of A^2 by applying Rayleigh-Chebyshev to the operator
	// -A^2 with lamdaMin for this computation.

    double lambdaMinTemp = -eigValues[0];
	double lambdaMaxTemp  =  0.0;

#ifdef _OPEN_MP_
    signOp 		= -1;
    shift 		= 0.0;
    squareFlag  = true;
    opStar.setSignShiftAndSquareFlag(signOp,shift,squareFlag);
    RayleighChebyshev < Vtype, OpStar < Vtype, Otype > , VRandomizeOpType > RCeigProcedure;
#else
    signOp 		= -1;
    shift 		= 0.0;
    squareFlag  = true;
    opStarRef.setSignShiftAndSquareFlag(signOp,shift,squareFlag);
 	RayleighChebyshev < Vtype, OpStarRef < Vtype, Otype > , VRandomizeOpType > RCeigProcedure;
#endif


    RCeigProcedure.setMinIntervalPolyDegreeMax(minIntervalPolyDegreeMax);
	RCeigProcedure.setVerboseFlag();

    long subspaceIncrementSize = 1;
    long bufferSize            = 1;
    long eigCount              = 1;

    RCeigProcedure.setNonRandomStartFlag();
#ifdef _OPEN_MP_
    RCeigProcedure.getMinIntervalEigenSystem(lambdaMinTemp, lambdaMaxTemp, lambdaMaxTemp, subspaceTol, subspaceIncrementSize,
    bufferSize, eigCount, vStart, opStar, randOp, eigValues, eigVectors);
#else
	RCeigProcedure.getMinIntervalEigenSystem(lambdaMinTemp, lambdaMaxTemp, lambdaMaxTemp, subspaceTol, subspaceIncrementSize,
    bufferSize, eigCount, vStart, opStarRef, randOp, eigValues, eigVectors);
#endif

	// Compute estimate of the extremal eigenvalue of A

	vStart = eigVectors[0];
	double vNorm2 = vStart.dot(vStart);
	oP.applyForwardOp(vStart);
	double extremalEig  = eigVectors[0].dot(vStart)/vNorm2;

    //
    // If we have a positive Rayleigh quotient, or one whose magnitude differs
    // from the square root of the magnitude of the -A^2 eigenvalue, then use
    // as an upper bound sqrt(std::abs(eigValues[0])). This provides a good
    // upper bound even in the case when there is +/- eigensystem degeneracy.
    //
    if( (std::abs(std::abs(extremalEig) - sqrt(std::abs(eigValues[0])))  > subspaceTol) || (extremalEig >= 0.0) )
    {
    // set lambdaMax and clear the eigenvectors array

    lambdaMax = sqrt(std::abs(eigValues[0]));
    eigVectors.clear();
    }
    else // In this case we have a high-quality estimate of the lowest eigen pair
    {

    // Capture the eigenpair and insert the eigenvector as the first element of the subspace

    lambdaMinFoundFlag = true;
    lambdaMin  =   extremalEig;
    if(subspace.size() ==  0)  {subspace.resize(1,vStart);}
    subspace[0] = eigVectors[0];

    eigVectors.clear();

    // now work with A + ||lambdaMin|| to obtain an estimate of lambdaMax

	signOp                = +1;
    shift                 = std::abs(lambdaMin);
    squareFlag            = false;
    computeEigVectorsFlag = true;

	eigValues.resize(1,0.0);

#ifdef _OPEN_MP_
    opStar.setSignShiftAndSquareFlag(signOp,shift,squareFlag);
	subspaceIter_OpStar.applySubspaceIteration(subspaceIterationCount, maxEstimateSubspaceSize, vStart, opStar, randOp, eigValues, eigVectors, computeEigVectorsFlag);
#else
    opStarRef.setSignShiftAndSquareFlag(signOp,shift,squareFlag);
    subspaceIter_OpStarRef.applySubspaceIteration(subspaceIterationCount, maxEstimateSubspaceSize, vStart, opStarRef, randOp, eigValues, eigVectors, computeEigVectorsFlag);
#endif

    // Now work with -(A + std::abs(lambdaMin))

	signOp                = -1;
    shift                 = std::abs(lambdaMin);
    squareFlag            = false;
    computeEigVectorsFlag = true;

    lambdaMinTemp =   -eigValues[0];
	lambdaMaxTemp  =  0.0;

    RCeigProcedure.setNonRandomStartFlag();
#ifdef _OPEN_MP_
    opStar.setSignShiftAndSquareFlag(signOp,shift,squareFlag);
    RCeigProcedure.getMinIntervalEigenSystem(lambdaMinTemp, lambdaMaxTemp, lambdaMaxTemp, subspaceTol, subspaceIncrementSize,
    bufferSize, eigCount, vStart, opStar, randOp, eigValues, eigVectors);
#else
    opStarRef.setSignShiftAndSquareFlag(signOp,shift,squareFlag);
	RCeigProcedure.getMinIntervalEigenSystem(lambdaMinTemp, lambdaMaxTemp, lambdaMaxTemp, subspaceTol, subspaceIncrementSize,
    bufferSize, eigCount, vStart, opStarRef, randOp, eigValues, eigVectors);
#endif

    // Compute estimate of the maximal eigenvalue of A

	vStart = eigVectors[0];
	vNorm2 = vStart.dot(vStart);
	oP.applyForwardOp(vStart);
	lambdaMax  = eigVectors[0].dot(vStart)/vNorm2;
    }

    //
    // Now have a good estimate of lambdaMax, so create a subspace for the
    // lowest eigenvalues.
    //

	// Initialize subspace return argument

	// Quick return if we've already found lambdaMin and we only
	// want a subspace of size 1.
	//
    if((lambdaMinFoundFlag)&&(subspaceSize == 1))
    {
    printf("lambdaMin = %10.5f lambdaMax = %10.5f \n",lambdaMin,lambdaMax);
    return;
    }

	// If the lambdaMin has been accurately determined, then
	// cache it to re-insert into subspace after power iteration.

	if(lambdaMinFoundFlag){eigVectors[0] = subspace[0];}

    bool setNonRandomFlag;
    if(nonRandomStartFlag)
    {
    setNonRandomFlag = true;
    }

    if((not nonRandomStartFlag)&&(lambdaMinFoundFlag))
    {
    subspace.resize(1);
    setNonRandomFlag = true;
    }
    else
    {
    setNonRandomFlag = false;
    subspace.clear();
    }

 	// Create an approximate subspace for the subspace size lowest eigenvalues by
 	// applying a subspace iteration to
 	//
 	// -(A-lambdaMax)
 	//

    signOp                = -1;
    shift                 = -lambdaMax;
    squareFlag            = false;
    computeEigVectorsFlag = true;
    eigValues.resize(subspaceSize,0.0);


#ifdef _OPEN_MP_
    opStar.setSignShiftAndSquareFlag(signOp,shift,squareFlag);
    if(setNonRandomFlag) subspaceIter_OpStar.setNonRandomStart();
	subspaceIter_OpStar.applySubspaceIteration(subspaceIterationCount, subspaceSize, vStart, opStar, randOp, eigValues, subspace, computeEigVectorsFlag);
#else
    opStarRef.setSignShiftAndSquareFlag(signOp,shift,squareFlag);
    if(setNonRandomFlag) subspaceIter_OpStarRef.setNonRandomStart();
    subspaceIter_OpStarRef.applySubspaceIteration(subspaceIterationCount, subspaceSize, vStart, opStarRef, randOp, eigValues, subspace, computeEigVectorsFlag);
#endif

    // lambda* = approximation to -(lambdaMin-lambdaMax) = lambdaMax-lambdaMin, so lambdaMiin = lambdaMax - lambda*

    for(long i = 0; i < (long)eigValues.size(); i++)
    {
    eigValues[i] = lambdaMax - eigValues[i];
    }


  	if(lambdaMinFoundFlag)
  	{
  	subspace[0] = eigVectors[0];
  	orthogonalizeVarray(subspace,eigVectors[0]); // re-orthonormalize using eigVectors[0] as the temp array.
  	}
  	else
  	{
    lambdaMin = eigValues[0];
    }
}
//
//  orthogonalizeVarray uses modified Gram-Schmidt to orthonoramlize
//  the vectors in vArray. It is assumed that the input std::vector
//  vTemp has been initialized.
//
void orthogonalizeVarray(std::vector< Vtype >& vArray, Vtype& vTemp)
{
    long subspaceSize = (long)vArray.size();
	double rkk;
	double rkj;

#ifndef _VBLAS_
//  Orthogonalize the subspace vectors using Modified Gram-Schmidt

    for(long k = 1; k <= subspaceSize; k++)
    {
        rkk     = sqrt(vArray[k-1].dot(vArray[k-1]));
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
#ifdef _OPEN_MP_
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

    long subspaceIterationCount;

    OpStar            < Vtype, Otype > opStar;
    OpStarRef         < Vtype, Otype > opStarRef;
	SubspaceIteration < Vtype, OpStar< Vtype , Otype >,     VRandomizeOpType > subspaceIter_OpStar;
    SubspaceIteration < Vtype, OpStarRef < Vtype , Otype >, VRandomizeOpType > subspaceIter_OpStarRef;
    VRandomizeOpType  randOp;
    bool nonRandomStartFlag;
    long minIntervalPolyDegreeMax;
};



#endif
