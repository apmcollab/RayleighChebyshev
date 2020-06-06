
//
//                        SubspaceIteration.h
//
//
//  This applySubspaceIteration(...) iteration method of this class
//  carries out a specified number of subspace iterations and then
//  computes the eigenvalues (and optionally the eigenvectors) associated
//  with the projection of the operator onto the final subspace.
//
//  The eigenvalues are returned with the ordering of algebraically largest
//  to smallest.
//
//  eigValues[0] >=  eigValues[1] >= eigValues[2] ...
//
//
//  ToDo: Think about a better way to multi-thread the computation of
//         A*V required for the formation of V'*AV.
//
// Author: Chris Anderson
// Date  : Jan. 18, 2014
//
/*
#############################################################################
#
# Copyright 2014-2015 Chris Anderson
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
#include <std::vector>

#ifndef SUBSPACE_ITERATION_
#define SUBSPACE_ITERATION_

template < class Vtype, class Otype, class VRandomizeOpType > class SubspaceIteration
{
public:

    SubspaceIteration()
    {
    initialize();
    }

    SubspaceIteration(const SubspaceIteration& S)
    {
    initialize();
    }

    void initialize()
    {
    nonRandomStartFlag = false;
    }

    void setNonRandomStart()
    {
    nonRandomStartFlag = true;
    }

    void clearNonRandomStart()
    {
    nonRandomStartFlag = false;
    }

	void applySubspaceIteration(long iterationCount, long subspaceSize, Vtype& vStart, Otype& oP,
	VRandomizeOpType& randOp, std::vector<double>& eigValues, std::vector < Vtype > & eigVectors,
	bool computeEigVectorsFlag = true)
	{
	eigValues.clear();
	eigValues.resize(subspaceSize,0.0);

	vTemp.initialize(vStart);

    //
    // Set up approximating subspace.
    //

    long vectorDimension = vStart.getDimension();

    if(subspaceSize > vectorDimension) {subspaceSize = vectorDimension;}

    // Initialize subspace vectors using random vectors, or input
    // starting vectors if the latter is specified.

    long inputSubspaceSize = eigVectors.size();

    if((not nonRandomStartFlag)||(inputSubspaceSize == 0))
    {
    	eigVectors.resize(subspaceSize,vStart);
    	for(long k = 0; k < subspaceSize; k++)
    	{
    		randOp.randomize(eigVectors[k]);
    	}
    }
    else
    {
    	if(subspaceSize > inputSubspaceSize)
    	{
            eigVectors.resize(subspaceSize,vStart);
    		for(long k = inputSubspaceSize-1; k < subspaceSize; k++)
    		{
    			randOp.randomize(eigVectors[k]);
    		}
    	}
    	else
    	{
    		eigVectors.resize(subspaceSize);
    	}
    }

    // Perform an initial orthogonalization

    orthogonalizeVarray(eigVectors);

    // Carry out iterationCount subspace iterations

    for(long i = 0; i < iterationCount; i++)
	{
		for(long k = 0; k < subspaceSize; k++)
		{
   		 oP.applyForwardOp(eigVectors[k]);
    	}
    	orthogonalizeVarray(eigVectors);
    }

    // Compute the eigenvalues and eigenvectors of the
    // using the final subspace

    RC_Double2Darray           VtAV;
    RC_Double2Darray  VtAVeigVector;

    VtAV.initialize(subspaceSize,subspaceSize);
    VtAVeigVector.initialize(subspaceSize,subspaceSize);

//  Form Vt*A*V. This implementation assumes A is a self-adjoint
//  matrix with respect to the associated std::vector's dot product.

    long i; long j;

    for(i = 0; i < subspaceSize; i++)
    {
        vTemp     = eigVectors[i];
        oP.applyForwardOp(vTemp);
#ifdef _OPEN_MP_
		#pragma omp parallel for \
		private(j) \
		schedule(static,1)
#endif
        for(j = i; j < subspaceSize; j++)
        {
        VtAV(j,i) = eigVectors[j].dot(vTemp);
        VtAV(i,j) = VtAV(j,i);
        }
    }
//
//  The jacobiMethod procedure returns the eigenvalues
//  and eigenvectors ordered from largest to smallest, e.g.
//
//  VtAVeigValue[0] >= VtAVeigValue[1] >= VtAVeigValue[2] ...
//
    double* VtAVdataPtr;
    double* VtAVeigValueDataPtr;
    double* VtAVeigVectorDataPtr;

    VtAVdataPtr           = VtAV.getDataPointer();
    VtAVeigValueDataPtr   = &eigValues[0];
    VtAVeigVectorDataPtr  = VtAVeigVector.getDataPointer();

    JacobiDiagonalizer  jacobiMethod;
    jacobiMethod.getEigenSystem(VtAVdataPtr, subspaceSize, VtAVeigValueDataPtr, VtAVeigVectorDataPtr);

    // Return if we don't need to create the approximate eigenvectors as well.

    if(not computeEigVectorsFlag) return;

    // Construct required array of std::vector temporaries

    std::vector < Vtype> vArrayTmp(subspaceSize);

	long     k;
	double rkk;
#ifndef _VBLAS_
    //
    // Create eigenvectors from the current subspace
    //
    for(k = 0; k < subspaceSize; k++)
    {
        vArrayTmp[k].initialize(eigVectors[0]);
        vArrayTmp[k] *= VtAVeigVector(0,k);
        for(i = 1; i < subspaceSize; i++)
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

#ifdef _OPEN_MP_
		#pragma omp parallel for \
		private(i,k,rkk) \
		schedule(static,1)
#endif
    for(k = 0; k < subspaceSize; k++)
    {
        vArrayTmp[k].initialize(eigVectors[0]);
        vArrayTmp[k].scal(VtAVeigVector(0,k));

        for(i = 1; i < subspaceSize; i++)
        {
        vArrayTmp[k].axpy(VtAVeigVector(i,k),eigVectors[i]);
        }

        rkk = vArrayTmp[k].nrm2();
        vArrayTmp[k].scal(1.0/rkk);
    }
#endif

//
//  Capture the eigenvectors
//
    for(k = 0; k < subspaceSize; k++)
    {
        eigVectors[k] = vArrayTmp[k];
	}
}

//
//  orthogonalizeVarray uses modified Gram-Schmidt to orthonoramlize
//  the vectors in vArray.
//
//  This routine assumes that the class data member
//  vTemp has been initialized
//
void orthogonalizeVarray(std::vector< Vtype >& vArray)
{
    long subspaceSize = vArray.size();
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

    Vtype             vTemp;
    bool nonRandomStartFlag;
};

#endif
