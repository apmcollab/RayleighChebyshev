#ifndef  RC_JACOBI_DIAGONALIZER_
#define  RC_JACOBI_DIAGONALIZER_

#define JACOBI_DIAGONALIZER_TOL 1.0e-15


//
//####################################################################
//                    JacobiDiagonalizer.h
//####################################################################
/**
   A class whose member functions utilize Jacobi's method to find
   eigenvalues and optionally eigenvectors of a real symmetric matrix.

   The member functions of this class assume the I/O data for the
   eigenvalues and eigenvectors are stored in contiguous memory
   locations. If one is passing a data pointer to the data from a matrix or
   vector class, then one has to be sure that the data storage for
   that class is stored in contiguous memory locations.

   If you are unsure, then use temporary 2D and 1D arrays that
   utilize contiguous memory locations.

Chris Anderson
UCLA Dept. of Mathematics
(C) UCLA 2006

Version : Tue Feb 14 16:02:58 2012
*/

/*
#############################################################################
#
# Copyright 2006- Chris Anderson
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

//####################################################################
//

#include <cmath>
#include <iostream>

//
// The default assumed storage format for the input matrix
// is a matrix stored by rows.  Column storage format for
// the input matrix is specified by invoking the
// member function clearIOdataRowStorage() after
// a class instance is declared but before any computation
// is performed.
//
// Internally the matrix is stored by columns.
//

class JacobiDiagonalizer
{
public:

JacobiDiagonalizer(void)
{
    matrixData  = 0;
    eVecData    = 0;
    eigVecFlag  = 0;
    diagData    = 0;
    sortIndex   = 0;
    sortIncreasing   = false;
    IOdataRowStorage = true;
}

~JacobiDiagonalizer(void)
{
    if(matrixData != 0) delete [] matrixData;
    if(eVecData   != 0) delete [] eVecData;
    if(diagData   != 0) delete [] diagData;
    if(sortIndex  != 0) delete [] sortIndex;
}



void setSortIncreasing(bool val)
{
	sortIncreasing = val;
}

void clearSortIncreasing()
{
	sortIncreasing = false;
}

void setIOdataRowStorage(bool val)
{
	IOdataRowStorage = val;
}

void clearIOdataRowStorage()
{
	IOdataRowStorage = false;
}

void applyJacobiTransformation(long i, long j,double c, double s, double t)
{
    long k;

    double tau = s/(1.0 + c);
    double t1; 
    double t2;

    for(k = 0; k < i; k++)
    {
     t1 = M(k,i);
     t2 = M(k,j);
     M(k,i) = t1 + s*(t2 - tau*t1);
     M(k,j) = t2 - s*(t1 + tau*t2);
    }

    for(k = j+1; k < matrixDimension; k++)
    {
    t1 = M(i,k);
    t2 = M(j,k);
    M(i,k) = t1 + s*(t2 - tau*t1);
    M(j,k) = t2 - s*(t1 + tau*t2);
    }

    for(k = i+1; k < j; k++)
    {
	t1 = M(i,k);
    t2 = M(k,j);
    M(i,k) = t1 + s*(t2 - tau*t1);
    M(k,j) = t2 - s*(t1 + tau*t2);
    }
    
    //
    // Capture the increments for the diagonal elements 
    // 

    diagData[i] += t*M(i,j);
    diagData[j] -= t*M(i,j);

    M(i,j)  = 0.0;
}

void updateEigenvectors(long i, long j, double c, double s)
{
   long k;

    double tau = s/(1.0 + c);
    double t1; 
    double t2;

    for(k = 0; k < matrixDimension; k++)
    {
     t1 = eVector(k,i);
     t2 = eVector(k,j);
     eVector(k,i) = t1 + s*(t2 - tau*t1);
     eVector(k,j) = t2 - s*(t1 + tau*t2);
    }
}
//
//  This routine computes the eigenvalues of the matrix 
//  using Jacobi's method. There is no bounds checking
//  on the arrays. 
//
//  --Input--
//  
//  mData   : Double array of size n*n containing the data
//            for a symmetric matrix. Only the upper triangular
//            part of the matrix data is actually used. 
//
//    n     : The dimension of the system
//
//  eigVal  : Double array of size n to hold the computed 
//            eigenvalues
//
//  --Output-- 
//
//  eigVal  :  contains the eigenvalues (sorted)
//
// Note: The default assumed storage format for the input matrix
// is a matrix stored by rows.  Column storage format for
// the input matrix is specified by invoking the
// member function clearIOdataRowStorage() after
// a class instance is declared but before any computation
// is performed.
//
void getEigenvalues(double* mData, long n, double* eigVal)
{
	eigVecFlag = 0;
    initialize(mData,n);

    computeEigenDecomposition();

    long k;
//
//  Capture the eignevalues
//
    for(k = 0; k < matrixDimension; k++)
    {
    eigVal[k] = M(k,k);
    }

    sortWithIndex(eigVal, sortIndex, matrixDimension);
}
//
//  This routine computes the eigenvalues of the matrix 
//  using Jacobi's method. There is no bounds checking
//  on the arrays. 
//
//  --Input--
//  
//  mData   : Double array of size n*n containing the data
//            for a symmetric matrix. Only the upper triangular
//            part of the matrix data is actually used. 
//
//    n    : The dimension of the system
//
//  eigVal : Double array of size n to hold the computed 
//           eigenvalues
//
//  eigVec : Double array of size n*n to hold the computed 
//           eigenvectors.
//
//  --Output-- 
//
//  eigVal : contains the computed eigenvalues (sorted)
//
//  eigVec : the jth column of the array contains the
//           jth normalized eigenvector. The array data 
//           is stored by rows. 
//
// Note: The default assumed storage format for the input matrix
// is a matrix stored by rows.  Column storage format for
// the input matrix is specified by invoking the
// member function clearIOdataRowStorage() after
// a class instance is declared but before any computation
// is performed.
//              
void getEigenSystem(double* mData, long n, double* eigVal, double* eigVec)
{
	eigVecFlag = 1;
    initialize(mData,n);
    computeEigenDecomposition();

    long k;
//
//  Capture the eignevalues
//
    for(k = 0; k < matrixDimension; k++)
    {
    eigVal[k] = M(k,k);
    }

	sortWithIndex(eigVal, sortIndex, matrixDimension);
//
//  Capture the eigenvectors (sorted)
//
    long i; long j;

    if(IOdataRowStorage)
    {
	for(i = 0; i < matrixDimension; i++)
    {
    for(j = 0; j < matrixDimension; j++)
    {
    *(eigVec + j + i*matrixDimension) = eVector(i,sortIndex[j]);
    }}
    }
    else
    {
    for(i = 0; i < matrixDimension; i++)
    {
    for(j = 0; j < matrixDimension; j++)
    {
    *(eigVec + i + j*matrixDimension) = eVector(i,sortIndex[j]);
    }}
    }
}

void computeEigenDecomposition()
{
    double omega          = OffFrobeniusNorm2();
    double N              = (matrixDimension*(matrixDimension-1))/2.0;
    double tauThreshold   = std::sqrt(omega/N);
    int    rotFlag;
    double omegaIncrement;
    double omegaBase;

    // cout << "Initial Threshold : " << tauThreshold << endl;

    long i; long j; long k;

    double tau; double t; 
    double c;   double s;
    double M1; double M2;

    double signTau = 1.0;
    long sweepCount = 0;

    while(tauThreshold > JACOBI_DIAGONALIZER_TOL)
    {
    rotFlag        = 0;
    omegaIncrement = 0.0;
    omegaBase      = omega;

    for(k = 0; k < matrixDimension; k++)
    {
    diagData[k] = 0.0;
    }

    for(i = 0;   i < matrixDimension-1; i++)
    {
    for(j = i+1; j < matrixDimension; j++)
    {
		if(std::abs(M(i,j)) > tauThreshold)
		{

		omegaIncrement += M(i,j)*M(i,j);

        rotFlag        = 1;
        omega          = omegaBase - omegaIncrement;

        if(omega < 0.0) 
        {
        omega          = OffFrobeniusNorm2();
        omegaIncrement = 0.0;
        omegaBase      = omega;
        }

		tauThreshold   = std::sqrt(omega/N);

        M1     =  M(i,i)      - M(j,j);
        M2     =  diagData[i] - diagData[j];
		tau     = (M1 + M2)/(2.0*M(i,j));
		signTau = (tau >= 0.0) ? 1.0 : -1.0;
		t       = signTau/(std::abs(tau) + std::sqrt(1.0 + tau*tau));
		c       = 1.0/(std::sqrt(1.0 + t*t));
		s       = c*t; 

		applyJacobiTransformation(i,j,c,s,t);
        if(eigVecFlag == 1){updateEigenvectors(i,j,c,s);}
		}
    }}

    if(rotFlag == 0) break;
    //
    // Update diagonal elements
    //
    for(k = 0; k < matrixDimension; k++)
    {
    M(k,k) += diagData[k];
    }

    //
    // Recompute threshold
    //

    omega          = OffFrobeniusNorm2();
    tauThreshold   = std::sqrt(omega/N);
    sweepCount++;

    //cout << "Threshold : " << tauThreshold << endl;
    }

    //cout << "Number of Sweeps " << sweepCount << endl;
}

inline double& M(long i, long j)
{
        return *(matrixData + i + j*matrixDimension);
}

inline double& eVector(long i, long j)
{
        return *(eVecData  + i  + j*matrixDimension);
}
//
// Initializes the internal data arrays and copies
// over the data.
//
void initialize(double* mData, long n)
{
    if(matrixData != 0)
    {
    if(n != matrixDimension)
    {
    delete [] matrixData; matrixData = 0;
    delete [] diagData;   diagData   = 0;
    delete [] sortIndex;  sortIndex  = 0;
    if(eigVecFlag == 1)
    {if(eVecData != 0){delete [] eVecData;  eVecData = 0;}}
    }
    }
    matrixDimension = n;
    matrixDataSize  = matrixDimension*matrixDimension;

    if(matrixData  == 0) matrixData  = new double[matrixDataSize];
    if(diagData    == 0) diagData    = new double[matrixDimension];
    if(sortIndex   == 0) sortIndex   = new long[matrixDimension];

    if(eigVecFlag   == 1)
    {
     if(eVecData == 0)
     eVecData = new double[matrixDimension*matrixDimension];
    }
//
//  Store input matrix in matrixData
//
    long i; long j; long k;

    if(IOdataRowStorage)
    {
    	for(i = 0; i < matrixDimension; i++)
    	{
    		for(j = i; j < matrixDimension; j++)
    		{
    			M(i,j) = *(mData + j + i*matrixDimension);
    		}}
    }
    else // input data in column storage
    {
    	for(i = 0; i < matrixDimension; i++)
    	{
    		for(j = i; j < matrixDimension; j++)
    		{
    			M(i,j) = *(mData + i + j*matrixDimension);
    		}}
    }
//
//  Initialize eigenvector data array
//
    if(eigVecFlag == 1)
    {
    for(k = 0; k < matrixDimension*matrixDimension; k++)
    {eVecData[k] = 0.0;}

    for(i = 0; i < matrixDimension; i++)
    {
      eVector(i,i) = 1.0;
    }
    }
}
//
//  Returns the current state of the internal matrix 
//
void getCurrentMatrix(double* mData)
{
	long i; long j; 

	for(i = 0; i < matrixDimension; i++)
    {
    for(j = i; j < matrixDimension; j++)
    {
    *(mData + j + i*matrixDimension) = M(i,j);
    *(mData + i + j*matrixDimension) = M(i,j);
    }}
}


double OffFrobeniusNorm2()
{
    long i; 
    long j;
    double fNorm2 = 0.0;
	for(i = 0; i < matrixDimension; i++)
    {
    for(j = i+1; j < matrixDimension; j++)
    {
    fNorm2 += M(i,j)*M(i,j);
    }}
    return fNorm2;
}

void sortWithIndex(double* vals, long* index, long n)
{
	long i,j,inc;

    for(i = 0; i < n; i++) // set up index array 
    {index[i] = i;}

//
//  Do a shell sort
//
	double valTmp;
    long   indTmp;
	inc=1;
	do 
    {
    inc *= 3;
    inc++;
	} while (inc <= n);
	do 
    {
		inc /= 3;
		for (i=inc+1;i<=n;i++) 
        {
			valTmp=vals[i-1];
            indTmp=index[i-1];
			j=i;
			if(sortIncreasing)
			{
			while (vals[(j-inc)-1] > valTmp) {
				vals[j-1]  = vals[(j-inc)-1];
                index[j-1] = index[(j-inc)-1];
				j -= inc;
				if (j <= inc) break;
			}}
			else
			{
				while (vals[(j-inc)-1] < valTmp) {
				vals[j-1]  = vals[(j-inc)-1];
                index[j-1] = index[(j-inc)-1];
				j -= inc;
				if (j <= inc) break;
			}}

			vals[j-1]  = valTmp;
            index[j-1] = indTmp;
		}
	} while (inc > 1);
}

double* matrixData;
double* eVecData;
double* diagData;
long  * sortIndex;

double                  tol;
int              eigVecFlag;
long        matrixDimension;
long         matrixDataSize;

bool         sortIncreasing;
bool       IOdataRowStorage;
};

#undef JACOBI_DIAGONALIZER_TOL

#endif


 
