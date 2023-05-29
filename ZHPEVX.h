/*
 * ZHPEVX.h
 *
 *  Created on: Apr 9, 2021
 *      Author: anderson
 */
#include "LapackInterface/SCC_LapackHeaders.h"
#include "LapackMatrixCmplx16.h"

/*
#############################################################################
#
# Copyright 2021- Chris Anderson
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


/*
subroutine zhpevx	(	character 	JOBZ,
character 	RANGE,
character 	UPLO,
integer 	N,
complex*16, dimension( * ) 	AP,
double precision 	VL,
double precision 	VU,
integer 	IL,
integer 	IU,
double precision 	ABSTOL,
integer 	M,
double precision, dimension( * ) 	W,
complex*16, dimension( ldz, * ) 	Z,
integer 	LDZ,
complex*16, dimension( * ) 	WORK,
double precision, dimension( * ) 	RWORK,
integer, dimension( * ) 	IWORK,
integer, dimension( * ) 	IFAIL,
integer 	INFO
)
 */

#ifndef  ZHPEVX_H_
#define  ZHPEVX_H_

namespace SCC
{

extern "C"  void zhpevx_(char*JOBZ, char* RANGE, char* UPLO,long* N, double* AP, double* VL, double* VU,
                         long*   IL, long*   IU, double*   ABSTOL, long*   M, double* W, double* Z, long* LDZ,
             double* WORK, double* RWORK, long* IWORK, long* IFAIL, long* INFO);

class ZHPEVX
{
public :

	ZHPEVX(){}

	void initialize()
	{
	AP.initialize();
    WORK.initialize();
    RWORK.clear();
    IWORK.clear();
    IFAIL.clear();
	}

	// Computes the eigCount algebraically smallest eigenvalues and eigenvectors.
	// The value returned is the number of eigenvalues found.

	long createAlgSmallestEigensystem(long eigCount, SCC::LapackMatrixCmplx16& A, std::vector<double>& eigValues,
			                          SCC::LapackMatrixCmplx16& eigVectors)
	{
    if(A.getRowDimension() != A.getColDimension())
	{
			throw std::runtime_error("\nZHPEVX : Non-square matrix input argument  \n");
	}

    long N = A.getRowDimension();

    if(eigCount > N)
    {
    	std::stringstream sout;
    	sout << "\nZHPEVX Error \n";
    	sout << "Requested number of eigenvalues/eigenvectors exceeds system dimension. \n";
    	throw std::runtime_error(sout.str());
    }

    char JOBZ   = 'V'; // Specify N for eigenvalues only
    char RANGE  = 'I'; // Specify index range of eigenvalues to be find (A for all, V for interval)
    char UPLO   = 'U'; // Store complex Hermetian matrix in upper trianglar packed form

    AP.initialize(A.createUpperTriPacked());

    double VL = 0;
    double VU = 0;

    long IL = 1;        // Index of smallest eigenvalue returned
    long IU = eigCount; // Index of largest  eigenvalue returned

    char   DLAMCH_IN = 'S';
    double ABSTOL    =  2.0*(dlamch_(&DLAMCH_IN));

    long M = 0;                            // Number of eigenvalues output

    eigValues.clear();                     // W parameter in original call
    eigValues.resize(N,0.0);

    long LDZ   = N;
    long Mstar = (IU-IL) + 1;              // Maximal number of eigenvalues to be computed when using index specification

    eigVectors.initialize(LDZ,Mstar);      // Matrix whose columns containing the eigenvectors (Z in original call)

    long INFO = 0;

    WORK.initialize();
    RWORK.clear();
    IWORK.clear();
    IFAIL.clear();

    WORK.initialize(2*N,1);
    RWORK.resize(7*N,0.0);
    IWORK.resize(5*N,0);
    IFAIL.resize(N,0);


    zhpevx_(&JOBZ, &RANGE, &UPLO,&N,AP.mData.getDataPointer(),&VL,&VU,&IL,&IU,&ABSTOL,&M,eigValues.data(),
    eigVectors.mData.getDataPointer(),&LDZ,WORK.mData.getDataPointer(),RWORK.data(),IWORK.data(),IFAIL.data(),&INFO);

    if(INFO != 0)
    {
    	std::stringstream sout;
    	sout << "\nZHPEVX \nError INFO = " << INFO << "\n";
    	throw std::runtime_error(sout.str());
    }

    // resize the eig values array to the number of eigenvalues found

    eigValues.resize(M);
    return M;
	}
	
	
    // Computes the eigCount algebraically smallest eigenvalues and eigenvectors.
	// The value returned is the number of eigenvalues found.

	long createEigensystem(SCC::LapackMatrixCmplx16& A, std::vector<double>& eigValues, SCC::LapackMatrixCmplx16& eigVectors)
	{
    if(A.getRowDimension() != A.getColDimension())
	{
			throw std::runtime_error("\nZHPEVX : Non-square matrix input argument  \n");
	}

    long N = A.getRowDimension();

    char JOBZ   = 'V'; // Specify N for eigenvalues only
    char RANGE  = 'A'; // Specify index range of eigenvalues to be find (A for all, V for interval)
    char UPLO   = 'U'; // Store complex Hermetian matrix in upper trianglar packed form

    AP.initialize(A.createUpperTriPacked());

    double VL = 0;
    double VU = 0;

    long IL = 1; // Index of smallest eigenvalue returned
    long IU = N; // Index of largest  eigenvalue returned

    char   DLAMCH_IN = 'S';
    double ABSTOL    =  2.0*(dlamch_(&DLAMCH_IN));

    long M = 0;                            // Number of eigenvalues output

    eigValues.clear();                     // W parameter in original call
    eigValues.resize(N,0.0);

    long LDZ   = N;
    long Mstar = N;                       // Maximal number of eigenvalues to be computed when using index specification

    eigVectors.initialize(LDZ,Mstar);      // Matrix whose columns containing the eigenvectors (Z in original call)

    long INFO = 0;

    WORK.initialize();
    RWORK.clear();
    IWORK.clear();
    IFAIL.clear();

    WORK.initialize(2*N,1);
    RWORK.resize(7*N,0.0);
    IWORK.resize(5*N,0);
    IFAIL.resize(N,0);


    zhpevx_(&JOBZ, &RANGE, &UPLO,&N,AP.mData.getDataPointer(),&VL,&VU,&IL,&IU,&ABSTOL,&M,eigValues.data(),
    eigVectors.mData.getDataPointer(),&LDZ,WORK.mData.getDataPointer(),RWORK.data(),IWORK.data(),IFAIL.data(),&INFO);

    if(INFO != 0)
    {
    	std::stringstream sout;
    	sout << "\nZHPEVX \nError INFO = " << INFO << "\n";
    	throw std::runtime_error(sout.str());
    }

    // resize the eig values array to the number of eigenvalues found

    eigValues.resize(M);
    return M;
	}

    long createAlgSmallestEigenvalues(long eigCount, SCC::LapackMatrixCmplx16& A, std::vector<double>& eigValues)
	{
    if(A.getRowDimension() != A.getColDimension())
	{
	    throw std::runtime_error("\nZHPEVX : Non-square matrix input argument  \n");
	}

    long N = A.getRowDimension();

    if(eigCount > N)
    {
    	std::stringstream sout;
    	sout << "\nZHPEVX Error \n";
    	sout << "Requested number of eigenvalues exceeds system dimension. \n";
    	throw std::runtime_error(sout.str());
    }

    char JOBZ   = 'N'; // Specify N for eigenvalues only
    char RANGE  = 'I'; // Specify index range of eigenvalues to be find (A for all, V for interval)
    char UPLO   = 'U'; // Store complex Hermetian matrix in upper trianglar packed form

    AP.initialize(A.createUpperTriPacked());

    double VL = 0;
    double VU = 0;

    long IL = 1;        // Index of smallest eigenvalue returned
    long IU = eigCount; // Index of largest  eigenvalue returned

    char   DLAMCH_IN = 'S';
    double ABSTOL    =  2.0*(dlamch_(&DLAMCH_IN));

    long M = 0;                     // Number of eigenvalues output

    eigValues.clear();              // W parameter in original call
    eigValues.resize(N,0.0);

    long LDZ     = 1;
    double Zdata = 0.0;              // Double value to be used as reference for null eigenvector

    long INFO = 0;

    WORK.initialize();
    RWORK.clear();
    IWORK.clear();
    IFAIL.clear();

    WORK.initialize(2*N,1);
    RWORK.resize(7*N,0.0);
    IWORK.resize(5*N,0);
    IFAIL.resize(N,0);


    zhpevx_(&JOBZ, &RANGE, &UPLO,&N,AP.mData.getDataPointer(),&VL,&VU,&IL,&IU,&ABSTOL,&M,eigValues.data(),
    &Zdata,&LDZ,WORK.mData.getDataPointer(),RWORK.data(),IWORK.data(),IFAIL.data(),&INFO);

    if(INFO != 0)
    {
    	std::stringstream sout;
    	sout << "\nZHPEVX \nError INFO = " << INFO << "\n";
    	throw std::runtime_error(sout.str());
    }

    eigValues.resize(M);
    return M;
	}


	SCC::LapackMatrixCmplx16      AP; // For storing packed matrix in packed Hermitian form

    SCC::LapackMatrixCmplx16    WORK;
    std::vector<double>        RWORK;
    std::vector<long>          IWORK;
    std::vector<long>          IFAIL;


};


} // SCC namespace

#endif /* COMPLEXEIG_ZHPEVX_H_ */
