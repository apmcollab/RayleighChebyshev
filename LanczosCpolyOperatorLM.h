//
//              LanczosCpolyOperatorLM.h
//
//     LanczosCpolyOperator for Large Matrix problems
//
// This version of LanczosCpolyOperator is a multi-threaded implementation
// that uses only one copy of the operator whose eigensystem is being computed,
// and hence facilitates the construction of the eigensystem of very high dimensional
// linear operators. In order to exploit multi-threading of the application of the
// polynomial in the operator as implemented in this class, the operator
// class's apply(std::vector<Vtype>& V) implementation must internally utilize
// multi-threading, i.e. the apply(...) is not nested within a parallellized loop.
//
// Instances of this class are scaled Lanczos "C" polynomial operators
// The apply(...) member function of this class applies the Lanczos C
// in the associated operator to a vector.
//
// This is a templated class with respect to both vector and operator
// types.
//
// The apply routine is not multi-threaded, as this is typically
// done externally e.g. the application of the operator
// to a block of vectors is carried out by multi-threading the
// loop over each vector with each thread being associated with
// a separate instance of a LanczosCpolyOperatorLM.
//
// Chris Anderson 2022
//
// Updated : Refactored code by removing duplicate functionality
// now provided by the internal instance of LanczosCpoly, and
// updated documentation.
/*
   The minimal functionality required of the classes
   that are used in this template are

   Vtype
   ---------
   A vector class with the following member functions:

   Vtype()                            (null constructor)
   Vtype(const Vtype&)                (copy constructor)

   initialize()                       (null initializer)
   initialize(const Vtype&)           (copy initializer)

   operator =                         (duplicate assignemnt)
   operator +=                        (incremental addition)
   operator -=                        (incremental subtraction)
   operator *=(double alpha)          (scalar multiplication)

   if VBLAS_ is defined, then the Vtype class must also possess member functions

   void   scal(double alpha)                                (scalar multiplication)
   void   axpy(double alpha,const Vtype& x)                 (this = this + alpah*x)
   void   axpby(double alpha,const Vtype& x, double beta)   (this = alpha*x + beta*this)

   If OpenMP is defined, then the vector class should NOT SET any class or static
   variables of the vector class arguments to copy, dot, or axpy. Also,
   no class variables or static variables should be set by nrm2().


   ############################################################################

   Otype
   ----------

   An operator class with the following member function:

   void apply(std::vector<Vtype>& Varray)

   which applies the operator to all vectors in the argument Varray and returns the result in Varray.

   To take advantage of multi-threading this apply operator should multi-thread
   internally, i.e. use a multi-threaded loop to apply the operator to each of the
   individual vectors.

   ############################################################################
*/
/*
#############################################################################
#
# Copyright 2009-2022 Chris Anderson
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
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include "LanczosCpoly.h"

#ifndef LANCZOS_C_POLY_OPERATOR_LM_
#define LANCZOS_C_POLY_OPERATOR_LM_

template <class Vtype, class Otype>
class LanczosCpolyOperatorLM
{
public:

LanczosCpolyOperatorLM()
{
      lanczosCpoly.initialize();
      Op                  = 0;
};

LanczosCpolyOperatorLM(long polyDegree, long repetitionFactor,
double  lambdaMax,  double shift, Otype& Op)
{
    lanczosCpoly.initialize(polyDegree, repetitionFactor, lambdaMax, shift);

    this->Op                  = &Op;
}

void initialize(long polyDegree, long repetitionFactor, 
double  lambdaMax, double shift, Otype& Op)
{
    lanczosCpoly.initialize(polyDegree, repetitionFactor, lambdaMax, shift);

    this->Op                  = &Op;
}

~LanczosCpolyOperatorLM(void)
{};

void setShift(double shift)
{
    lanczosCpoly.setShift(shift);
}

void setPolyDegree(long polyDegree)
{
	lanczosCpoly.setPolyDegree(polyDegree);
}

void setRepetitionFactor(long repetitionFactor)
{
    lanczosCpoly.setRepetitionFactor(repetitionFactor);
}


//
// If 
// lambdaMax      = maximal  eigenvalue of A
// sigma          = applied shift with the constraint that sigma + lambda >= 0 for all lambda
// A'             = (A + sigma)/[(lambdaMax + sigma)/UpperXstar]
//
// The apply(Vtype& v) operator of this class applies the operator
// Pm(A') to the input vector v. 
//


#ifndef VBLAS_
void apply(std::vector<Vtype>& vArray)
{
    long repCount;
    long k;

    double UpperXStar       = lanczosCpoly.UpperXStar;
    double shift            = lanczosCpoly.shift;
    double lambdaMax        = lanczosCpoly.lambdaMax;
    long   polyDegree       = lanczosCpoly.polyDegree;
    long   repetitionFactor = lanczosCpoly.repetitionFactor;


    double starFactor = UpperXStar; // 1.0 - XStar;
    double rhoB       = lambdaMax/starFactor + shift/starFactor;
    double gamma1     = -2.0/(rhoB - 2.0*shift);
    double gamma2     =  2.0 - (4.0*shift)/rhoB;

    size_t vSize = vArray.size();

    vn.resize(vSize);
    vnm1.resize(vSize);
    vnm2.resize(vSize);

#ifdef _OPENMP
       #pragma omp parallel for
#endif
	for(size_t p = 0; p < vSize; p++)
	{
       vn[p].initialize(vArray[p]);
       vnm1[p].initialize(vArray[p]);
       vnm2[p].initialize(vArray[p]);
	}

    vnArrayPtr   = &vn;
    vnm1ArrayPtr = &vnm1;
    vnm2ArrayPtr = &vnm2;



    for(repCount = 1; repCount <= repetitionFactor; repCount++)
    {
        // 
        // initialization of recurrance
        //
#ifdef _OPENMP
       #pragma omp parallel for
#endif
        for(size_t p = 0; p < vSize; p++)
        {
        (*vnm2ArrayPtr)[p] = vArray[p];
        (*vnm1ArrayPtr)[p] = vArray[p];
        }

        Op->apply(*vnm1ArrayPtr);

#ifdef _OPENMP
       #pragma omp parallel for
#endif
        for(size_t p = 0; p < vSize; p++)
        {
        (*vnm1ArrayPtr)[p]  *=  gamma1;
        (*vnm1ArrayPtr)[p]  += (*vnm2ArrayPtr)[p];
        (*vnm1ArrayPtr)[p]  *=  gamma2;
        }
        // 
        // general recurrance
        //
        for(k = 2; k <= polyDegree; k++)
        {

#ifdef _OPENMP
       #pragma omp parallel for
#endif
        for(size_t p = 0; p < vSize; p++)
        {
        (*vnArrayPtr)[p]  =  (*vnm1ArrayPtr)[p];
        }

        Op->apply(*vnArrayPtr);
        
#ifdef _OPENMP
       #pragma omp parallel for
#endif
        for(size_t p = 0; p < vSize; p++)
        {
        (*vnArrayPtr)[p]   *=  gamma1;
        (*vnArrayPtr)[p]   += (*vnm1ArrayPtr)[p];
        (*vnArrayPtr)[p]   *=  gamma2;
        (*vnm2ArrayPtr)[p] *= -1.0;
        (*vnArrayPtr)[p]   += (*vnm2ArrayPtr)[p];
        }

        // 
        // swap pointers to implicitly shift the 
        // indices of the iteration vectors
        //

        vTmpArrayPtr = vnm2ArrayPtr;
        vnm2ArrayPtr = vnm1ArrayPtr;
        vnm1ArrayPtr = vnArrayPtr;
        vnArrayPtr   = vTmpArrayPtr;
        }
     
#ifdef _OPENMP
       #pragma omp parallel for
#endif
        for(size_t p = 0; p < vSize; p++)
        {
        (*vnm1ArrayPtr)[p] *= (1.0/double(polyDegree+1));
        vArray[p]      = (*vnm1ArrayPtr)[p];
        }
     }
}
#endif


#ifdef VBLAS_

void apply(std::vector<Vtype>& vArray)
{
    long repCount;
    long k;

    double UpperXStar       = lanczosCpoly.UpperXStar;
    double shift            = lanczosCpoly.shift;
    double lambdaMax        = lanczosCpoly.lambdaMax;
    long   polyDegree       = lanczosCpoly.polyDegree;
    long   repetitionFactor = lanczosCpoly.repetitionFactor;


    double starFactor = UpperXStar; // 1.0 - XStar;
    double rhoB       = lambdaMax/starFactor + shift/starFactor;
    double gamma1     = -2.0/(rhoB - 2.0*shift);
    double gamma2     =  2.0 - (4.0*shift)/rhoB;

    size_t vSize = vArray.size();

    vn.resize(vSize);
    vnm1.resize(vSize);
    vnm2.resize(vSize);

#ifdef _OPENMP
       #pragma omp parallel for
#endif
	for(size_t p = 0; p < vSize; p++)
	{
       vn[p].initialize(vArray[p]);
       vnm1[p].initialize(vArray[p]);
       vnm2[p].initialize(vArray[p]);
	}

    vnArrayPtr   = &vn;
    vnm1ArrayPtr = &vnm1;
    vnm2ArrayPtr = &vnm2;


    for(repCount = 1; repCount <= repetitionFactor; repCount++)
     {

        //
        // initialization of recurrance
        //
#ifdef _OPENMP
       #pragma omp parallel for
#endif
        for(size_t p = 0; p < vSize; p++)
        {
        (*vnm2ArrayPtr)[p] = vArray[p];
        (*vnm1ArrayPtr)[p] = vArray[p];
        }

        Op->apply(*vnm1ArrayPtr);


        for(size_t p = 0; p < vSize; p++)
        {
        (*vnm1ArrayPtr)[p].axpby(gamma2,(*vnm2ArrayPtr)[p],gamma1*gamma2);
        }

        //
        // general recurrance
        //
        for(k = 2; k <= polyDegree; k++)
        {

#ifdef _OPENMP
       #pragma omp parallel for
#endif
        for(size_t p = 0; p < vSize; p++)
        {
        (*vnArrayPtr)[p] = (*vnm1ArrayPtr)[p];
        }

        Op->apply(*vnArrayPtr);

#ifdef _OPENMP
       #pragma omp parallel for
#endif
        for(size_t p = 0; p < vSize; p++)
        {
        (*vnArrayPtr)[p].axpby(gamma2,(*vnm1ArrayPtr)[p],gamma1*gamma2);
        (*vnArrayPtr)[p].axpy(-1.0,(*vnm2ArrayPtr)[p]);
        }
        //
        // swap pointers to implicitly shift the
        // indices of the iteration vectors
        //

        vTmpArrayPtr = vnm2ArrayPtr;
        vnm2ArrayPtr = vnm1ArrayPtr;
        vnm1ArrayPtr = vnArrayPtr;
        vnArrayPtr   = vTmpArrayPtr;
        }

#ifdef _OPENMP
       #pragma omp parallel for
#endif
        for(size_t p = 0; p < vSize; p++)
        {
        (*vnm1ArrayPtr)[p] *= (1.0/double(polyDegree+1));
        vArray[p]      = (*vnm1ArrayPtr)[p];
        }
    }
}
#endif

	LanczosCpoly lanczosCpoly;

     long   polyDegree;
     long   repetitionFactor;
     double lambdaMax;
     double shift;
     double XStar;
     double UpperXStar;
    
     Otype* Op;

     std::vector<Vtype> vn;
     std::vector<Vtype> vnm1;
     std::vector<Vtype> vnm2;


     std::vector<Vtype>* vnArrayPtr;
     std::vector<Vtype>* vnm1ArrayPtr;
     std::vector<Vtype>* vnm2ArrayPtr;
     std::vector<Vtype>* vTmpArrayPtr;
};

#endif


 
