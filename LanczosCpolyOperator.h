//
// LanczosCpolyOperator.h
//
// Instances of this class are scaled Lanczos "C" polynomial operators
// The apply(...) member function of this class applies the Lanczos C
// polynomial in the associated operator to a vector.
//
// The associated operator is referenced via a pointer. Instances
// of the associated operator are managed externally.
//
// This is a templated class with respect to both vector and operator
// types.
//
// The apply routine is not multi-threaded, as this is typically
// done externally e.g. the application of the operator
// to a block of vectors is carried out by multi-threading the
// loop over each vector with each thread being associated with
// a separate instance of a LanczosCpolyOperator.
//
//
// Chris Anderson 2005
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

   void apply(Vtype& V)

   ############################################################################
*/
/*
#############################################################################
#
# Copyright 2009-2023 Chris Anderson
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


#include <cmath>
#include <stdexcept>
#include "LanczosCpoly.h"

#ifndef LANCZOS_C_POLY_OPERATOR_
#define LANCZOS_C_POLY_OPERATOR_

template <class Vtype, class Otype>
class LanczosCpolyOperator
{
public:

LanczosCpolyOperator()
{
      lanczosCpoly.initialize();
      Op                  = nullptr;
      this->vnPtr         = nullptr;
      this->vnm1Ptr       = nullptr;
      this->vnm2Ptr       = nullptr;
      this->vTmpPtr       = nullptr;
};

LanczosCpolyOperator(const LanczosCpolyOperator& cPolyOp)
{
	  initialize();
	  if(cPolyOp.Op == nullptr) {return;}

      lanczosCpoly.initialize(cPolyOp.lanczosCpoly);
      Op  = cPolyOp.Op;
};

LanczosCpolyOperator(long polyDegree, long repetitionFactor, 
double  lambdaMax,  double shift, Otype& Op)
{
    lanczosCpoly.initialize(polyDegree, repetitionFactor, lambdaMax, shift);

    this->Op                  = &Op;
    this->vnPtr               = nullptr;
    this->vnm1Ptr             = nullptr;
    this->vnm2Ptr             = nullptr;
    this->vTmpPtr             = nullptr;
}

void initialize()
{
	  lanczosCpoly.initialize();
      Op = nullptr;

      if(vnPtr   != nullptr)  {delete     vnPtr;}
      if(vnm1Ptr != nullptr)  {delete   vnm1Ptr;}
      if(vnm2Ptr != nullptr)  {delete   vnm2Ptr;}

      this->vnPtr         = nullptr;
      this->vnm1Ptr       = nullptr;
      this->vnm2Ptr       = nullptr;
      this->vTmpPtr       = nullptr;
}


// The following two member functions are used together to
// associate an operator with a LanczosCpolyOperator class
// instance separately from the Lanczos C polynomial
// parameters.
//
void initialize(Otype& Op)
{
      lanczosCpoly.initialize();
      this->Op = &Op;
};

void setLanczosCpolyParameters(long polyDegree, long repetitionFactor,
double  lambdaMax, double shift)
{
	if(this->Op == nullptr)
	{
		std::string errMsg = "\n     LanczosCpolyOperator : setting Lanczos C poly parameters before \n";
	                errMsg += "\n    associating an operator with the instance. \n";
		throw std::runtime_error(errMsg);
	}

	lanczosCpoly.initialize(polyDegree, repetitionFactor, lambdaMax, shift);
}



void initialize(long polyDegree, long repetitionFactor, 
double  lambdaMax, double shift, Otype& Op)
{
    lanczosCpoly.initialize(polyDegree, repetitionFactor, lambdaMax, shift);

    this->Op = &Op;
    
    if(vnPtr   != nullptr) {delete   vnPtr;}
    if(vnm1Ptr != nullptr) {delete   vnm1Ptr;}
    if(vnm2Ptr != nullptr) {delete   vnm2Ptr;}

    this->vnPtr         = nullptr;
    this->vnm1Ptr       = nullptr;
    this->vnm2Ptr       = nullptr;
    this->vTmpPtr       = nullptr;
}



~LanczosCpolyOperator(void)
{
    if(vnPtr   != nullptr) {delete   vnPtr;}
    if(vnm1Ptr != nullptr) {delete   vnm1Ptr;}
    if(vnm2Ptr != nullptr) {delete   vnm2Ptr;}
};

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
// sigma          = applied shift with the constraint that sigma + lambda > 0 for all lambda
// A'             = (A + sigma)/[(lambdaMax + sigma)/UpperXstar]
//
// The apply(Vtype& v) operator of this class applies the operator
// Pm(A') to the input vector v. 
//
#ifndef VBLAS_
void apply(Vtype& v)
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

    // No gap : matrix is a multiple of the identity

    if(std::abs(shift + lambdaMax) <= 1.0e-15*std::abs(lambdaMax))
    {
    	Op->apply(v);
    	return;
    }

    if(vnPtr   == nullptr) {vnPtr   = new Vtype(v);}
    if(vnm1Ptr == nullptr) {vnm1Ptr = new Vtype(v);}
    if(vnm2Ptr == nullptr) {vnm2Ptr = new Vtype(v);}

    for(repCount = 1; repCount <= repetitionFactor; repCount++)
    {
        // 
        // initialization of recurrance
        //
        *vnm2Ptr = v;     
        *vnm1Ptr = v;     
        Op->apply(*vnm1Ptr);
        *vnm1Ptr  *=  gamma1;
        *vnm1Ptr  += *vnm2Ptr;
        *vnm1Ptr  *=  gamma2;
        // 
        // general recurrance
        //
        for(k = 2; k <= polyDegree; k++)
        {
        *vnPtr = *vnm1Ptr;
        Op->apply(*vnPtr);
        *vnPtr *=  gamma1;
        *vnPtr += *vnm1Ptr;
        *vnPtr *=  gamma2;

        *vnm2Ptr *= -1.0;
        *vnPtr   += *vnm2Ptr;
        
        // 
        // swap pointers to implicitly shift the 
        // indices of the iteration vectors
        //

        vTmpPtr = vnm2Ptr;
        vnm2Ptr = vnm1Ptr;
        vnm1Ptr = vnPtr;
        vnPtr   = vTmpPtr;
        }
     
        *vnm1Ptr *= (1.0/double(polyDegree+1));
        v         = *vnm1Ptr;
     }
}
#endif
#ifdef VBLAS_

void apply(Vtype& v)
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

    // No gap so matrix is a multiple of the identity so multiplication
    // by operator.

    if(std::abs(shift + lambdaMax) <= 1.0e-15*std::abs(lambdaMax))
    {
    	Op->apply(v);
    	return;
    }


    if(vnPtr   == nullptr) {vnPtr   = new Vtype(v);}
    if(vnm1Ptr == nullptr) {vnm1Ptr = new Vtype(v);}
    if(vnm2Ptr == nullptr) {vnm2Ptr = new Vtype(v);}

     for(repCount = 1; repCount <= repetitionFactor; repCount++)
     {
        //
        // initialization of recurrance
        //
        *vnm2Ptr = v;
        *vnm1Ptr = v;
        Op->apply(*vnm1Ptr);

        vnm1Ptr->axpby(gamma2,*vnm2Ptr,gamma1*gamma2);

        //
        // general recurrance
        //
        for(k = 2; k <= polyDegree; k++)
        {
        *vnPtr = *vnm1Ptr;
        Op->apply(*vnPtr);
        vnPtr->axpby(gamma2,*vnm1Ptr,gamma1*gamma2);
        vnPtr->axpy(-1.0,*vnm2Ptr);
        //
        // swap pointers to implicitly shift the
        // indices of the iteration vectors
        //

        vTmpPtr = vnm2Ptr;
        vnm2Ptr = vnm1Ptr;
        vnm1Ptr = vnPtr;
        vnPtr   = vTmpPtr;
        }

        vnm1Ptr->scal((1.0/double(polyDegree+1)));
        v = *vnm1Ptr;
    }
}
#endif

	LanczosCpoly lanczosCpoly;

     Otype* Op;
     Vtype* vnPtr;
     Vtype* vnm1Ptr;   
     Vtype* vnm2Ptr;    
     Vtype* vTmpPtr;
};

#endif


 
