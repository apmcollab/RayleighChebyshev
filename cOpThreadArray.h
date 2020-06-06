#ifndef  C_OP_THREADING_ARRAY_
#define  C_OP_THREADING_ARRAY_

#include <omp.h>
#include "LanczosCpolyOperator.h"
/**
 *
 *
 *  cOpThreadArray : A helper class to manage copies of LanczosCpolyOperators
 *                   for multi-threaded execution of MinIntervalEigFinder
 *
 */
//
// This cOpThreadArray is created to hold copies of the LanczosCpolyOperator
// class for separate threads.  This class couldn't be a generic one since the
// LanczosCpolyOperator operator class contains references to the underlying
// operator whose eigensystem is being computed, and these references need to
// point to copies of the operator used for separate threads.
//
// The idea behind the class:
//
// If M is the number of threads, then outside the eigensystem iteration loop,
// a cOpThreadArray instance is constructed that has M copies of the LanczosCpolyOperator
// and M copies of the operator whose eigensystem is being computed.
//
// Within the iteration, the initialize member function is used to initialize
// copies of LanczosCpolyOperator appropriate for the particular iteration
// (each copy still using references to the copies of the operator)
//
// The cOpThreadArray is declared firstprivate, so a shallow copy of the class
// variables is created for each thread. This is satisfactory, since this just
// copies the array of references to the collection of LanczosCpolyOperator
// instances. The pth thread uses the instance that is referenced by the
// the pth element of the array of LanczosCpolyOperator pointers.
//
// This construction essentially mimics an MPI approach, where there would
// have separate copies of the operator associated with each processor.
//
// The tricky part is figuring out how to construct the new and delete operators
// so one doesn't get memory leaks or segmentation faults.
//
// This code will work fine with the existing LanczosCpolyOperator
// class, however, the operator class (of type Otype) must also now have a
// copy constructor --- e.g. we need to be able to replicate the operator for
// the additional threads.
//
// Author: Chris Anderson,
// Initial construction : Aug. 29, 2009
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
template <class Vtype, class Otype > class cOpThreadArray
{
public :

cOpThreadArray(Otype& Op)
{
	numThreads = omp_get_max_threads();
	cOpArray   = new LanczosCpolyOperator< Vtype , Otype >*[numThreads];
	OpArray    = new Otype*[numThreads];

	//
	// Create instances of the LanczosCpolyOperator for each thread
	//

	for(long k = 0; k < numThreads; k++)
	{
		cOpArray[k] = new LanczosCpolyOperator< Vtype , Otype >();
	}

	//
	// Create copies of the operator for threads >= 1
	//

	OpArray[0] = &Op;
	for(long k = 1; k < numThreads; k++)
	{
		OpArray[k]  = new Otype(Op);
	}
}
//
// Shallow copy so local thread version contain references to the
// objects instantiated by the 0th thread
//
cOpThreadArray(const cOpThreadArray& oArray)
{
	numThreads = oArray.numThreads;
	cOpArray   = new LanczosCpolyOperator< Vtype , Otype >*[numThreads];

	for(long k = 0; k < numThreads; k++)
	{
	cOpArray[k]= oArray.cOpArray[k];
	}
	//
	// Shallow instance destroy
	//
	destroyFlag = 1;
}

void initialize(long polyDegree, long repetitionFactor,double lambdaBound,
double shift)
{
	numThreads = omp_get_max_threads();
	for(long k = 0; k < numThreads; k++)
	{
    cOpArray[k]->initialize(polyDegree,repetitionFactor,lambdaBound,shift,*OpArray[k]);
	}
	//
	// deep instance destroy
	//
	destroyFlag = 2;
}

~cOpThreadArray()
{
    if(destroyFlag == 0)                // Do nothing
    {}
    else if(destroyFlag == 1)           // Delete the array of pointers to the objects
	{
	delete [] cOpArray;
	}
    else                                 // Delete all temporary objects created for
    {                                    // parallel threaded constructs and array of pointers

    delete cOpArray[0];
	for(long k = 1; k < numThreads; k++)
	{
		// delete thread copy of operator used by
		// the LanczosCpolyOperator

		delete OpArray[k];

		// delete the thread copy of
		// the LanczosCpolyOperator

		delete cOpArray[k];
	}

	//
	// delete the array of LanczosCpolyOperator pointers
	//
	delete [] cOpArray;
	delete [] OpArray;
    }
}

void apply(Vtype& V)
{
	cOpArray[omp_get_thread_num()]->apply(V);
}

LanczosCpolyOperator< Vtype , Otype > ** cOpArray;
                                 Otype**  OpArray;
long numThreads;
int destroyFlag;
};

#endif
