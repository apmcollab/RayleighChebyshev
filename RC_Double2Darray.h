#ifndef _RC_Double2Darray_
#define _RC_Double2Darray_

//
// Class RC_Double2Darray is a container class for storing a 2D array
// of doubles by rows. This class is used by the RayleighChebeyshev
// eigensystem procedure.
//
// Since minimal functionality is required for the 2D array in the
// RayleighChebyshev procedure, this class is used to avoid
// external dependencies if a  "full-bodied' two dimensional array class
// were used.
//
// No bounds checking is performed.
//
// Author Chris Anderson March, 7, 2012
//
/*
#############################################################################
#
# Copyright 2012-2015 Chris Anderson
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
class RC_Double2Darray
{
public:
	RC_Double2Darray()
	{
	dataPtr = 0;
	rowSize = 0;
	colSize = 0;
	};

    RC_Double2Darray(const RC_Double2Darray& A)
	{
    dataPtr = 0;
	initialize(A);
	}


    RC_Double2Darray(long rowSize, long colSize)
	{
	dataPtr = 0;
	initialize(rowSize,colSize);
	}

	virtual ~RC_Double2Darray()
	{
	destroyData();
	}

	void initialize()
	{
	destroyData();
	dataPtr = 0;
	rowSize = 0;
    colSize = 0;
	}

    void initialize(long rowSize, long colSize)
    {
    destroyData();
    this->rowSize = rowSize;
    this->colSize = colSize;
    allocateData();
    }

    void initialize(const RC_Double2Darray& A)
    {
    destroyData();
    rowSize = A.rowSize;
    colSize = A.colSize;
    initialize(rowSize,colSize);
    if(dataPtr != 0)
    {for(long k = 0; k < rowSize*colSize; k++) {dataPtr[k] = A.dataPtr[k];}}
    }


    inline double&  operator()(long i1, long i2)
    {
    return *(dataPtr +  i2  + i1*colSize);
    };

    inline double&  operator()(long i1, long i2) const
    {
    return *(dataPtr +  i2 + i1*colSize);
    };

    void operator=(const RC_Double2Darray& A)
    {
    if((rowSize != A.rowSize)||(colSize != A.colSize))
    {
    initialize(A);
    }
    else
    {
    if(dataPtr != 0)
    {for(long k = 0; k < rowSize*colSize; k++) {dataPtr[k] = A.dataPtr[k];}}
    }
    }

    double* getDataPointer(){return dataPtr;};

    const double* getDataPointer() const {return dataPtr;};

    void allocateData()
    {
    if((rowSize > 0)&&(colSize > 0)) {dataPtr = new double[rowSize*colSize];}
    else                             {dataPtr = 0;}
    }

    void destroyData()
    {
    if(dataPtr != 0) delete [] dataPtr;
    dataPtr = 0;
    }

	double* dataPtr;
	long    rowSize;
	long    colSize;
};


#endif

