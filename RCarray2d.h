#include <vector>

#ifndef RC_ARRAY_2D_
#define RC_ARRAY_2D_

//
// Class RCarray2d is a templated container class for storing a 2D array
// of specified data type by columns. This class is used by the
// RayleighChebeyshev eigensystem procedures.
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
# Copyright 2012-2023 Chris Anderson
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
template <typename Dtype > class RCarray2d
{
public:
	RCarray2d()
	{
	arrayData.clear();
	rowSize = 0;
	colSize = 0;
	};

    RCarray2d(const RCarray2d& A)
	{
    if(arrayData.size() == 0) {initialize(); return;}
	initialize(A);
	}


    RCarray2d(long rowSize, long colSize)
	{
	initialize(rowSize,colSize);
	}

	virtual ~RCarray2d()
	{}

	void initialize()
	{
	arrayData.clear();
	rowSize = 0;
    colSize = 0;
	}

    void initialize(long rowSize, long colSize)
    {
    arrayData.resize(rowSize*colSize,0.0);
    this->rowSize = rowSize;
    this->colSize = colSize;
    }

    void initialize(const RCarray2d& A)
    {
    rowSize   = A.rowSize;
    colSize   = A.colSize;
    arrayData.clear();
    arrayData = A.arrayData;
    }

    long getRowSize() const
    {
    return rowSize;
    }

    long getColSize() const
    {
    return colSize;
    }

    inline Dtype&  operator()(long i1, long i2)
    {
    return arrayData[i1  + i2*rowSize];
    };

    const inline Dtype&  operator()(long i1, long i2) const
    {
    return arrayData[i1 + i2*rowSize];
    };

    void operator=(const RCarray2d& A)
    {
    if((rowSize != A.rowSize)||(colSize != A.colSize)){initialize(A);}
    else{arrayData = A.arrayData;}
    }

    Dtype* getDataPointer(){return arrayData.data();};

    const Dtype* getDataPointer() const {return arrayData.data();};


	std::vector<Dtype> arrayData;
	long                  rowSize;
	long                  colSize;
};


#endif

