#ifndef _OpStar_
#define _OpStar_
//
//###################################################################################
//                                 Class OpStar
//###################################################################################
//
// Given an existing operator the purpose of this class is to create
// an operator that implements
//
// OpStar = signOp*(Op + shift)^p
//
// where p = 1 or p = 2.
//
// The default constructor/destructor creates/destroys a copy of the
// operator Op. This convention is followed so that when arrays of
// the operator are created to be used in a multi-threaded loop,
// separate copies of the operator are used for each thread.
//
// The OpStarRef class (whose definition is given below) is an equivalent
// class that uses pointers, rather than instance copies,
// to the underlying operator.
//
//###################################################################################
/*
#############################################################################
#
# Copyright 2005-2015 Chris Anderson
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
template < class Vtype, class Otype > class OpStar
{
    public:

	OpStar()
	{
	OpPtr    = 0;
	vTempPtr = 0;
	initialize();
	}

	OpStar(Otype& oP,int signOp, double shift, bool squareFlag = false)
	{
	OpPtr    = 0;
	vTempPtr = 0;
	initialize(oP,signOp,shift,squareFlag);
	}

	OpStar(const OpStar & oPstar)
	{
	OpPtr    = 0;
	vTempPtr = 0;
	if(oPstar.OpPtr == 0)
	{
	initialize();
	return;
	}
	initialize(*(oPstar.OpPtr),oPstar.signOp,oPstar.shift,oPstar.squareFlag);
	}

	~OpStar()
	{
    if(OpPtr != 0)    {delete OpPtr;}
    if(vTempPtr != 0) {delete vTempPtr;}
	}

	void initialize()
	{
	if(OpPtr != 0)    {delete OpPtr;}
	if(vTempPtr != 0) {delete vTempPtr;}
	OpPtr    = 0;
	vTempPtr = 0;

	shift      = 0.0;
	signOp     =  +1;
    squareFlag = false;
	}

	void initialize(Otype& oP, int signOp, double shift, bool squareFlag = false)
	{
	if(OpPtr != 0) {delete OpPtr;}
	this->OpPtr  = new Otype(oP);

	this->shift      = shift;
	this->signOp     = signOp;
    this->squareFlag = squareFlag;
	}

	void setSignShiftAndSquareFlag(int signOp, double shift, bool squareFlag)
	{
	this->shift      = shift;
	this->signOp     = signOp;
    this->squareFlag = squareFlag;
	}
//
//  In this forward operator, if a shift is used, then the evaluation
//  requires a vector temporary. This temporary is instantiated on the
//  first call and then re-used.
//
	void applyForwardOp(Vtype& V)
	{
	double smallShiftIgnoreValue = 1.0e-15;

	bool   useShift = false;
	if(fabs(shift) > smallShiftIgnoreValue)
	{
	useShift = true;
	}

	// Computing shift*V, and initializing a temporary if needed.
	//
	// To do: If vector blas are available, use those operations

	if(useShift)
	{
		if(vTempPtr == 0)
		{vTempPtr = new Vtype(V);}          // Create a copy if vTemp doesn't exist
		else
		{*vTempPtr = V;}                     // Assignment if vTemp does exist

	*vTempPtr *= shift;
	}

	OpPtr->applyForwardOp(V);
	if(useShift)   {V += *vTempPtr;}

	if(not squareFlag)
	{
	if(signOp < 0) {V *= -1.0;}
	return;
	}

	if(useShift)
	{
    *vTempPtr  = V;
    *vTempPtr *= shift;
    }
    OpPtr->applyForwardOp(V);

    if(useShift)
    {V += *vTempPtr;}

	if(signOp < 0) {V *= -1.0;}
	}

	Otype*     OpPtr;
	int       signOp;
	double     shift;
	bool  squareFlag;
	Vtype*  vTempPtr;
};
//
//###################################################################################
//                              Class OpStarRef
//###################################################################################
//
//
// Given an existing operator the purpose of this class is to create
// an operator that implements
//
// OpStar = signOp*(Op  + shift)^p
//
// where p = 1 or p = 2.
//
// This class is identical to OpStar, but does not use a duplicate
// operator Op internally; it only uses references to an externally
// specified operator.
//
// The copy constructor only copies the pointer to the underlying
// operator Op. The destructor does not call the destructor
// for the underlying Op.
//
//###################################################################################
//
template < class Vtype, class Otype > class OpStarRef
{
    public:

	OpStarRef()
	{
	OpPtr    = 0;
	vTempPtr = 0;
	initialize();
	}

	OpStarRef(Otype& oP,int signOp, double shift, bool squareFlag = false)
	{
	OpPtr    = 0;
	vTempPtr = 0;
	initialize(oP,signOp,shift,squareFlag);
	}

	OpStarRef(const OpStarRef & oPstar)
	{
	OpPtr    = 0;
	vTempPtr = 0;
	if(oPstar.OpPtr == 0)
	{
	initialize();
	return;
	}
	initialize(*(oPstar.OpPtr),oPstar.signOp,oPstar.shift,oPstar.squareFlag);
	}

	~OpStarRef()
	{
    if(vTempPtr != 0) {delete vTempPtr;}
	}

	void initialize()
	{
	if(vTempPtr != 0) {delete vTempPtr;}
	OpPtr    = 0;
	vTempPtr = 0;

	shift      = 0.0;
	signOp     =  +1;
    squareFlag = false;
	}

	void initialize(Otype& oP, int signOp, double shift, bool squareFlag = false)
	{
	this->OpPtr      = &oP;
	this->shift      = shift;
	this->signOp     = signOp;
    this->squareFlag = squareFlag;
	}

	void setSignShiftAndSquareFlag(int signOp, double shift, bool squareFlag)
	{
	this->shift      = shift;
	this->signOp     = signOp;
    this->squareFlag = squareFlag;
	}
//
//  In this forward operator, if a shift is used, then the evaluation
//  requires a vector temporary. This temporary is instantiated on the
//  first call and then re-used.
//
	void applyForwardOp(Vtype& V)
	{
	double smallShiftIgnoreValue = 1.0e-15;

	bool   useShift = false;
	if(fabs(shift) > smallShiftIgnoreValue)
	{
	useShift = true;
	}

	// Computing shift*V, and initializing a temporary if needed.
	//
	// To do: If vector blas are available, use those operations

	if(useShift)
	{
		if(vTempPtr == 0)
		{vTempPtr = new Vtype(V);}          // Create a copy if vTemp doesn't exist
		else
		{*vTempPtr = V;}                     // Assignment if vTemp does exist

	*vTempPtr *= shift;
	}
	OpPtr->applyForwardOp(V);
	if(useShift)   {V += *vTempPtr;}

	if(not squareFlag)
	{
	if(signOp < 0) {V *= -1.0;}
	return;
	}

	if(useShift)
	{
    *vTempPtr  = V;
    *vTempPtr *= shift;
    }
    OpPtr->applyForwardOp(V);

    if(useShift)
    {V += *vTempPtr;}

	if(signOp < 0) {V *= -1.0;}
	}

	Otype*     OpPtr;
	int       signOp;
	double     shift;
	bool  squareFlag;
	Vtype*  vTempPtr;
};

#endif
