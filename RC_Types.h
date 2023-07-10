/*
 * RC_Types.h
 *
 *  Created on: May 31, 2023
 *      Author: anderson
 */

// A class for storing named enums for the RayleighChebyshev programs

/*
#############################################################################
#
# Copyright 2005-2023 Chris Anderson
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
#ifndef RC_TYPES_H_
#define RC_TYPES_H_

class RC_Types
{
public:

	enum StopCondition {DEFAULT, COMBINATION, EIGENVALUE_ONLY, RESIDUAL_ONLY};
};




#endif /* RC_TYPES_H_ */
