#ifndef Clock_IT_
#define Clock_IT_
//
// This version of Class ClockIt can be used on either a Linux/Unix or MS system 
// to time portions of code. 
//
// The class depends upon calls to the gettimeofday(..) (Linux: in <sys    ime.h> ) or the
// QueryPerformanceCounter (MS : in <Windows.h>) functions. 
//
// Author: Chris Anderson. Version date: Mon 16 Feb 2015 09:03:58 AM PST 
//
// Updates: 
// 03/15/2020  : Updated constructor to inialize LARGE_INTEGER variables in MS version (CRA)
// 
// Usage Snippet:

/*
#include "Timing/ClockIt.h"
int main()
{

ClockIt clock1;              // declare ClockIt instance

clock1.start();              // start the instance (reads the clock)

// === Code to be timed === //  // carry out computation

clock1.stop();               // stop the clock (reads the clock again)

cout << "Time elapsed in Milli-seconds: " << clock1.getMilliSecElapsedTime() << endl;

}
*/

/*
#############################################################################
#
# Copyright  2015-2020 Chris Anderson
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

#ifndef _MSC_VER
//
// This version of Class ClockIt can be used on a Linux/Unix system to time portions of code. It
// depends upon calls to the gettimeofday(..) function in sys    ime.h to read the system
// clock.
//
#include <sys/time.h>

class ClockIt
{
public:

    ClockIt()
    {
    elapsedTime = 0; 
    }

    ClockIt(const ClockIt& clock)
    {
    elapsedTime = clock.elapsedTime;
    t1          = clock.t1;
    t2          = clock.t2;
    }

    //
    // Reads the clock 
    //

    inline void start() {gettimeofday(&t1, 0);}

    //
    // Reads the clock again, and computes the 
    // the elapsed time. 
    //
    // Returns the time in milliseconds since start() called
    //

    inline double stop()
    {
    gettimeofday(&t2, 0);
    //
    // Compute elapsed time in milliseconds
    //
    elapsedTime  = (t2.tv_sec  - t1.tv_sec)  * 1000.0;    // seconds to milliseconds
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;    // microseconds to milliseconds
    return elapsedTime;
    }

    double getMilliSecElapsedTime()
    {return elapsedTime;}

    double getMicroSecElapsedTime()
    {return elapsedTime*1000.0;}

    double getSecElapsedTime()
    {return elapsedTime/1000.0;}

    timeval t1, t2;
    double elapsedTime;
};
#else
//
// This version of Class ClockIt can be used on a MS system to time portions of code. It
// depends upon calls to the QPC timer function to read the system
// clock. 
// 
#include <Windows.h>

class ClockIt
{
public:

    ClockIt()
    {
        elapsedTime         = 0;
        StartingTime        = { 0,0 };
        EndingTime          = { 0,0 };
        ElapsedMicroseconds = { 0,0 };
        Frequency           = { 0,0 };
    }

    ClockIt(const ClockIt& clockIt)
    {
        elapsedTime = clockIt.elapsedTime;
        t1 = clockIt.t1;
        t2 = clockIt.t2;

        StartingTime        = clockIt.StartingTime;
        EndingTime          = clockIt.EndingTime;
        ElapsedMicroseconds = clockIt.ElapsedMicroseconds;
        Frequency           = clockIt.Frequency;
    }

    //
    // Reads the clock 
    //

    inline void start()
    {
        QueryPerformanceFrequency(&Frequency);
        QueryPerformanceCounter(&StartingTime);
    }

    //
    // Reads the clock again, and computes the 
    // the elapsed time. 
    //
    // Returns the time in milliseconds since start() called
    //

    inline double stop()
    {
        QueryPerformanceCounter(&EndingTime);
        ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
        //
        // Compute elapsed time in milliseconds
        //
        ElapsedMicroseconds.QuadPart *= 1000000;
        ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
        elapsedTime = (double)(ElapsedMicroseconds.QuadPart)/1000.0;
        return elapsedTime;
    }

    double getMilliSecElapsedTime()
    {
        return elapsedTime;
    }

    double getMicroSecElapsedTime()
    {
        return elapsedTime*1000.0;
    }

    double getSecElapsedTime()
    {
        return elapsedTime / 1000.0;
    }

    timeval t1, t2;
    double elapsedTime;

    LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
    LARGE_INTEGER Frequency;
};

#endif

#endif
