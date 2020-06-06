//
// LaczosCpoly.h
//
// Instances are scaled Lanczos "C"  polynomials
//
// Ref: "Applied Analysis" by C. Lanczos
// Discussion starting Chap. III, Sect. 7.
//
// This class is created for the construction of instances of  "LanczosCPolyOperator"
// an operator class whose apply(..)  member function applies the C polynomial
// of a linear operator to a vector. The latter operator is an ingredient in the Rayleigh-Chebyshev
// procedure for finding the smallest  eigenvalues of a Hermitian matrix as well
// as for variants of the method of C-iterations for finding solutions
// to linear systems of equations.
//
// The formulas used in this class are slightly different
// than those used by Lanczos; specifically the
// indexing of the vectors in the recurrance used
// to apply the polynomial and the initial variable transformation.
//
// Let Pm(x) = Chebyshev polynomial of the second kind of degree m.
// We associate with Pm(x) a critical evaluation point, Xstar, defined
// to be the smallest value such that if z is any value in [Xstar,1-Xstar] 
// then ||Pm(z)|| <= ||Pm(x)|| for any x in [0,Xstar] or [1-Xstar,1]. 
//
// In order to suppress large eigenvalue components, an upper limit value, 
// UpperXstar, is chosen to be the second largest root of Pm(x). For this value we
// have UpperXstar <  1-Xstar, and so that if z is in [Xstar,UpperXstar] 
// then ||Pm(z)|| <= ||Pm(x)|| for any x in [0,Xstar]. 
//
// If 
//
// lambdaMax      = maximal value of lambda
// sigma          = applied shift with the constraint that lambda + sigma >= 0 for all lambda.
// lambda'        = (lambda + sigma)/[(lambdaMax + sigma)/UpperXstar]
//                = (lambda + sigma)/rhoB
//
// The apply(...)  operator of this class returns the value of the
// polynomial Pm(lambda').  
//
// Note that the restriction on sigma implies that if lambda < 0, sigma must
// have a positive non-zero value. Typically one sets sigma = -lambdaMin.
//
// The use of the initial transformation of lambda to lambda' insures that 
// the point x used to determine the value of Pm(lambda') corresponding
// to lambda 
//  
// x = (lambda+sigma)/rhoB   
//   = UpperXstar*[(lambda+sigma)/(lambdaMax + sigma)]
//  <= UpperXstar
//  <= (1- Xstar)
//
// This bound on x is independent of the shift sigma as long as lambda + sigma > 0.
//
// Another property of the scaling is that if we define the 
// critical lambda value
//
// lambdaStar = rhoB*(Xstar) - sigma 
//
// then for all lambda_j and lambda_k such that
//
// lambda_j < lambdaStar and lambda_k > lambdaStar
//
// Pm((lambda_j + sigma)/rhoB) > Pm((lambda_k + sigma)/rhoB)
//
// => the value of Pm(lambda_j')corresponding to lambda_j is > 
//    the value of Pm(lambda_k')corresponding to lambda_k.
//   
// Here  rhoB = [(lambdaMax + sigma)/UpperXstar] and
// lambdaStar is the inverse of Xstar under the transformation of
// lambda to lambda'.
//
// In the polydegree = 1 case, Xstar = upperXstar = .5, which,
// with the scaling used, insures that the [0,lambdaMax] is 
// transformed with the linear function that is 1 at x = 0 and
// 0 at the lambdaMax.
//
// Chris Anderson 2005
//
// Updated : Modified documentation so that the description
// more accurately represents the functionality
// of the code. June 10, 2013 CRA.
//
//
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
#include <cmath>


#ifndef LANCZOS_C_POLY_
#define LANCZOS_C_POLY_

class LanczosCpoly
{
public:

LanczosCpoly()
{
      initialize();
};

LanczosCpoly(const LanczosCpoly& P)
{
     this->polyDegree          = P.polyDegree;
     this->repetitionFactor    = P.repetitionFactor;
     this->lambdaMax           = P.lambdaMax;
     this->shift               = P.shift;
     this->XStar               = P.XStar;
     this->UpperXStar          = P.UpperXStar;
}

LanczosCpoly(long polyDegree, long repetitionFactor, 
double  lambdaMax,  double shift)
{
     this->polyDegree          = polyDegree;
     this->repetitionFactor    = repetitionFactor;
     this->lambdaMax           = lambdaMax;
     this->shift               = shift;
     this->XStar               = this->getXStar();
     this->UpperXStar          = this->getUpperXStar();
}

void initialize()
{
      polyDegree          = 0;
      repetitionFactor    = 0;
      lambdaMax           = 0.0;
      shift               = 0.0;
      XStar               = 0.0;
      UpperXStar          = 1.0;
}

void initialize(long polyDegree, long repetitionFactor, 
double  lambdaMax, double shift)
{
     this->polyDegree          = polyDegree;
     this->repetitionFactor    = repetitionFactor;
     this->lambdaMax           = lambdaMax;
     this->shift               = shift;
     this->XStar               = this->getXStar();
     this->UpperXStar          = this->getUpperXStar();
}

~LanczosCpoly(void){};

void setShift(double shift)
{
    this->shift = shift;
}

void setPolyDegree(long polyDegree)
{
    this->polyDegree    = polyDegree;
    this->XStar         = this->getXStar();
    this->UpperXStar    = this->getUpperXStar();
}
void setRepetitionFactor(long repetitionFactor)
{
    this->repetitionFactor    = repetitionFactor;
}

//
// If 
// lambdaMax      = maximal eigenvalue of A
// sigma          = applied shift with the constrain sigma+lambda >= 0 for all lambda
// lambda'        = (lambda + sigma)/[(lambdamax + sigma)/UpperXstar]
//
// The apply(double lambda) operator of this class returns the value of
// the polynomial Pm(lambda'). 
//
double apply(double lambda)
{
    double starFactor = UpperXStar; // 1.0 - XStar;
    double rhoB       = lambdaMax/starFactor + shift/starFactor;
    return pow(evaluatePm((lambda + shift)/rhoB),int(repetitionFactor));
}

//
// Formation of Pm(x) 
//
// C = 2 - 4*x
//
// m = polyDegree
//
// b_0   = 1                (m = 0 =  0th order)
// b_1   = C*b_0            (m = 1 =  1st order)
// b_2   = C*b_1 - b_0      (m = 2 =  2nd order)
// b_3   = C*b_2 - b_1      (m = 3 =  3rd order)
//       *
//       *
// b_m   = C*b_m-1 - b_m-2  (m     = mth order)
//
// Pm(x) = b_m/(m+1)
//
// Another form of the polynomial :
//
// define theta by sin^2(theta/2) = x
//
// then
//
// Pm(x) = sin((m+1)*x)/((m+1)*sin(x))
//
// Where m = degree of the Tchebyschev polynomial.
//
// P0(x) = 1
// P1(x) = 1 - 2*x
// P2(x) = 1 - (16/3)x + (16/3)*x^2
// 

double evaluatePm(double x)
{
     if(polyDegree == 0) return 1.0;
     if(polyDegree == 1) return 1.0 - 2.0*x;

     long k;
     double xnm1; double xnm2; double xtmp;
    // 
    // initialization of recurrance
    //
    xnm2  = 1.0;     
    xtmp  = xnm2;     
    xtmp *= x;
    xtmp *= -2.0;
    xtmp += xnm2;
    xtmp *= 2.0;
    xnm1 = xtmp;
    // 
    // general recurrance
    //
    for(k = 2; k <= polyDegree; k++)
    {
        xtmp = xnm1;
        xtmp *= x;
        xtmp *= -2.0;
        xtmp += xnm1;
        xtmp *= 2.0;

        xnm2 *= -1.0;
        xtmp += xnm2;
     
        xnm2 = xnm1;
        xnm1 = xtmp;
    }
    xnm1 *= (1.0/double(polyDegree+1));
    return xnm1;
}
//
// X* is the value in (0,1) so that if x is in (0,X*)
// then |Pm(x)| > |Pm(z)| for all z in (X*, 1-X*).
//
// Moreover, the range of a well defined inverse for Pm are
// the values of eta in [Pm(X*), infinity].
//
// For polynomial degrees 1 and 2, the values are chosen
// to provide reasonable defaults, since for these degrees
// the general selection principle for higher degree
// polynomials isn't applicable. Reasonable defaults consist
// of x*'s that are decreasing and ux*'s are slightly less
// than (1-x*) and increasing.
// 
double getXStar()
{
    if(polyDegree == 0) return 1.0;
    if(polyDegree == 1) return 0.3;

    double xp = getPmPrimeFirstRoot();
    double yp = evaluatePm(xp);
    return      evaluateInversePm(-yp);
}

double getUpperXStar()
{
    if(polyDegree == 0) return 1.0;
    if(polyDegree == 1) return 0.6;
    if(polyDegree == 2)
    {
    return ((1.0 - getXStar() + 0.6)/2.0);
    }

    return getPmNextToLastRoot();
}
//
// lambda* is that value so that if lambda_j < lambda* 
// and lambda_k > lambda* then 
//
// Pm((lambda_j + sigma)/rhoB) > Pm((lambda_k + sigma)/rhoB)
//
// where rhoB = [(lambdaMax + sigma)/upperXstar]. Moreover,
// any eigenvalues of Pm((A+sigma)/rhoB)) that are larger than
// Pm((lambda* + sigma)/rhoB) are in one to one association with
// eigenvalues of A < lambda*. Thus the dominant (in magnitude)
// eigenvalues of Pm((A+sigma)/rhoB)) correspond to the minimal
// eigenvalues of A + sigma.
//
// Note : If the minimum eigenvalue (rhoMin) for A or a shifted A > 0 , then
//        one interprets lambda* < rhoMin as implying that there are no
//        eigenvalues for which one has a well defined inverse. Specifically
//        one can't recover the eigenvalues of A from the eigenvalues of
//        the scaled C-olynomial applied to A.
//
//        It is best to have a shift = -rhoMin. If one can't get this
//        exactly, it is better to choose a shift that leads to some
//        negative eigenvalues, rather than all positive eigenvalues.
//
double getLambdaStar()
{
     double starFactor = UpperXStar;
     double rhoB       = lambdaMax/starFactor + shift/starFactor;
     return rhoB*XStar - shift;
}
//
// The range of a well defined inverse for (Pm)^repetitionFactor is 
// (Pm(X*))^(repetitionFactor) <= eta <= infinity.
// 

double getCpolyInverseRangeLowerBound()
{
    double etaLowerBound = pow(evaluatePm(XStar),int(repetitionFactor));
    return etaLowerBound;
}

double getCpolyInverseEigenvalue(double eta)
{
     eta = pow(eta,1.0/double(repetitionFactor));
     eta = evaluateInversePm(eta);
     double starFactor = UpperXStar;
     double rhoB       = lambdaMax/starFactor + shift/starFactor;
     return rhoB*eta - shift;
}


double evaluatePolynomial(double lambda)
{
    double starFactor = UpperXStar;
    double rhoB       = lambdaMax/starFactor + shift/starFactor;
    return pow(evaluatePm((lambda + shift)/rhoB),int(repetitionFactor));
}

double getPmNextToLastRoot()
{
    //
    // The secant method with a good initial guess.
    //
    double tol = 1.0e-12;

    double xa;   double fa;
    double xb;   double fb; 
    double xc;   
    double diff;

    double PmPrimeRoot = getPmPrimeFirstRoot();

    xa  = 1.0 - PmPrimeRoot;
    fa  = evaluatePm(xa);
    xb  = 1.0 - 1.5*PmPrimeRoot;
    fb  = evaluatePm(xb);

    while(std::abs(xa-xb) > tol)
    {
        diff = (fb-fa)/(xb-xa);
        xc   = xb - evaluatePm(xb)/diff;
        xa   = xb;
        fa   = fb;
        xb   = xc;
        fb   = evaluatePm(xb);
    }
    return xb;

}
double getPmPrimeFirstRoot()
{
    //
    // The secant method with a good initial guess.
    //
    double tol = 1.0e-12;
    double   m = double(polyDegree);
    double xa;   double fa;
    double xb;   double fb; 
    double xc;   
    double diff;

    xa = (((4.49)*(4.49))/(4.0*(m+1)*(m+1)));
    fa = evaluatePmPrime(xa);
    xb = (((4.49)*(4.49))/(4.0*(m)*(m+1)));
    fb = evaluatePmPrime(xb);
        
    while(std::abs(xa-xb) > tol)
    {
        diff = (fb-fa)/(xb-xa);
        xc   = xb - evaluatePmPrime(xb)/diff;
        xa   = xb;
        fa   = fb;
        xb   = xc;
        fb   = evaluatePmPrime(xb);
    }
    return xb;
}

double evaluatePmPrime(double x)
{
    double t1,t2,t3,t4,t6,t9,t10,t11,t18,t25;
    double m = double(polyDegree);
    t1 = m+1.0;
    t2 = std::sqrt(x);
    t3 = std::asin(t2);
    t4 = t1*t3;
    t6 = 1/t2;
    t9 = std::sqrt(1.0-x);
    t10 = 1/t9;
    t11 = std::sin(2.0*t3);
    t18 = t11*t11;
    t25 = std::cos(2.0*t4)*t6*t10/t11-std::sin(2.0*t4)/t1/t18*std::cos(2.0*t3)*t6*t10;
    return t25;
}
double evaluateInversePm(double y)
{
    double tol = 1.0e-12;
    double m   = double(polyDegree);

    double xa = 0.0;   double fa = 0.0;
    double xb = 0.0;   double fb = 0.0;
    double xc = 0.0;
    double diff = 0.0;

    //
    // Invert Pm(x)
    //
    if(y < 0.0) // Extend Pm^(-1) to be constant for neg. values.
    {
        return std::sin(0.5*(3.14159265359/(m+1)))*std::sin(0.5*(3.14159265359/(m+1)));
    }

    if(y <= 5.0)
    {
    xa = 0.0; 
    fa = 1.0;
    xb  = sin(0.5*(3.14159265359/(m+1))); 
    xb *= xb;
    fb = 0.0;

    while(std::abs(xa-xb) > tol)
    {
        diff = (fb-fa)/(xb-xa);
        xc   = xb - (evaluatePm(xb)-y)/diff;
        xa   = xb;
        fa   = fb;
        xb   = xc;
        fb   = evaluatePm(xb);
    }
    }
    // 
    // Transform equations using logs 
    // for large positive arguments 
    //
    if(y > 5.0)
    {
        xa = 0.0;
        fa = 0.0;
        xb = -4.0/((m+1.0)*(m+1.0));
        fb = log(evaluatePm(xb));
        while(std::abs(xa-xb) > tol)
        {
        diff = (fb-fa)/(xb-xa);
        xc   = xb - (log(evaluatePm(xb))-log(y))/diff;
        xa   = xb;
        fa   = fb;
        xb   = xc;
        fb   = log(evaluatePm(xb));
        }
    }
    return xb;
}
//
// Given a prescribed shift and a bound on lambda, lambdaBound, and
// and a lambdaStar such that  -shift < lambdaStar < lambdaBound
// then this routine computes the degree of the polynomial (starDegree)
// and a larger lambda bound (starBound) so that the transformation 
//
// lambda -> (lambda + shift)/(starBound + shift)/upperXstar 
//
// maps lambdaStar to Xstar. Hence for all values of lambda in
// [-shift,lambdaStar], the transformed values are mapped to an interval
// upon which the polynomial Pm(x) is strictly monotone decreasing and
// m = starDegree. 
//
// It finds these values by first finding the degree of the polynomial
// so that the value of transformed lambdaStar will be greater than Xstar 
// and then increases the value of the spectral bound so that the 
// transformed lambdaStar coincides with Xstar. 
//
// A solution is guaranteed since for polyDegree = 1 the transformed 
// lambdaStar < Xstar and as one increases polyDegree the value of
// Xstar decreases monotonically, so there exists a polyDegree > 1 which
// bracket the solution. 
// 
// 
void getStarDegreeAndSpectralRadius(double shift, double lambdaBound, 
double lambdaStar, long polyDegreeMax, long& starDegree, double& starBound)
{
    double lambdaStarShift = lambdaStar + shift;
    double ratio;

    long MA; 
    long MB;
    int exitFlag;

    // Check for degree 1

    polyDegree = 1;
    ratio      = (getXStar()/getUpperXStar());
    if(ratio*(lambdaBound + shift) < lambdaStarShift)
    {
    starDegree = 1;
    starBound  =  (lambdaStarShift/ratio) - shift;
    return;
    }

//  Find values of polyDegree that bracket the solution 
//  by successively doubling to find an upper bound 

    MA       = 1; 
    MB       = 2;

    exitFlag = 0;
    while((exitFlag == 0)&&(MB < polyDegreeMax))
    {
        polyDegree = MB;
        ratio      = (getXStar()/getUpperXStar());
        if(ratio*(lambdaBound + shift) < lambdaStarShift )
        {exitFlag    = 1;}
        else
        {MB *= 2;}
    }

    if(MB >= polyDegreeMax) 
    {
    starDegree  = polyDegreeMax;
    polyDegree  = starDegree;
    starBound   = lambdaBound;
    return;
    }
//
//  Use a bisection procedure to bracket the solution 
//
    double valMid; 
    long   Mmid; long Mdiff;

    exitFlag = 0;
    while(exitFlag == 0)
    {
    Mdiff = (MB - MA);
    if(Mdiff > 1)
    {
        Mmid = MA + Mdiff/2;
        polyDegree  = Mmid;
        ratio       = (getXStar()/getUpperXStar());
        valMid      = ratio*(lambdaBound + shift) - lambdaStarShift;
        if(valMid >= 0){ MA = Mmid;}
        else           { MB = Mmid;}
    }
    else
    {exitFlag = 1;};
    }
//
//  Adjust spectral radius bound to obtain equality
//
    starDegree  = MB;
    polyDegree  = starDegree;
    ratio       = (getXStar()/getUpperXStar());
    starBound  = (lambdaStarShift/ratio) - shift;
}

//
// Experimental
//

//
// This routine sets the class data parameters so that the apply(lambda) member
// function evaluates
//
// Pm( (lambda + spectralRadius)/(2*spectralRadius))
//
// where m = polyDegree
//
void setTwoSidedParameters(double spectralRadius, long polyDegree)
{
    this->polyDegree       = polyDegree;
    this->shift            = spectralRadius;
    this->lambdaMax        = spectralRadius;
    this->UpperXStar       = 1.0;
    this->repetitionFactor = 1;
}
//
//                    getTwoSidedStarDegree
//
//  Let lambdaStar be a value in  [-spectralRadius,spectralRadius].
//
//  This routine determines the highest degree polynomial < polyDegreeMax
//  such that the value of the mth Chebyshev polyomial of the second kind
//
//  Pm((lambda + spectralRadius)/(2*spectralRadius))
//
//  is monotone decreasing for
//
//  lambda in  [-spectralRadius, -|lambdaStar|]
//
//  and monotone increasing for
//
//  lambda in [spectralRadius-|lambdaStar|, spectralRadius]
//
//
void getTwoSidedStarDegree(double spectralRadius, double lambdaStar, long polyDegreeMax, long& starDegree)
{
    //
    // We need lambdaStar > 0 so always use absolute values
    //

    lambdaStar = std::abs(lambdaStar);

    // Using the procedure to determine xStar at the left edge
    // lambdaStarShift is the shift + the opposite of lambdaStar.

    double shift;
    double lambdaStarShift;

    shift           =  spectralRadius;
    lambdaStarShift = -lambdaStar + shift;


    long MA;
    long MB;
    int exitFlag;

    this->polyDegree       = 1;
    this->shift            = shift;
    this->lambdaMax        = spectralRadius;
    this->repetitionFactor = 1;
    this->UpperXStar        = 1.0;

    // Check for degree 1

    polyDegree       = 1;
    double xStar    = getXStar();

    // Degree 1 is already monotone

    if(xStar*(spectralRadius + shift) < lambdaStarShift)
    {
    starDegree       = 1;
    this->polyDegree = 1;
    return;
    }

//  Find values of polyDegree that bracket the solution
//  by successively doubling to find an upper bound

    MA       = 1;
    MB       = 2;

    exitFlag = 0;
    while((exitFlag == 0)&&(MB < polyDegreeMax))
    {
        polyDegree = MB;
        xStar      = getXStar();
        if(xStar*(spectralRadius + shift) < lambdaStarShift )
        {exitFlag    = 1;}
        else
        {MB *= 2;}
    }

    if(MB >= polyDegreeMax)
    {
    starDegree = polyDegreeMax;
    this->polyDegree = polyDegreeMax;
    this->lambdaMax  = spectralRadius;
    return;
    }
//
//  Use a bisection procedure to bracket the solution
//
    double valMid;
    long   Mmid; long Mdiff;

    exitFlag = 0;
    while(exitFlag == 0)
    {
    Mdiff = (MB - MA);
    if(Mdiff > 1)
    {
        Mmid = MA + Mdiff/2;
        polyDegree = Mmid;
        xStar      = getXStar();
        valMid     = xStar*(spectralRadius + shift) - lambdaStarShift;
        if(valMid >= 0){ MA = Mmid;}
        else           { MB = Mmid;}
    }
    else
    {exitFlag = 1;};
    }

    starDegree        = MA;
    this->lambdaMax  = spectralRadius;
    this->polyDegree = starDegree;
}



double getStarBound(double lambdaStar, double shift, long starDegree)
{
	double lambdaStarShift = lambdaStar + shift;
    double ratio;

	polyDegree  = starDegree;
    ratio       = (getXStar()/getUpperXStar());
    return (lambdaStarShift/ratio) - shift;
}

     long   polyDegree;
     long   repetitionFactor;
     double lambdaMax;
     double shift;
     double XStar;
     double UpperXStar;
};

#endif


 
