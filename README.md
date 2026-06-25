## RayleighChebyshev

This repository contains the RayleighChebyshev and supporting classes. The RayleighChebyshev class is a templated class with member functions for computing eigenpairs corresponding to the algebraically lowest eigenvalues of a symmetric or complex Hermitian linear operator.

The eigenvalues are returned in a vector<double> instance while the eigenvectors are internally allocated and returned in a vector<Vtype> class instance.

OpenMP multi-thread usage is enabled by defining _OPENMP

The Rayleigh-Chebyshev method is described in the paper

Christopher R. Anderson, "A Rayleigh–Chebyshev procedure for finding the smallest eigenvalues and associated eigenvectors of large sparse Hermitian matrices",
Journal of Computational Physics,
Volume 229, Issue 19,2010,Pages 7477-7487, ISSN 0021-9991,
https://doi.org/10.1016/j.jcp.2010.06.030.

The name comes from the fact that the efficiency of the underlying subspace iteration is obtained through the use of Chebyshev polynomial acceleration where the coefficients of the polynomials are adaptively determined from information obtained from the Rayleigh quotient. It is a subspace method specifically designed for determining a small number of the lowest eigenvalues and eigenvectors of Schrodinger operators where it is necessary to identify and distinguish between degenerate and nearly degenerate eigenstates. 

### Prerequisites
C++17, LAPACK
### Versioning
Release : 2.0.1
### Date
September, 17, 2024
### Authors
Chris Anderson
### License
GPLv3  For a copy of the GNU General Public License see <http://www.gnu.org/licenses/>.




