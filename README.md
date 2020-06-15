## RayleighChebyshev

The RayleighChebyshev class is a templated class with member functions for computing eigenpairs
corresponding to the lowest eigenvalues of a linear operator. It is
assumed that all of the eigenvalues of the operator are real and
that there is a basis of orthogonal eigenvectors. The routine is designed
for symmetric linear operators, but symmetry is not exploited in
the implementation of the procedure.

The eigenvalues are returned in a vector<double>  instance
while the eigenvectors are internally allocated and returned in
a vector<Vtype> class instance.

OpenMP multi-thread usage is enabled by defining _OPENMP



### Prerequisites
C++11
### Versioning
Release : 1.0.2
### Date
June, 15, 2020
### Authors
Chris Anderson
### License
GPLv3  For a copy of the GNU General Public License see <http://www.gnu.org/licenses/>.
### Acknowledgements



