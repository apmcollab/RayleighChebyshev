## RayleighChebyshev

The RayleighChebyshev class is a templated class with member functions for computing eigenpairs
corresponding to the lowest eigenvalues of a symmetric or complex Hermitian linear operator. It is
assumed that all of the eigenvalues of the operator are real and that there is a basis of orthogonal eigenvectors. 

The eigenvalues are returned in a vector<double> instance while the eigenvectors are internally allocated and returned in a vector<Vtype> class instance.

OpenMP multi-thread usage is enabled by defining _OPENMP


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




