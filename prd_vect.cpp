// Produit matrice-vecteur
# include <cassert>
# include <vector>
# include <iostream>
# include <mpi.h>

// ---------------------------------------------------------------------
class Matrix : public std::vector<double>
{
public:
    Matrix (int dim);
    Matrix( int nrows, int ncols );
    Matrix( const Matrix& A ) = delete;
    Matrix( Matrix&& A ) = default;
    ~Matrix() = default;

    Matrix& operator = ( const Matrix& A ) = delete;
    Matrix& operator = ( Matrix&& A ) = default;
    
    double& operator () ( int i, int j ) {
        return m_arr_coefs[i + j*m_nrows];
    }
    double  operator () ( int i, int j ) const {
        return m_arr_coefs[i + j*m_nrows];
    }
    
    std::vector<double> operator * ( const std::vector<double>& u ) const;
    
    std::ostream& print( std::ostream& out ) const
    {
        const Matrix& A = *this;
        out << "[\n";
        for ( int i = 0; i < m_nrows; ++i ) {
            out << " [ ";
            for ( int j = 0; j < m_ncols; ++j ) {
                out << A(i,j) << " ";
            }
            out << " ]\n";
        }
        out << "]";
        return out;
    }
private:
    int m_nrows, m_ncols;
    std::vector<double> m_arr_coefs;
};
// ---------------------------------------------------------------------
inline std::ostream& 
operator << ( std::ostream& out, const Matrix& A )
{
    return A.print(out);
}
// ---------------------------------------------------------------------
inline std::ostream&
operator << ( std::ostream& out, const std::vector<double>& u )
{
    out << "[ ";
    for ( const auto& x : u )
        out << x << " ";
    out << " ]";
    return out;
}
// ---------------------------------------------------------------------
std::vector<double> 
Matrix::operator * ( const std::vector<double>& u ) const
{
    const Matrix& A = *this;
    assert( u.size() == unsigned(m_ncols) );
    std::vector<double> v(m_nrows, 0.);
    for ( int i = 0; i < m_nrows; ++i ) {
        for ( int j = 0; j < m_ncols; ++j ) {
            v[i] += A(i,j)*u[j];
        }            
    }
    return v;
}

// =====================================================================
Matrix::Matrix (int dim) : m_nrows(dim), m_ncols(dim),
                           m_arr_coefs(dim*dim)
{
    for ( int i = 0; i < dim; ++ i ) {
        for ( int j = 0; j < dim; ++j ) {
            (*this)(i,j) = (i+j)%dim;
        }
    }
}
// ---------------------------------------------------------------------
Matrix::Matrix( int nrows, int ncols ) : m_nrows(nrows), m_ncols(ncols),
                                         m_arr_coefs(nrows*ncols)
{
    int dim = (nrows > ncols ? nrows : ncols );
    for ( int i = 0; i < nrows; ++ i ) {
        for ( int j = 0; j < ncols; ++j ) {
            (*this)(i,j) = (i+j)%dim;
        }
    }    
}
// =====================================================================

void mySum(double *invec, double *inoutvec, int *len, MPI_Datatype *dtype)
{
    int i;
    for ( i=0; i<*len; i++ ) inoutvec[i] += invec[i];
}


int main( int nargs, char* argv[] )
{
    // On initialise le contexte MPI qui va s'occuper :
	//    1. Créer un communicateur global, COMM_WORLD qui permet de gérer
	//       et assurer la cohésion de l'ensemble des processus créés par MPI;
	//    2. d'attribuer à chaque processus un identifiant ( entier ) unique pour
	//       le communicateur COMM_WORLD
	//    3. etc...
	MPI_Init( &nargs, &argv );
	// Pour des raisons de portabilité qui débordent largement du cadre
	// de ce cours, on préfère toujours cloner le communicateur global
	// MPI_COMM_WORLD qui gère l'ensemble des processus lancés par MPI.
	MPI_Comm globComm;
	MPI_Comm_dup(MPI_COMM_WORLD, &globComm);
	// On interroge le communicateur global pour connaître le nombre de processus
	// qui ont été lancés par l'utilisateur :
	int nbp;
	MPI_Comm_size(globComm, &nbp);
	// On interroge le communicateur global pour connaître l'identifiant qui
	// m'a été attribué ( en tant que processus ). Cet identifiant est compris
	// entre 0 et nbp-1 ( nbp étant le nombre de processus qui ont été lancés par
	// l'utilisateur )
	int rank;
	MPI_Comm_rank(globComm, &rank);

	MPI_Status status ;

    const int N = 120;
    int Nloc = N/nbp;
    
    MPI_Op vSum;
    MPI_Op_create( (MPI_User_function *)mySum, 1, &vSum );

/*
    MPI_Datatype myVector;
    MPI_Type_vector(N, 1, 1, MPI_DOUBLE, &myVector);
    MPI_Type_commit(&myVector);
*/    
    
    
    Matrix A(N);
    //std::cout  << "A : " << A << std::endl;
    std::vector<double> u( N );
    for ( int i = 0; i < N; ++i ) u[i] = i+1;
    //std::cout << " u : " << u << std::endl;
    //std::vector<double> v = A*u;
    //std::cout << "A.u = " << v << std::endl;


/*
//Produit parallèle matrice – vecteur par colonne
    std::vector<double> v_part( N );
    std::vector<double> v_final( N );
    for (int i = 0 ; i < N ; i++){
        v_part[i] = 0;
        for (int j = Nloc*rank ; j < Nloc*(rank+1) ; j++){
            v_part[i] += A(i,j)*u[j];
        }
    }
    MPI_Allreduce (&v_part[0], &v_final[0], N, MPI_DOUBLE, vSum, MPI_COMM_WORLD);
    std::cout << "v = A*u : " << v_final << " ( je suis le processus n°" << rank << ".)\n";
*/
//Produit parallèle matrice – vecteur par ligne
    std::vector<double> v_part( Nloc);
    std::vector<double> v_final( N );
    for (int i = 0 ; i < Nloc ; i++){
        v_part[i] = 0;
        for (int j = 0 ; j < N ; j++){
            v_part[i] += A(i+Nloc*rank,j)*u[j];
        }
    }
    MPI_Allgather (&v_part[0], Nloc, MPI_DOUBLE, &v_final[0], Nloc, MPI_DOUBLE, MPI_COMM_WORLD );
    //MPI_Allgather( sendarray, 100, MPI_INT, rbuf, 100, MPI_INT, comm); 
    std::cout << "v = A*u : " << v_final << " ( je suis le processus n°" << rank << ".)\n";


    MPI_Finalize();
    return EXIT_SUCCESS;
}