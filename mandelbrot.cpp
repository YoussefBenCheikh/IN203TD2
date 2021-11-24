# include <iostream>
# include <cstdlib>
# include <string>
# include <chrono>
# include <cmath>
# include <vector>
# include <fstream>
# include <mpi.h>

/** Une structure complexe est définie pour la bonne raison que la classe
 * complex proposée par g++ est très lente ! Le calcul est bien plus rapide
 * avec la petite structure donnée ci--dessous
 **/
struct Complex
{
    Complex() : real(0.), imag(0.)
    {}
    Complex(double r, double i) : real(r), imag(i)
    {}
    Complex operator + ( const Complex& z )
    {
        return Complex(real + z.real, imag + z.imag );
    }
    Complex operator * ( const Complex& z )
    {
        return Complex(real*z.real-imag*z.imag, real*z.imag+imag*z.real);
    }
    double sqNorm() { return real*real + imag*imag; }
    double real,imag;
};

std::ostream& operator << ( std::ostream& out, const Complex& c )
{
  out << "(" << c.real << "," << c.imag << ")" << std::endl;
  return out;
}

/** Pour un c complexe donné, calcul le nombre d'itérations de mandelbrot
 * nécessaires pour détecter une éventuelle divergence. Si la suite
 * converge, la fonction retourne la valeur maxIter
 **/
int iterMandelbrot( int maxIter, const Complex& c)
{
    Complex z{0.,0.};
    // On vérifie dans un premier temps si le complexe
    // n'appartient pas à une zone de convergence connue :
    // Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
    if ( c.real*c.real+c.imag*c.imag < 0.0625 )
        return maxIter;
    if ( (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625 )
        return maxIter;
    // Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}    
    if ((c.real > -0.75) && (c.real < 0.5) ) {
        Complex ct{c.real-0.25,c.imag};
        double ctnrm2 = sqrt(ct.sqNorm());
        if (ctnrm2 < 0.5*(1-ct.real/ctnrm2)) return maxIter;
    }
    int niter = 0;
    while ((z.sqNorm() < 4.) && (niter < maxIter))
    {
        z = z*z + c;
        ++niter;
    }
    return niter;
}

/**
 * On parcourt chaque pixel de l'espace image et on fait correspondre par
 * translation et homothétie une valeur complexe c qui servira pour
 * itérer sur la suite de Mandelbrot. Le nombre d'itérations renvoyé
 * servira pour construire l'image finale.
 
 Sortie : un vecteur de taille W*H avec pour chaque case un nombre d'étape de convergence de 0 à maxIter
 MODIFICATION DE LA FONCTION :
 j'ai supprimé le paramètre W étant donné que maintenant, cette fonction ne prendra plus que des lignes de taille W en argument.
 **/
void 
computeMandelbrotSetRow( int W, int H, int maxIter, int num_ligne, int* pixels)
{
    // Calcul le facteur d'échelle pour rester dans le disque de rayon 2
    // centré en (0,0)
    double scaleX = 3./(W-1);
    double scaleY = 2.25/(H-1.);
    //
    // On parcourt les pixels de l'espace image :
    for ( int j = 0; j < W; ++j ) {
       Complex c{-2.+j*scaleX,-1.125+ num_ligne*scaleY};
       pixels[j] = iterMandelbrot( maxIter, c );
    }
}

std::vector<int>
computeMandelbrotSet( int W, int H, int maxIter )
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::vector<int> pixels(W*H);
    start = std::chrono::system_clock::now();
    // On parcourt les pixels de l'espace image :
    for ( int i = 0; i < H; ++i ) {
      computeMandelbrotSetRow(W, H, maxIter, i, pixels.data() + W*(H-i-1) );
    }
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Temps calcul ensemble mandelbrot : " << elapsed_seconds.count() 
              << std::endl;
    return pixels;
}

std::vector<int>
computeMandelbrotSetMultiThread( int W, int H, int H1, int H2, int maxIter )
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::vector<int> pixels(W*(H2-H1));
    start = std::chrono::system_clock::now();
    // On parcourt les pixels de l'espace image :
    for ( int i = H1; i < H2; ++i ) {
      computeMandelbrotSetRow(W, H, maxIter, i, pixels.data() + W*(i-H1) );
    }
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Temps calcul ensemble mandelbrot : " << elapsed_seconds.count() 
              << std::endl;
    return pixels;
}

/** Construit et sauvegarde l'image finale **/
void savePicture( const std::string& filename, int W, int H, const std::vector<int>& nbIters, int maxIter )
{
    double scaleCol = 1./maxIter;//16777216
    std::ofstream ofs( filename.c_str(), std::ios::out | std::ios::binary );
    ofs << "P6\n"
        << W << " " << H << "\n255\n";
    for ( int i = 0; i < W * H; ++i ) {
        double iter = scaleCol*nbIters[i];
        unsigned char r = (unsigned char)(256 - (unsigned (iter*256.) & 0xFF));
        unsigned char b = (unsigned char)(256 - (unsigned (iter*65536) & 0xFF));
        unsigned char g = (unsigned char)(256 - (unsigned( iter*16777216) & 0xFF));
        ofs << r << g << b;
    }
    ofs.close();
}



int main(int nargs, char *argv[] ) 
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


    const int W = 800;
    const int H = 600;
    // Normalement, pour un bon rendu, il faudrait le nombre d'itérations
    // ci--dessous :
    //const int maxIter = 16777216;
    const int maxIter = 8*65536;
    //auto iters = computeMandelbrotSet( W, H, maxIter );
    //savePicture("mandelbrot.tga", W, H, iters, maxIter);
    
    /*
    // decoupage par lignes
    auto start = std::chrono::system_clock::now();
    auto iters_i = computeMandelbrotSetMultiThread(W, H, rank*H/nbp, (rank+1)*H/nbp, maxIter );
	if (rank == 0){
        double scale = 1./maxIter;//16777216
        std::ofstream o( "mandelbrot.tga", std::ios::out | std::ios::binary );
        ofs << "P6\n"
            << W << " " << H << "\n255\n";
        for ( int i = 0; i < W * H/nbp; ++i ) {
                double iter = scale*iters_i[i];
                unsigned char r = (unsigned char)(256 - (unsigned (iter*256.) & 0xFF));
                unsigned char b = (unsigned char)(256 - (unsigned (iter*65536) & 0xFF));
                unsigned char g = (unsigned char)(256 - (unsigned( iter*16777216) & 0xFF));
                ofs << r << g << b;
            }
        for ( int k = 1; k < nbp ; k++){
            std::cout << "Receiving  processus n°" << k << ", size : " << W * (H*(k+1)/nbp  - H*k/nbp) << ".\n";
            MPI_Recv (&iters_i[0] , W * (H*(k+1)/nbp  - H*k/nbp), MPI_INT , k, 0, MPI_COMM_WORLD ,& status );
            std::cout << "Received processus n°" << k << ".\n";
            for ( int i = 0; i < W * (H*(k+1)/nbp  - H*k/nbp); ++i ) {
                double iter = scale*iters_i[i];
                unsigned char r = (unsigned char)(256 - (unsigned (iter*256.) & 0xFF));
                unsigned char b = (unsigned char)(256 - (unsigned (iter*65536) & 0xFF));
                unsigned char g = (unsigned char)(256 - (unsigned( iter*16777216) & 0xFF));
                ofs << r << g << b;
            }
        }
        o.close();
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "Temps calcul ensemble mandelbrot : " << elapsed_seconds.count() << std::endl;

    }else{
        //MPI_Send (&iters_i , 1, MPI_INT , 0, 0, MPI_COMM_WORLD );
        MPI_Send(&iters_i[0], W * (H*(rank+1)/nbp  - H*rank/nbp), MPI_INT, 0, 0, MPI_COMM_WORLD);
        std::cout << " Sent ( je suis le processus n°" << rank << ".)\n";

    }

*/
   //maitre-esclave
    
    if ( rank == 0 ) // 0 => master
    {
        auto start = std::chrono::system_clock::now();
        std::vector<int> pixels(W*H);
        int count_task = 0;
        for ( int i = 1 ; i < nbp ; ++i ) {
            MPI_Send(&count_task, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            
            count_task += 1;
        }
        while ( count_task < H) {
            
            std::vector<int> pixels_temp(W+1);
            MPI_Recv(&pixels_temp[0] , W+1 , MPI_INT , MPI_ANY_SOURCE, 0, MPI_COMM_WORLD ,& status );
            for (int k = 0; k < W ; k++){
                pixels[W*pixels_temp[W]+k] = pixels_temp[k];
            }
            //recvv( r e s u l t , . . . , MPI_ANY_SOURCE, . . . , &s t a t u s ) ;
            MPI_Send(&count_task, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            //send(&count_task , 1 , MPI_INT , s t a t u s .MPI_SOURCE, . . . ) ;
            count_task += 1 ;
        }
       
        count_task = -1;
        for ( int i = 1 ; i <nbp ; ++i ){
            MPI_Send(&count_task, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            //send(&count_task , 1 , MPI_INT , i , . . . ) ;
        }
        savePicture("mandelbrot.tga", W, H,  pixels, maxIter );
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "Temps calcul ensemble mandelbrot : " << elapsed_seconds.count() << std::endl;
    }else{
        // Cas travailleur
        int num_task = 0;
        while (num_task != -1){
            MPI_Recv(&num_task, 1 , MPI_INT , MPI_ANY_SOURCE, 0, MPI_COMM_WORLD ,& status );
            if (num_task >= 0) {
                int num_ligne = num_task;
                std::vector<int> pixels(W+1);
                computeMandelbrotSetRow(W, H, maxIter, num_ligne, pixels.data());
                pixels[W] = num_task;
                
                MPI_Send(&pixels[0], W+1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                
            }
        }

    }
    

	// Création d'un fichier pour ma propre sortie en écriture :
	//std::stringstream fileName;
	//fileName << "Output" << std::setfill('0') << std::setw(5) << rank << ".txt";
	//std::ofstream output( fileName.str().c_str() );
	
	// Rajout du programme ici...
	
	//output.close();
	// A la fin du programme, on doit synchroniser une dernière fois tous les processus
	// afin qu'aucun processus ne se termine pendant que d'autres processus continue à
	// tourner. Si on oublie cet instruction, on aura une plantage assuré des processus
	// qui ne seront pas encore terminés.
	MPI_Finalize();

    return EXIT_SUCCESS;
 }
    
