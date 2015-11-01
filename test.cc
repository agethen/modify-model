#include <blob.hh>
#include <iostream>
int main( int argc, char ** argv ){

	BlobHandler<float> hnd;

	std::vector<int> shape{ 4, 4 };

	auto q = std::shared_ptr<std::vector<float>>( new std::vector<float>( ) );
	for( int i = 0; i < 16; i++ )
		q->push_back( i );
	hnd.setData( q, shape );

	// hnd.setShape( shape );
	// hnd.random( 0.0, 1.0 );	



	auto d = hnd.data();

	std::vector<int> s{ 2, 3 };
	auto p = std::shared_ptr<std::vector<float>>( new std::vector<float>() );

	hnd.readWithShape( p, s );

	hnd.print();

	return EXIT_SUCCESS;
}