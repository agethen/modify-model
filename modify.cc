#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include <caffe/proto/caffe.pb.h>

#include <blob.hh>

caffe::LayerParameter * getLayerByName( std::string name, caffe::NetParameter * net ){
	for( int i = 0; i < net.layer_size(); i++ ){
		caffe::LayerParameter * l = net.mutable_layer( i );
		if( l->name() == name )
			return l;
	}
	return nullptr;
}

caffe::LayerParameter * getLayerByInt( int idx, caffe::NetParameter * net ){
	if( idx >= net.layer_size() || idx < 0 )
		return nullptr;
	return net.mutable_layer( idx );
}

template<typename T>
bool loadNetwork( std::string filename, T * network ){
	using google::protobuf::io::IstreamInputStream;
	using google::protobuf::io::ZeroCopyInputStream;
	using google::protobuf::io::CodedInputStream;

	std::cout << "Reading file " << filename << std::endl;
	std::fstream input( filename.c_str(), std::ios::in | std::ios::binary );

	ZeroCopyInputStream * raw_input = new IstreamInputStream(&input);
	CodedInputStream * coded_input = new CodedInputStream(raw_input);
	coded_input->SetTotalBytesLimit( 800000000, 1000000000 );

	if( !network->ParseFromCodedStream( coded_input ) ){
		std::cerr << "Failed to read from stream" << std::endl;
		return false;
	}

  return true;
}

int main( int argc, char ** argv ){
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	std::string filename = "out.caffemodel";

	caffe::NetParameter net;
	loadNetwork<caffe::NetParameter>( filename, &net );

	std::string name = "conv1";
	auto l = getLayerByName( name );

	BlobHandler weights, bias;
	weights.fromBlob( l->mutable_blobs( 0 ) );
	bias.fromBlob( l->mutable_blobs( 1 ) );

	return EXIT_SUCCESS;
}