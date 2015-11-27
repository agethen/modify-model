#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include <caffe/proto/caffe.pb.h>

#include <blob.hh>

caffe::LayerParameter * getLayerByName( std::string name, caffe::NetParameter * net ){
	for( int i = 0; i < net->layer_size(); i++ ){
		caffe::LayerParameter * l = net->mutable_layer( i );
		if( l->name() == name )
			return l;
	}
	return nullptr;
}

caffe::LayerParameter * getLayerByInt( int idx, caffe::NetParameter * net ){
	if( idx >= net->layer_size() || idx < 0 )
		return nullptr;
	return net->mutable_layer( idx );
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


template<typename T>
void saveNetwork( std::string outputfile, T & network ){
	std::fstream output( outputfile, std::ios::out | std::ios::trunc | std::ios::binary);

 	if( !network.SerializeToOstream(&output) )
		std::cerr << "Encountered error writing proto file" << std::endl;
}

int main( int argc, char ** argv ){
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	std::string filename = "vgg_16_action_flow_pretrain.caffemodel";

	caffe::NetParameter net;
	loadNetwork<caffe::NetParameter>( filename, &net );

	std::string name = "conv1_1";
	auto l = getLayerByName( name, &net );

	if( l == nullptr ){
		std::cerr << "Could not find layer" << std::endl;
		return EXIT_FAILURE;
	}

	BlobHandler<float> weights, bias;
	weights.fromBlob( l->mutable_blobs( 0 ) );
	bias.fromBlob( l->mutable_blobs( 1 ) );

	// Application: Repeat channel
	// Note that this does not automatically adjust biases or previous/following layers
	weights.inflateAlongAxis( 1, 30, FLAG_RGB & FLAG_REPEAT );	// Increase the dimension of axis 1 to 30, expecting RGB information to be repeated
	weights.update();																						// Commit changes

	saveNetwork( "output.caffemodel", net );

	return EXIT_SUCCESS;
}