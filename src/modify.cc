#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include <caffe/proto/caffe.pb.h>

// Custom includes
#include <blob.hh>
#include <view.hh>

std::string input_filename 	= "vgg_16_action_flow_pretrain.caffemodel";
std::string output_filename = "output.caffemodel";

bool flag_view 		= false;

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

template <typename Dtype>
bool loadLayerBlobs( std::string layer, caffe::NetParameter & net, BlobHandler<Dtype> & weights, BlobHandler<Dtype> & bias ){
	auto l = getLayerByName( layer, &net );

	if( l == nullptr ){
		std::cerr << "Could not find layer: " << layer << std::endl;
		return false;
	}

	weights.fromBlob( l->mutable_blobs( 0 ) );
	bias.fromBlob( l->mutable_blobs( 1 ) );
	return true;
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

void parseArguments( int32_t argc, char ** argv ){
  for( int32_t i = 1; i < argc; i++ ){
    std::string argument = std::string( argv[i] );

    if( argument == "--view" ){
      flag_view = true;
      std::cout << "Using -- view: Display layers only." << std::endl;
    }

    if( argument == "--input" ){
      input_filename = std::string( argv[i+1] );
    }

    if( argument == "--output" ){
    	output_filename = std::string( argv[i+1] );
    }
  }  
}

int main( int argc, char ** argv ){
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	parseArguments( argc, argv );

	caffe::NetParameter net;
	loadNetwork<caffe::NetParameter>( input_filename, &net );

	if( flag_view ){
		viewNetwork( net );
		return EXIT_SUCCESS;
	}

	BlobHandler<float> weights, bias;
	
	if( !loadLayerBlobs<float>( "conv1", net, weights, bias ) )
		return EXIT_FAILURE;

	{
		// Application: Repeat channel
		// Note that this does not automatically adjust biases or previous/following layers
		weights.inflateAlongAxis( 1, 20, FLAG_RGB | FLAG_AVERAGE | FLAG_REPEAT );	// Increase the dimension of axis 1 to 30, expecting RGB information to be repeated
		weights.update();																						// Commit changes
	}

	{
		// Application: Reduce Dimension
		// When removing data, we do not need to care about repeating or averaging information.
		// Instead we can simply specify the new shape and read it with readWithShape().
		// auto data = std::shared_ptr<std::vector<float>>( new std::vector<float>() );

		// std::vector<int> ns = weights.shape();
		// ns[1] = 1;

		// weights.readWithShape( data, ns );
		// weights.setData( data, ns );
		// weights.update();
	}

	saveNetwork( output_filename, net );

	return EXIT_SUCCESS;
}