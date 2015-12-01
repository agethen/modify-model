#include <view.hh>

void viewNetwork( caffe::NetParameter & net, int from, int to ){
	int32_t num_layers = net.layer_size();

	if( !num_layers )
		std::cerr << "Layer type does not seem to be compatible with V2" << std::endl;

  std::cout << "Number of layers: " << num_layers << std::endl;
  
 	from = std::max( 0, from );
 	to = std::min( to, net.layer_size() );

  for( int32_t i = from; i < to; i++ ){
    caffe::LayerParameter * layer = net.mutable_layer(i);

    std::string layer_type = layer->type();
    std::cout << "V2 Layer " << i << ", Type: " << layer_type << std::endl;

    if( layer_type == "Convolution" || layer_type == "InnerProduct" ){
      viewLayer( layer );
    }
  }
}

void viewNetwork( caffe::NetParameter & net ){
	viewNetwork( net, 0, net.layer_size() );
}

void viewLayer( caffe::LayerParameter * layer ){
  std::cout << "\t" << "Name: " << layer->name() << std::endl;
  if( layer->blobs_size() > 0 ){

    BlobHandler<float> weights, bias;
    weights.fromBlob( layer->mutable_blobs( 0 ) );
    bias.fromBlob( 		layer->mutable_blobs( 1 ) );

    std::cout << "\t" << "Shape Weights:\t" << weights.shape() << std::endl;
    std::cout << "\t" << "Shape Biases:\t" << bias.shape() << std::endl;
  }else{
    std::cout << "Empty layer" << std::endl;
  }
}


void listSolverObjects( caffe::SolverParameter & solver ){
	std::cout << "\nListing Solver Objects: " << std::endl;

	if( solver.has_net() )
		std::cout << "\thas_net(): " << solver.net() << std::endl;
	if( solver.has_net_param() )
		std::cout << "\thas_net_param()" << std::endl;
	if( solver.has_train_net() )
		std::cout << "\thas_train_net(): "  << solver.train_net() << std::endl;
	if( solver.test_net_size() )
		std::cout << "\ttest_net_size(): " << solver.test_net_size()  << std::endl;
	if( solver.has_train_net_param() )
		std::cout << "\thas_train_net_param()"  << std::endl;
	if( solver.test_net_param_size() )
		std::cout << "\ttest_net_param_size(): " << solver.test_net_param_size() << std::endl;
	if( solver.has_train_state() )
		std::cout << "\thas_train_state()" << std::endl;
	if( solver.test_state_size() )
		std::cout << "\ttest_state_size(): " << solver.test_state_size() << std::endl;
	if( solver.test_iter_size() )
		std::cout << "\ttest_iter_size(): " << solver.test_iter_size() << std::endl;
	if( solver.has_test_interval() )
		std::cout << "\thas_test_interval()" << std::endl;
	if( solver.has_test_compute_loss() )
		std::cout << "\thas_test_compute_loss()" << std::endl;
	if( solver.has_test_initialization() )
		std::cout << "\thas_test_initialization()" << std::endl;
	if( solver.has_base_lr() )
		std::cout << "\thas_base_lr(): " << solver.base_lr() << std::endl;
	if( solver.has_display() )
		std::cout << "\thas_display()" << std::endl;
	if( solver.has_average_loss() )
		std::cout << "\thas_average_loss()" << std::endl;
	if( solver.has_max_iter() )
		std::cout << "\thas_max_iter(): " << solver.max_iter() << std::endl;
	if( solver.has_iter_size() )
		std::cout << "\thas_iter_size(): " << solver.iter_size() << std::endl;
	if( solver.has_lr_policy() )
		std::cout << "\thas_lr_policy(): " << solver.lr_policy() << std::endl;
	if( solver.has_gamma() )
		std::cout << "\thas_gamma(): " << solver.gamma() << std::endl;
	if( solver.has_power() )
		std::cout << "\thas_power(): " << solver.power() << std::endl;
	if( solver.has_momentum() )
		std::cout << "\thas_momentum(): " << solver.momentum() << std::endl;
	if( solver.has_weight_decay() )
		std::cout << "\thas_weight_decay(): " << solver.weight_decay() << std::endl;
	if( solver.has_regularization_type() )
		std::cout << "\thas_regularization_type(): " << solver.regularization_type() << std::endl;
	if( solver.has_stepsize() )
		std::cout << "\thas_stepsize(): " << solver.stepsize() << std::endl;
	if( solver.stepvalue_size() )
		std::cout << "\tstepvalue_size(): " << solver.stepvalue_size() << std::endl;
	if( solver.has_clip_gradients() )
		std::cout << "\thas_clip_gradients() " << solver.clip_gradients() << std::endl;
	if( solver.has_snapshot() )
		std::cout << "\thas_snapshot(): " << solver.snapshot() << std::endl;
	if( solver.has_snapshot_prefix() )
		std::cout << "\thas_snapshot_prefix(): " << solver.snapshot_prefix() << std::endl;
	if( solver.has_snapshot_diff() )
		std::cout << "\thas_snapshot_diff(): " << solver.snapshot_diff() << std::endl;
	if( solver.has_snapshot_format() )
		std::cout << "\thas_snapshot_format()" << std::endl;
	if( solver.has_solver_mode() )	
		std::cout << "\thas_solver_mode()" << std::endl;
	if( solver.has_device_id() )
		std::cout << "\thas_device_id(): " << solver.device_id() << std::endl;
	if( solver.has_random_seed() )
		std::cout << "\thas_random_seed(): " << solver.random_seed() << std::endl;
	if( solver.has_solver_type() )
		std::cout << "\thas_solver_type()" << std::endl;
	if( solver.has_delta() )
		std::cout << "\thas_delta(): " << solver.delta() << std::endl;
	if( solver.has_momentum2() )
		std::cout << "\thas_momentum2(): " << solver.momentum2() << std::endl;
	if( solver.has_rms_decay() )
		std::cout << "\thas_rms_decay(): " << solver.rms_decay() << std::endl;
	if( solver.has_debug_info() )
		std::cout << "\thas_debug_info(): " << solver.debug_info() << std::endl;
	if( solver.has_snapshot_after_train() )
		std::cout << "\thas_snapshot_after_train(): " << solver.snapshot_after_train() << std::endl;

	std::cout << "------------------" << std::endl;
}

std::ostream & operator<<( std::ostream & stream, std::vector<int> data){
	stream << "[ ";
	for( auto d : data )
		stream << d << " ";
	stream << "]";
	return stream;
}
