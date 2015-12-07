#include <blob.hh>

template <typename Dtype>
BlobHandler<Dtype>::BlobHandler(){
	blob_ = nullptr;
	data_ = std::shared_ptr<std::vector<Dtype>>( new std::vector<Dtype>() );
}

template <typename Dtype>
BlobHandler<Dtype>::~BlobHandler(){
	shape_.clear();
}

template <typename Dtype>
int BlobHandler<Dtype>::size(){
	if( !shape_.size() )
		return 0;

	int size = 1;
	for( auto d : shape_ ){
		size *= d;
	}

	return size;
}

template <typename Dtype>
void BlobHandler<Dtype>::update(){
	if( !blob_ )
		return;

	setBlobShape();	

	auto tmp_data = data_;

	while( tmp_data->size() < size() )
		tmp_data->push_back( 0.0 );

	blob_->clear_data();

	for( auto e : *tmp_data )
		blob_->add_data( e );
}

template <typename Dtype>
void BlobHandler<Dtype>::zero(){
	int s = this->size();

	data_->clear();

	for( int i = 0; i < s; i++ )
		data_->push_back( 0 );
}

template <typename Dtype>
void BlobHandler<Dtype>::random( float mu, float sigma ){
	std::default_random_engine generator;
	std::normal_distribution<Dtype> distribution( mu, sigma );

	int s = this->size();

	data_->clear();

	for( int i = 0; i < s; i++ )
		data_->push_back( distribution( generator ) );
}

template <typename Dtype>
std::vector<int> BlobHandler<Dtype>::parseShape(){
	std::vector<int> shape = std::vector<int>();

	if( !blob_ )
		return shape;

	if( blob_->has_shape() ){
		caffe::BlobShape * s = blob_->mutable_shape();
		for( int i = 0; i < s->dim_size(); i++ )
			shape.push_back( s->dim( i ) );
	}else{
		shape.push_back( blob_->num() );
		shape.push_back( blob_->channels() );
		shape.push_back( blob_->height() );
		shape.push_back( blob_->width() );
	}
	return shape;
}

template <typename Dtype>
void BlobHandler<Dtype>::setBlobShape(){
	if( !blob_ )
		return;

	if( !blob_->has_shape() ){
		// Outdated format, but still used...
		std::vector<int> tmp = shape_;

		while( tmp.size() < 4 )
			tmp.push_back( 1 );

		if( tmp.size() > 4 )
			std::cerr << "New Shape too large, ignoring additional dims." << std::endl;

		blob_->set_num( tmp[0] );
		blob_->set_channels( tmp[1] );
		blob_->set_height( tmp[2] );
		blob_->set_width( tmp[3] );

	}else{
		caffe::BlobShape * bs = blob_->mutable_shape();
		bs->clear_dim();

		for( auto d : shape_ )
			bs->add_dim( d );
	}
}

template <typename Dtype>
void BlobHandler<Dtype>::fromBlob( caffe::BlobProto * blob ){
	blob_ = blob;

	shape_ = parseShape();

	data_ = std::shared_ptr<std::vector<Dtype>>( new std::vector<Dtype>() );
	for( int i = 0; i < blob->data_size(); i++ )
		data_->push_back( blob->data( i ) );
}

template <typename Dtype>
void BlobHandler<Dtype>::readWithShape( std::shared_ptr<std::vector<Dtype>> data, std::vector<int> new_shape ){
	std::vector<int> current_shape;

	if( new_shape.size() == shape_.size() )
		current_shape = shape_;
	
	// If the shapes don't match, we add singular dimensions to the smaller one
	if( new_shape.size() > shape_.size() ){
		current_shape = std::vector<int>( shape_.begin(), shape_.end() );
		for( int i = shape_.size(); i < new_shape.size(); i++ )
			current_shape.push_back( 1 );
	}

	if( new_shape.size() < shape_.size() ){
		current_shape = shape_;
		for( int i = new_shape.size(); i < shape_.size(); i++ )
			new_shape.push_back( 1 );
	}

	auto blob_data = this->data();

	recursiveRead( data, blob_data, 0, 0, new_shape, current_shape );
}

template <typename Dtype>
void BlobHandler<Dtype>::recursiveRead( std::shared_ptr<std::vector<Dtype>> data,
			 std::shared_ptr<std::vector<Dtype>> blob_data, int idx, int blob_idx,
			 std::vector<int> new_shape, std::vector<int> current_shape ){

	if( current_shape.empty() ){
		
		while( data->size() <= idx )
			data->push_back( 0 );

		data->at( idx ) = blob_data->at( blob_idx );
		return;
	}

	int min = std::min( current_shape[0], new_shape[0] );
	for( int i = 0; i < min; i++ ){
		int blocksize = 1;
		int new_blocksize = 1;

		for( int j = 1; j < current_shape.size(); j++ )
			blocksize *= current_shape[j];

		for( int j = 1; j < new_shape.size(); j++ )
			new_blocksize *= new_shape[j];

		recursiveRead( data, blob_data, idx+i*new_blocksize, blob_idx+i*blocksize,
		 	std::vector<int>( new_shape.begin()+1, new_shape.end() ), 
		 	std::vector<int>( current_shape.begin()+1, current_shape.end() ) );
	}
}

template <typename Dtype>
void BlobHandler<Dtype>::inflateAlongAxis( int axis, int new_value, int flag = FLAG_ZERO ){
	if( !blob_ )
		return;

	if( axis >= shape_.size() ){
		std::cerr << "Unknown axis!" << std::endl;
		return;
	}

	if( shape_[axis] > new_value ){
		std::cerr << "Deflating shape not implemented!" << std::endl;
		return;
	}

	auto pattern_shape = shape_;
	auto new_shape = shape_;
	new_shape[axis] = new_value;
	
	for( int i = 0; i <= axis; i++ )
		pattern_shape[i] = 1;

	if( flag & FLAG_RGB )
		pattern_shape[axis] = 3;
	

	int blocksize = 1;
	int blocks = 1;

	for( int i = 0; i < axis; i++ )
		blocks *= shape_[i];

	for( int i = axis+1; i < shape_.size(); i++ )
		blocksize *= shape_[i];

	auto tmp_data		= std::shared_ptr<std::vector<Dtype>>( new std::vector<Dtype>() );

	if( flag & FLAG_ZERO ){
		readWithShape( tmp_data, new_shape );
		data_ = tmp_data;
		shape_ = new_shape;
		return;
	}

	for( int i = 0; i < blocks; i ++ ){
		auto pattern 		= std::shared_ptr<std::vector<Dtype>>( new std::vector<Dtype>() );
		recursiveRead( pattern, data_, 0, i*blocksize*shape_[axis], pattern_shape, shape_ );

		if( (flag & FLAG_RGB) && (flag & FLAG_AVERAGE) ){
			// Need to average the 3 RGB channel
			auto tmp_pattern = std::shared_ptr<std::vector<Dtype>>( new std::vector<Dtype>() );
			int len = pattern->size()/3;
			for( int j = 0; j < len; j++ ){
				Dtype val = pattern->at( j ) + pattern->at( len+j ) + pattern->at( 2*len + j );
				tmp_pattern->push_back( val/3.0 );
			}
			pattern = tmp_pattern;
		}

		repeatPattern( pattern, tmp_data, i*blocksize*new_value, ( (flag & FLAG_RGB) && !(flag & FLAG_AVERAGE) )?new_value/3:new_value );
	}
	data_ = tmp_data;
	shape_ = new_shape;
}

template <typename Dtype>
void BlobHandler<Dtype>::repeatPattern( std::shared_ptr<std::vector<Dtype>> pattern, std::shared_ptr<std::vector<Dtype>> data, int64_t offset, int64_t times ){
	while( data->size() < offset )
		data->push_back( 0.0 );

// TODO: Correctly saving data
	for( int64_t i = 0; i < times; i++)
		for( auto e : *pattern )
			data->push_back( e );
}

template <typename Dtype>
void BlobHandler<Dtype>::print(){
	if( shape_.empty() )
		return;

	recursivePrint( 0, shape_ );
}

template <typename Dtype>
void BlobHandler<Dtype>::print( std::vector<int> idx ){
	if( idx.size() > shape_.size() )
		return;

	for( int i = 0; i < idx.size(); i++ )
		if( idx[i] > shape_[i] )
			return;

	int offset = 0;
	int s = this->size();
	int r = 1;
	for( int i = 0; i < idx.size(); i++ ){
		r *= shape_[i];
		offset += idx[i]*(s/r);
	}

	std::vector<int> r_shape( shape_.begin()+idx.size(), shape_.end() );
	recursivePrint( offset, r_shape );
}

template <typename Dtype>
void BlobHandler<Dtype>::recursivePrint( int offset, std::vector<int> shape ){

	if( shape.empty() ){
		std::cout << data_->at( offset ) << " ";
		return;
	}

	std::vector<int> r_shape( shape.begin()+1, shape.end() );

	for( int i = 0; i < shape[0]; i++ ){
		int blocksize = 1;

		for( int j = 1; j < shape.size(); j++ )
			blocksize *= shape[j];

		recursivePrint( offset+i*blocksize, r_shape );
	}
	std::cout << std::endl;
}

template class BlobHandler<float>;
template class BlobHandler<double>;