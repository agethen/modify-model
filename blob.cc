#include <blob.hh>

template <typename Dtype>
BlobHandler<Dtype>::BlobHandler(){
	blob_ = nullptr;
	data_ = std::shared_ptr<std::vector<Dtype>>( new std::vector<Dtype>() );
}

template <typename Dtype>
BlobHandler<Dtype>::~BlobHandler(){
	shape_.clear();
	delete blob_;
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

	caffe::BlobShape * bs = blob_->mutable_shape();
	bs->clear_dim();

	for( auto d : shape_ )
		bs->add_dim( d );

	int s = this->size();
	blob_->clear_data();

	for( int i = 0; i < s; i++ )
		blob_->add_data( 0 );
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
void BlobHandler<Dtype>::fromBlob( caffe::BlobProto * blob ){
	blob_ = blob;

	caffe::BlobShape * s = blob_->mutable_shape();

	shape_.clear();
	for( int i = 0; i < s->dim_size(); i++ )
		shape_.push_back( s->dim( i ) );

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