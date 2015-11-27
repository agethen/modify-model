#ifndef BLOB_HH
#define BLOB_HH

#include <random>
#include <iostream>
#include <memory>
#include <caffe/proto/caffe.pb.h>


#define FLAG_ZERO	 		0x00
#define FLAG_AVERAGE	0x01
#define FLAG_REPEAT		0x02

#define FLAG_RGB			0x10

template <typename Dtype>
class BlobHandler{
	public:
		BlobHandler();
		~BlobHandler();

		int size();		

		inline std::vector<int> shape(){
			return shape_;
		}

		inline void setShape( std::vector<int> & shape ){
			shape_ = shape;
		}

		inline std::shared_ptr<std::vector<Dtype>> data(){
			return data_;
		}

		inline void setData( std::shared_ptr<std::vector<Dtype>> data, std::vector<int> shape ){
			this->setShape( shape );
			data_ = std::move( data );
		}

		void fromBlob( caffe::BlobProto * );

		inline caffe::BlobProto * blob(){
			if( !blob_ )
				blob_ = new caffe::BlobProto();

			this->update();
			return blob_;
		}
		void update();

		void zero();
		void random( float, float );

		void readWithShape( std::shared_ptr<std::vector<Dtype>>, std::vector<int> );
		
		inline void readFrom( BlobHandler & b ){
			b.readWithShape( this->data(), this->shape() );
		}

		void inflateAlongAxis( int axis, int new_value, int flag );

		void print();
		void print( std::vector<int> );
	private:
		void recursiveRead( std::shared_ptr<std::vector<Dtype>>, std::shared_ptr<std::vector<Dtype>>,
					 int, int, std::vector<int>, std::vector<int> );
		void recursivePrint( int, std::vector<int> );
		void repeatPattern( std::shared_ptr<std::vector<Dtype>>, std::shared_ptr<std::vector<Dtype>>, int64_t, int64_t );

		std::vector<int> parseShape();
		void setBlobShape();

		std::vector<int> shape_;

		caffe::BlobProto * blob_;
		std::shared_ptr<std::vector<Dtype>> data_;
};
#endif