#ifndef BLOB_HH
#define BLOB_HH

#include <random>
#include <iostream>
#include <memory>
#include <caffe/proto/caffe.pb.h>

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

		inline void fromBlob( caffe::BlobProto * );

		inline caffe::BlobProto * blob(){
			if( !blob_ )
				blob_ = new caffe::BlobProto();

			this->update();
			return blob_;
		}

		void zero();
		void random( float, float );

		void readWithShape( std::shared_ptr<std::vector<Dtype>>, std::vector<int> );

		inline void readFrom( BlobHandler & b ){
			b.readWithShape( this->data(), this->shape() );
		}

		void print();
		void print( std::vector<int> );
	private:
		void recursiveRead( std::shared_ptr<std::vector<Dtype>>, std::shared_ptr<std::vector<Dtype>>,
					 int, int, std::vector<int>, std::vector<int> );
		void recursivePrint( int, std::vector<int> );
		void update();

		std::vector<int> shape_;

		caffe::BlobProto * blob_;
		std::shared_ptr<std::vector<Dtype>> data_;
};
#endif