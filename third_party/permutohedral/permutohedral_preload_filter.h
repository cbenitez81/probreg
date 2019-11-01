/*
    Copyright (c) 2013, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <Eigen/Core>
using namespace Eigen;

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

static const int DIMENSION = 3;

/************************************************/
/***                Hash Table                ***/
/************************************************/

class HashTable {
   protected:
    void grow() {
        // Create the new memory and copy the values in
        int old_capacity = capacity_;
        capacity_ *= 2;
        std::vector<short> old_keys((old_capacity + 10) * key_size_);
		std::vector<float> old_values((old_capacity + 10) * value_size_);
        std::copy(keys_.begin(), keys_.end(), old_keys.begin());
		std::copy(values_.begin(), values_.end(), old_values.begin());
        std::vector<int> old_table(capacity_, -1);

        // Swap the memory
        table_.swap(old_table);
        keys_.swap(old_keys);
		values_.swap(old_values);

        // Reinsert each element
        for (int i = 0; i < old_capacity; i++)
            if (old_table[i] >= 0) {
                int e = old_table[i];
                size_t h = hash(getKey(e)) % capacity_;
                for (; table_[h] >= 0; h = h < capacity_ - 1 ? h + 1 : 0)
                    ;
                table_[h] = e;
            }
    }
    size_t hash(const short* k) {
        size_t r = 0;
        for (size_t i = 0; i < key_size_; i++) {
            r += k[i];
            r *= 1664525;
        }
        return r;
    }

   public:
    size_t key_size_, filled_, capacity_;
    size_t value_size_;
    std::vector<short> keys_;
    std::vector<int> table_;
    std::vector<float> values_;
    explicit HashTable(int key_size, int value_size, int n_elements)
        : key_size_(key_size),
          value_size_(value_size),
          filled_(0),
          capacity_(2 * n_elements),
          keys_((capacity_ / 2 + 10) * key_size_),
          table_(2 * n_elements, -1),
          values_((capacity_ / 2 + 10) * value_size_) {}
    int size() const { return filled_; }
    void reset() {
        filled_ = 0;
        std::fill(table_.begin(), table_.end(), -1);
    }
    int find(const short* k, bool create = false) {
        if (2 * filled_ >= capacity_) grow();
        // Get the hash value
        size_t h = hash(k) % capacity_;
        // Find the element with he right key, using linear probing
        while (1) {
            int e = table_[h];
            if (e == -1) {
                if (create) {
                    // Insert a new key and return the new id
                    for (size_t i = 0; i < key_size_; i++) keys_[filled_ * key_size_ + i] = k[i];
                    return table_[h] = filled_++;
                } else
                    return -1;
            }
            // Check if the current key is The One
            bool good = true;
            for (size_t i = 0; i < key_size_ && good; i++)
                if (keys_[e * key_size_ + i] != k[i]) good = false;
            if (good) return e;
            // Continue searching
            h++;
            if (h == capacity_) h = 0;
        }
    }
    const short* getKey(int i) const { return &keys_[i * key_size_]; }
};

class Permutohedral_preload
{
protected:
	struct Neighbors{
		int n1, n2;
		Neighbors( int n1=0, int n2=0 ):n1(n1),n2(n2){
		}
	};
	std::vector<int> offset_, rank_;
    std::vector<int> val_offset_;
	std::vector<float> barycentric_;
	std::vector<Neighbors> blur_neighbors_;
    HashTable hash_table_;
	// Number of elements, size of sparse discretized space, dimension of features
	int N_, M_;
	bool with_blur_;
	void sseCompute ( float* out, const float* in, int value_size, bool reverse=false, int start=0 ) const;
	void seqCompute ( float* out, const float* in, int value_size, bool reverse=false, int start=0 ) const;
public:
	Permutohedral_preload(int N,int M,int d, bool with_blur);
	void init ( const MatrixXf & features, const MatrixXf &in, bool with_blur = true );
    void init_with_val(const MatrixXf& feature,const MatrixXf& in, bool with_blur);
    void apply(float* out,const MatrixXf& feature);
	int getLatticeSize() const;
	MatrixXf compute ( const MatrixXf & v, bool reverse=false, int start=0 ) const;
	void compute ( MatrixXf & out, const MatrixXf & in, bool reverse=false, int start=0 ) const;
};
