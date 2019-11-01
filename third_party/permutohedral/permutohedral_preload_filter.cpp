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

#include "permutohedral_preload_filter.h"

#ifdef WIN32
inline int round(double X) { return int(X + .5); }
#endif

#ifdef __SSE__
// SSE Permutoheral lattice
#define SSE_PERMUTOHEDRAL
#endif

#if defined(SSE_PERMUTOHEDRAL)
#include <emmintrin.h>
#include <xmmintrin.h>
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#endif



/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

Permutohedral_preload::Permutohedral_preload(int N,int M,int d, bool with_blur) : N_(N), M_(M), with_blur_(with_blur),hash_table_(d,M,N) {}

void Permutohedral_preload::seqCompute(float* out, const float* in, int value_size, bool reverse, int start) const {
    constexpr int d_ = DIMENSION;
    // Shift all values by 1 such that -1 -> 0 (used for blurring)
    float* values = new float[(M_ + 2) * value_size];
    float* new_values = new float[(M_ + 2) * value_size];

    memset(values, 0, sizeof(float) * (M_ + 2) * value_size);
    memset(new_values, 0, sizeof(float) * (M_ + 2) * value_size);

    // Splatting
    for (int i = start; i < N_; i++) {
        for (int j = 0; j <= d_; j++) {
            const int o = offset_[i * (d_ + 1) + j] + 1;
            const float& w = barycentric_[i * (d_ + 1) + j];
            for (int k = 0; k < value_size; k++){
				values[o * value_size + k] += w * in[i * value_size + k];
			} 
        }
    }

    if (with_blur_) {
        for (int j = reverse ? d_ : 0; j <= d_ && j >= 0; reverse ? j-- : j++) {
            for (int i = 0; i < M_; i++) {
                float* old_val = values + (i + 1) * value_size;
                float* new_val = new_values + (i + 1) * value_size;

                int n1 = blur_neighbors_[j * M_ + i].n1 + 1;
                int n2 = blur_neighbors_[j * M_ + i].n2 + 1;
                float* n1_val = values + n1 * value_size;
                float* n2_val = values + n2 * value_size;
                for (int k = 0; k < value_size; k++) new_val[k] = old_val[k] + 0.5 * (n1_val[k] + n2_val[k]);
            }
            std::swap(values, new_values);
        }
    }
    // Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
    const float alpha = 1.0f / (1 + powf(2, -d_));

    // Slicing
    for (int i = 0; i < N_; i++) {
        for (int k = 0; k < value_size; k++) out[i * value_size + k] = 0;
        for (int j = 0; j <= d_; j++) {
            const int o = offset_[i * (d_ + 1) + j] + 1;
            const float& w = barycentric_[i * (d_ + 1) + j];
            for (int k = 0; k < value_size; k++)
                out[i * value_size + k] += w * values[o * value_size + k] * alpha;
        }
    }

    delete[] values;
    delete[] new_values;
}
#ifdef SSE_PERMUTOHEDRAL
void Permutohedral_preload::sseCompute(float* out, const float* in, int value_size, bool reverse, int start) const {
    constexpr int d_ = DIMENSION;
    const int sse_value_size = (value_size - 1) * sizeof(float) / sizeof(__m128) + 1;
    // Shift all values by 1 such that -1 -> 0 (used for blurring)
    __m128* sse_val = (__m128*)_mm_malloc(sse_value_size * sizeof(__m128), 16);
    __m128* values = (__m128*)_mm_malloc((M_ + 2) * sse_value_size * sizeof(__m128), 16);
    __m128* new_values = (__m128*)_mm_malloc((M_ + 2) * sse_value_size * sizeof(__m128), 16);

    __m128 Zero = _mm_set1_ps(0);

    for (int i = 0; i < (M_ + 2) * sse_value_size; i++){
		values[i] = new_values[i] = Zero;
	}
    for (int i = 0; i < sse_value_size; i++){
		sse_val[i] = Zero;
	}

    // Splatting
    for (int i = start; i < N_; i++) {
        memcpy(sse_val, in + i * value_size, value_size * sizeof(float));
        for (int j = 0; j <= d_; j++) {
            int o = offset_[i * (d_ + 1) + j] + 1;
            __m128 w = _mm_set1_ps(barycentric_[i * (d_ + 1) + j]);
            for (int k = 0; k < sse_value_size; k++){
				values[o * sse_value_size + k] += w * sse_val[k];
			}
        }
    }
    // Blurring
    if (with_blur_) {
        __m128 half = _mm_set1_ps(0.5);
        for (int j = reverse ? d_ : 0; j <= d_ && j >= 0; reverse ? j-- : j++) {
            for (int i = 0; i < M_; i++) {
                __m128* old_val = values + (i + 1) * sse_value_size;
                __m128* new_val = new_values + (i + 1) * sse_value_size;

                int n1 = blur_neighbors_[j * M_ + i].n1 + 1;
                int n2 = blur_neighbors_[j * M_ + i].n2 + 1;
                __m128* n1_val = values + n1 * sse_value_size;
                __m128* n2_val = values + n2 * sse_value_size;
                for (int k = 0; k < sse_value_size; k++)
                    new_val[k] = old_val[k] + half * (n1_val[k] + n2_val[k]);
            }
            std::swap(values, new_values);
        }
    }
    // Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
    constexpr float alpha = 1.0f / (1 + powf(2, -d_));

    // Slicing
    for (int i = 0; i < N_; i++) {
        for (int k = 0; k < sse_value_size; k++) sse_val[k] = Zero;
        for (int j = 0; j <= d_; j++) {
            int o = offset_[i * (d_ + 1) + j] + 1;
            __m128 w = _mm_set1_ps(barycentric_[i * (d_ + 1) + j] * alpha);
            for (int k = 0; k < sse_value_size; k++) sse_val[k] += w * values[o * sse_value_size + k];
        }
        memcpy(out + i * value_size, sse_val, value_size * sizeof(float));
    }

    _mm_free(sse_val);
    _mm_free(values);
    _mm_free(new_values);
}
#else
void Permutohedral::sseCompute(float* out, const float* in, int value_size, bool reverse, int start) const {
    seqCompute(out, in, value_size, reverse, start);
}
#endif
void Permutohedral_preload::compute(MatrixXf& out, const MatrixXf& in, bool reverse, int start) const {
    if (out.cols() != in.cols() || out.rows() != in.rows()) out = 0 * in;
    if (in.rows() <= 2)
        seqCompute(out.data(), in.data(), in.rows(), reverse, start);
    else
        sseCompute(out.data(), in.data(), in.rows(), reverse, start);
}
MatrixXf Permutohedral_preload::compute(const MatrixXf& in, bool reverse, int start) const {
    MatrixXf r;
    compute(r, in, reverse, start);
    return r;
}

int Permutohedral_preload::getLatticeSize() const { return M_; }

#ifdef SSE_PERMUTOHEDRAL
void Permutohedral_preload::init_with_val(const MatrixXf& feature,const MatrixXf& in, bool with_blur)  {
    // Compute the lattice coordinates for each feature [there is going to be a lot of magic here
    N_ = feature.cols();
	auto vd_ = in.rows();
    constexpr int d_ = DIMENSION;
    assert(d_ == feature.rows());
    with_blur_ = with_blur;
    HashTable hash_table(d_,vd_, N_ /**(d_+1)*/);

    constexpr int blocksize = sizeof(__m128) / sizeof(float);
    const __m128 invdplus1 = _mm_set1_ps(1.0f / (d_ + 1));
    const __m128 dplus1 = _mm_set1_ps(d_ + 1);
    const __m128 Zero = _mm_set1_ps(0);
    const __m128 One = _mm_set1_ps(1);

    // Allocate the class memory
    offset_.resize((d_ + 1) * (N_ + 16));
    std::fill(offset_.begin(), offset_.end(), 0);
    barycentric_.resize((d_ + 1) * (N_ + 16));
    std::fill(barycentric_.begin(), barycentric_.end(), 0);
    rank_.resize((d_ + 1) * (N_ + 16));

    // Allocate the local memory
    __m128 __attribute__((aligned(16))) scale_factor[d_];
    __m128 __attribute__((aligned(16))) f[d_];
    __m128 __attribute__((aligned(16))) elevated[d_ + 1];
    __m128 __attribute__((aligned(16))) rem0[d_ + 1];
    __m128 __attribute__((aligned(16))) rank[d_ + 1];
    float barycentric[(d_ + 2) * blocksize];
    short canonical[(d_ + 1) * (d_ + 1)];
    short key[d_ + 1];

    // Compute the canonical simplex
    for (int i = 0; i <= d_; i++) {
        for (int j = 0; j <= d_ - i; j++) {
            canonical[i * (d_ + 1) + j] = i;
        }
        for (int j = d_ - i + 1; j <= d_; j++) {
            canonical[i * (d_ + 1) + j] = i - (d_ + 1);
        }
    }

    // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
    float inv_std_dev = (with_blur_) ? sqrt(2.0 / 3.0) * (d_ + 1) : sqrt(1.0 / 6.0) * (d_ + 1);
    // Compute the diagonal part of E (p.5 in [Adams etal 2010])
    for (int i = 0; i < d_; i++) scale_factor[i] = _mm_set1_ps(1.0 / sqrt((i + 2) * (i + 1)) * inv_std_dev);

        // Setup the SSE rounding
#ifndef __SSE4_1__
    const unsigned int old_rounding = _mm_getcsr();
    _mm_setcsr((old_rounding & ~_MM_ROUND_MASK) | _MM_ROUND_NEAREST);
#endif

    // Compute the simplex each feature lies in
    for (int k = 0; k < N_; k += blocksize) {
        // Load the feature from memory
        float* ff = (float*)f;
        for (int j = 0; j < d_; j++)
            for (int i = 0; i < blocksize; i++) ff[j * blocksize + i] = k + i < N_ ? feature(j, k + i) : 0.0;

        // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])

        // sm contains the sum of 1..n of our faeture vector
        __m128 sm = Zero;
        for (int j = d_; j > 0; j--) {
            __m128 cf = f[j - 1] * scale_factor[j - 1];
            elevated[j] = sm - _mm_set1_ps(j) * cf;
            sm += cf;
        }
        elevated[0] = sm;

        // Find the closest 0-colored simplex through rounding
        __m128 sum = Zero;
        for (int i = 0; i <= d_; i++) {
            __m128 v = invdplus1 * elevated[i];
#ifdef __SSE4_1__
            v = _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT);
#else
            v = _mm_cvtepi32_ps(_mm_cvtps_epi32(v));
#endif
            rem0[i] = v * dplus1;
            sum += v;
        }

        // Find the simplex we are in and store it in rank (where rank describes what position coorinate i has
        // in the sorted order of the features values)
        for (int i = 0; i <= d_; i++) rank[i] = Zero;
        for (int i = 0; i < d_; i++) {
            __m128 di = elevated[i] - rem0[i];
            for (int j = i + 1; j <= d_; j++) {
                __m128 dj = elevated[j] - rem0[j];
                __m128 c = _mm_and_ps(One, _mm_cmplt_ps(di, dj));
                rank[i] += c;
                rank[j] += One - c;
            }
        }

        // If the point doesn't lie on the plane (sum != 0) bring it back
        for (int i = 0; i <= d_; i++) {
            rank[i] += sum;
            __m128 add = _mm_and_ps(dplus1, _mm_cmplt_ps(rank[i], Zero));
            __m128 sub = _mm_and_ps(dplus1, _mm_cmpge_ps(rank[i], dplus1));
            rank[i] += add - sub;
            rem0[i] += add - sub;
        }

        // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
        memset(barycentric, 0, sizeof(float) * (d_ + 2) * blocksize);
        for (int i = 0; i <= d_; i++) {
            __m128 v = (elevated[i] - rem0[i]) * invdplus1;

            // Didn't figure out how to SSE this
            float* fv = (float*)&v;
            float* frank = (float*)&rank[i];
            for (int j = 0; j < blocksize; j++) {
                int p = d_ - frank[j];
                barycentric[j * (d_ + 2) + p] += fv[j];
                barycentric[j * (d_ + 2) + p + 1] -= fv[j];
            }
        }

        // The rest is not SSE'd
        for (int j = 0; j < blocksize; j++) {
            // Wrap around
            barycentric[j * (d_ + 2) + 0] += 1 + barycentric[j * (d_ + 2) + d_ + 1];

            float* frank = (float*)rank;
            float* frem0 = (float*)rem0;
            // Compute all vertices and their offset
            for (int remainder = 0; remainder <= d_; remainder++) {
                for (int i = 0; i < d_; i++) {
                    key[i] = frem0[i * blocksize + j] +
                             canonical[remainder * (d_ + 1) + (int)frank[i * blocksize + j]];
                }
				auto offset_tmp = hash_table.find(key, true);
                offset_[(j + k) * (d_ + 1) + remainder] = offset_tmp;
                rank_[(j + k) * (d_ + 1) + remainder] = frank[remainder * blocksize + j];
                barycentric_[(j + k) * (d_ + 1) + remainder] = barycentric[j * (d_ + 2) + remainder];
				float w = barycentric[j * (d_ + 2) + remainder];
				for (int k1 = 0; k1 < M_; k++){
					hash_table.values_[offset_tmp  + k1] += w * in(k1,k+j);
				}
            }
        }
    }

    // Reset the SSE rounding
#ifndef __SSE4_1__
    _mm_setcsr(old_rounding);
#endif

    // This is normally fast enough so no SSE needed here
    // Find the Neighbors of each lattice point

    // Get the number of vertices in the lattice
    M_ = hash_table.size();

    if (with_blur_) {
        // Create the neighborhood structure
        blur_neighbors_.resize((d_ + 1) * M_);
		std::vector<float> new_values(M_ * hash_table.capacity_);
        short n1[d_ + 1];
        short n2[d_ + 1];
		auto &values = hash_table.values_;
        // For each of d+1 axes,
        for (int j = 0; j <= d_; j++) {
            for (int i = 0; i < hash_table.filled_; i++) {
                const short* key = hash_table.getKey(i);
                for (int k = 0; k < d_; k++) {
                    n1[k] = key[k] - 1;
                    n2[k] = key[k] + 1;
                }
                n1[j] = key[j] + d_;
                n2[j] = key[j] - d_;

                int n1_val = hash_table.find(n1,false)*M_;
                int n2_val = hash_table.find(n2,false)*M_;
		
				float n1_idx= 1;
				float n2_idx = 1;
				if(n1_val < 0){
					n1_val = 0;
					n1_idx = 0;
				}
				if(n2_val < 0){
					n2_val = 0;
					n2_val = 0;
				}
				// Mix values of the three vertices
                for (int k = 0; k < M_; k++)
                {
                    new_values[i*M_+k] = (0.25 * values[n1_val + k]*n1_idx + 0.5 * values[n1_val + k] + 0.25 * values[n2_val + k]*n2_idx);
                }
            }
        }
		std::swap(hash_table.values_, new_values);
    }
}
#else
//NOT IMPLEMENTED
void Permutohedral_preload::init_with_val(const MatrixXf& feature,const MatrixXf& in, bool with_blur) {
    /*
    // Compute the lattice coordinates for each feature [there is going to be a lot of magic here
    N_ = feature.cols();
	M_ = in.rows();
    constexpr int d_ = DIMENSION;
    assert(d_ == feature.rows());
    with_blur_ = with_blur;
    HashTable hash_table(d_,M_, N_ * (d_ + 1));

    // Allocate the class memory
    offset_.resize((d_ + 1) * N_);
    rank_.resize((d_ + 1) * N_);
    barycentric_.resize((d_ + 1) * N_);

    // Allocate the local memory
    float scale_factor[d_];
    float elevated[d_ + 1];
    float rem0[d_ + 1];
    float barycentric[d_ + 2];
    short rank[d_ + 1];
    short canonical[(d_ + 1) * (d_ + 1)];
    short key[d_ + 1];

    // Compute the canonical simplex
    for (int i = 0; i <= d_; i++) {
        for (int j = 0; j <= d_ - i; j++) canonical[i * (d_ + 1) + j] = i;
        for (int j = d_ - i + 1; j <= d_; j++) canonical[i * (d_ + 1) + j] = i - (d_ + 1);
    }

    // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
    float inv_std_dev = (with_blur_) ? sqrt(2.0 / 3.0) * (d_ + 1) : sqrt(1.0 / 6.0) * (d_ + 1);
    // Compute the diagonal part of E (p.5 in [Adams etal 2010])
    for (int i = 0; i < d_; i++) scale_factor[i] = 1.0 / sqrt(double((i + 2) * (i + 1))) * inv_std_dev;

    // Compute the simplex each feature lies in
    for (int k = 0; k < N_; k++) {
        // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
        const float* f = &feature(0, k);

        // sm contains the sum of 1..n of our faeture vector
        float sm = 0;
        for (int j = d_; j > 0; j--) {
            float cf = f[j - 1] * scale_factor[j - 1];
            elevated[j] = sm - j * cf;
            sm += cf;
        }
        elevated[0] = sm;

        // Find the closest 0-colored simplex through rounding
        float down_factor = 1.0f / (d_ + 1);
        float up_factor = (d_ + 1);
        int sum = 0;
        for (int i = 0; i <= d_; i++) {
            // int rd1 = round( down_factor * elevated[i]);
            int rd2;
            float v = down_factor * elevated[i];
            float up = ceilf(v) * up_factor;
            float down = floorf(v) * up_factor;
            if (up - elevated[i] < elevated[i] - down)
                rd2 = (short)up;
            else
                rd2 = (short)down;

            // if(rd1!=rd2)
            //	break;

            rem0[i] = rd2;
            sum += rd2 * down_factor;
        }

        // Find the simplex we are in and store it in rank (where rank describes what position coorinate i has
        // in the sorted order of the features values)
        for (int i = 0; i <= d_; i++) rank[i] = 0;
        for (int i = 0; i < d_; i++) {
            double di = elevated[i] - rem0[i];
            for (int j = i + 1; j <= d_; j++)
                if (di < elevated[j] - rem0[j])
                    rank[i]++;
                else
                    rank[j]++;
        }

        // If the point doesn't lie on the plane (sum != 0) bring it back
        for (int i = 0; i <= d_; i++) {
            rank[i] += sum;
            if (rank[i] < 0) {
                rank[i] += d_ + 1;
                rem0[i] += d_ + 1;
            } else if (rank[i] > d_) {
                rank[i] -= d_ + 1;
                rem0[i] -= d_ + 1;
            }
        }

        // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
        for (int i = 0; i <= d_ + 1; i++) barycentric[i] = 0;
        for (int i = 0; i <= d_; i++) {
            float v = (elevated[i] - rem0[i]) * down_factor;
            barycentric[d_ - rank[i]] += v;
            barycentric[d_ - rank[i] + 1] -= v;
        }
        // Wrap around
        barycentric[0] += 1.0 + barycentric[d_ + 1];

        // Compute all vertices and their offset
        for (int remainder = 0; remainder <= d_; remainder++) {
            for (int i = 0; i < d_; i++) key[i] = rem0[i] + canonical[remainder * (d_ + 1) + rank[i]];
            offset_[k * (d_ + 1) + remainder] = hash_table.find(key, true);
            rank_[k * (d_ + 1) + remainder] = rank[remainder];
            barycentric_[k * (d_ + 1) + remainder] = barycentric[remainder];
        }
    }

    // Find the Neighbors of each lattice point

    // Get the number of vertices in the lattice
    M_ = hash_table.size();

    if (with_blur_) {
        // Create the neighborhood structure
        blur_neighbors_.resize((d_ + 1) * M_);

        short n1[d_ + 1];
        short n2[d_ + 1];

        // For each of d+1 axes,
        for (int j = 0; j <= d_; j++) {
            for (int i = 0; i < M_; i++) {
                const short* key = hash_table.getKey(i);
                for (int k = 0; k < d_; k++) {
                    n1[k] = key[k] - 1;
                    n2[k] = key[k] + 1;
                }
                n1[j] = key[j] + d_;
                n2[j] = key[j] - d_;

                blur_neighbors_[j * M_ + i].n1 = hash_table.find(n1);
                blur_neighbors_[j * M_ + i].n2 = hash_table.find(n2);
            }
        }
    }
    */
}
#endif

#ifdef SSE_PERMUTOHEDRAL
void Permutohedral_preload::apply(float* out, const MatrixXf& feature){
    // Compute the lattice coordinates for each feature [there is going to be a lot of magic here
    N_ = feature.cols();
    constexpr int d_ = DIMENSION;
    assert(d_ == feature.rows());
    auto vd_ = M_;
    //HashTable hash_table(d_,vd_, N_ /**(d_+1)*/);
    for(auto k1 = 0;k1<feature.cols()*(M_-1);k1++){
        out[k1] = 0;
    }
    constexpr int blocksize = sizeof(__m128) / sizeof(float);
    const __m128 invdplus1 = _mm_set1_ps(1.0f / (d_ + 1));
    const __m128 dplus1 = _mm_set1_ps(d_ + 1);
    const __m128 Zero = _mm_set1_ps(0);
    const __m128 One = _mm_set1_ps(1);

    // Allocate the class memory
    offset_.resize((d_ + 1) * (N_ + 16));
    std::fill(offset_.begin(), offset_.end(), 0);
    barycentric_.resize((d_ + 1) * (N_ + 16));
    std::fill(barycentric_.begin(), barycentric_.end(), 0);
    rank_.resize((d_ + 1) * (N_ + 16));

    // Allocate the local memory
    __m128 __attribute__((aligned(16))) scale_factor[d_];
    __m128 __attribute__((aligned(16))) f[d_];
    __m128 __attribute__((aligned(16))) elevated[d_ + 1];
    __m128 __attribute__((aligned(16))) rem0[d_ + 1];
    __m128 __attribute__((aligned(16))) rank[d_ + 1];
    float barycentric[(d_ + 2) * blocksize];
    short canonical[(d_ + 1) * (d_ + 1)];
    short key[d_ + 1];

    // Compute the canonical simplex
    for (int i = 0; i <= d_; i++) {
        for (int j = 0; j <= d_ - i; j++) {
            canonical[i * (d_ + 1) + j] = i;
        }
        for (int j = d_ - i + 1; j <= d_; j++) {
            canonical[i * (d_ + 1) + j] = i - (d_ + 1);
        }
    }

    // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
    float inv_std_dev = (with_blur_) ? sqrt(2.0 / 3.0) * (d_ + 1) : sqrt(1.0 / 6.0) * (d_ + 1);
    // Compute the diagonal part of E (p.5 in [Adams etal 2010])
    for (int i = 0; i < d_; i++) scale_factor[i] = _mm_set1_ps(1.0 / sqrt((i + 2) * (i + 1)) * inv_std_dev);

        // Setup the SSE rounding
#ifndef __SSE4_1__
    const unsigned int old_rounding = _mm_getcsr();
    _mm_setcsr((old_rounding & ~_MM_ROUND_MASK) | _MM_ROUND_NEAREST);
#endif

    // Compute the simplex each feature lies in
    for (int k = 0; k < N_; k += blocksize) {
        // Load the feature from memory
        float* ff = (float*)f;
        for (int j = 0; j < d_; j++)
            for (int i = 0; i < blocksize; i++) ff[j * blocksize + i] = k + i < N_ ? feature(j, k + i) : 0.0;

        // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])

        // sm contains the sum of 1..n of our faeture vector
        __m128 sm = Zero;
        for (int j = d_; j > 0; j--) {
            __m128 cf = f[j - 1] * scale_factor[j - 1];
            elevated[j] = sm - _mm_set1_ps(j) * cf;
            sm += cf;
        }
        elevated[0] = sm;

        // Find the closest 0-colored simplex through rounding
        __m128 sum = Zero;
        for (int i = 0; i <= d_; i++) {
            __m128 v = invdplus1 * elevated[i];
#ifdef __SSE4_1__
            v = _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT);
#else
            v = _mm_cvtepi32_ps(_mm_cvtps_epi32(v));
#endif
            rem0[i] = v * dplus1;
            sum += v;
        }

        // Find the simplex we are in and store it in rank (where rank describes what position coorinate i has
        // in the sorted order of the features values)
        for (int i = 0; i <= d_; i++) rank[i] = Zero;
        for (int i = 0; i < d_; i++) {
            __m128 di = elevated[i] - rem0[i];
            for (int j = i + 1; j <= d_; j++) {
                __m128 dj = elevated[j] - rem0[j];
                __m128 c = _mm_and_ps(One, _mm_cmplt_ps(di, dj));
                rank[i] += c;
                rank[j] += One - c;
            }
        }

        // If the point doesn't lie on the plane (sum != 0) bring it back
        for (int i = 0; i <= d_; i++) {
            rank[i] += sum;
            __m128 add = _mm_and_ps(dplus1, _mm_cmplt_ps(rank[i], Zero));
            __m128 sub = _mm_and_ps(dplus1, _mm_cmpge_ps(rank[i], dplus1));
            rank[i] += add - sub;
            rem0[i] += add - sub;
        }

        // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
        memset(barycentric, 0, sizeof(float) * (d_ + 2) * blocksize);
        for (int i = 0; i <= d_; i++) {
            __m128 v = (elevated[i] - rem0[i]) * invdplus1;

            // Didn't figure out how to SSE this
            float* fv = (float*)&v;
            float* frank = (float*)&rank[i];
            for (int j = 0; j < blocksize; j++) {
                int p = d_ - frank[j];
                barycentric[j * (d_ + 2) + p] += fv[j];
                barycentric[j * (d_ + 2) + p + 1] -= fv[j];
            }
        }

        // The rest is not SSE'd
        for (int j = 0; j < blocksize; j++) {
            // Wrap around
            barycentric[j * (d_ + 2) + 0] += 1 + barycentric[j * (d_ + 2) + d_ + 1];

            float* frank = (float*)rank;
            float* frem0 = (float*)rem0;
            // Compute all vertices and their offset
            for (int remainder = 0; remainder <= d_; remainder++) {
                for (int i = 0; i < d_; i++) {
                    key[i] = frem0[i * blocksize + j] +
                             canonical[remainder * (d_ + 1) + (int)frank[i * blocksize + j]];
                }
				auto offset_tmp = hash_table_.find(key, false);

                //offset_[(j + k) * (d_ + 1) + remainder] = offset_tmp;
                //rank_[(j + k) * (d_ + 1) + remainder] = frank[remainder * blocksize + j];
                //barycentric_[(j + k) * (d_ + 1) + remainder] = barycentric[j * (d_ + 2) + remainder];
				float w = barycentric[j * (d_ + 2) + remainder];
				for (int k1 = 0; k1 < M_; k++){
                    if(offset_tmp > 0)
                        out[offset_tmp  + k1] += w*hash_table_.values_[offset_tmp  + k1];
					    //hash_table.values_[offset_tmp  + k1] += w * in(k1,k+j);
				}
            }
        }
    }

    // Reset the SSE rounding
#ifndef __SSE4_1__
    _mm_setcsr(old_rounding);
#endif
}
#else
//NOT IMPLEMENTED
void Permutohedral_preload::apply(const MatrixXf& feature,const MatrixXf& in, bool with_blur) {
    // Compute the lattice coordinates for each feature [there is going to be a lot of magic here
  /*  N_ = feature.cols();
	M_ = in.rows();
    constexpr int d_ = DIMENSION;
    assert(d_ == feature.rows());
    with_blur_ = with_blur;
    HashTable hash_table(d_,M_, N_ * (d_ + 1));

    // Allocate the class memory
    offset_.resize((d_ + 1) * N_);
    rank_.resize((d_ + 1) * N_);
    barycentric_.resize((d_ + 1) * N_);

    // Allocate the local memory
    float scale_factor[d_];
    float elevated[d_ + 1];
    float rem0[d_ + 1];
    float barycentric[d_ + 2];
    short rank[d_ + 1];
    short canonical[(d_ + 1) * (d_ + 1)];
    short key[d_ + 1];

    // Compute the canonical simplex
    for (int i = 0; i <= d_; i++) {
        for (int j = 0; j <= d_ - i; j++) canonical[i * (d_ + 1) + j] = i;
        for (int j = d_ - i + 1; j <= d_; j++) canonical[i * (d_ + 1) + j] = i - (d_ + 1);
    }

    // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
    float inv_std_dev = (with_blur_) ? sqrt(2.0 / 3.0) * (d_ + 1) : sqrt(1.0 / 6.0) * (d_ + 1);
    // Compute the diagonal part of E (p.5 in [Adams etal 2010])
    for (int i = 0; i < d_; i++) scale_factor[i] = 1.0 / sqrt(double((i + 2) * (i + 1))) * inv_std_dev;

    // Compute the simplex each feature lies in
    for (int k = 0; k < N_; k++) {
        // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
        const float* f = &feature(0, k);

        // sm contains the sum of 1..n of our faeture vector
        float sm = 0;
        for (int j = d_; j > 0; j--) {
            float cf = f[j - 1] * scale_factor[j - 1];
            elevated[j] = sm - j * cf;
            sm += cf;
        }
        elevated[0] = sm;

        // Find the closest 0-colored simplex through rounding
        float down_factor = 1.0f / (d_ + 1);
        float up_factor = (d_ + 1);
        int sum = 0;
        for (int i = 0; i <= d_; i++) {
            // int rd1 = round( down_factor * elevated[i]);
            int rd2;
            float v = down_factor * elevated[i];
            float up = ceilf(v) * up_factor;
            float down = floorf(v) * up_factor;
            if (up - elevated[i] < elevated[i] - down)
                rd2 = (short)up;
            else
                rd2 = (short)down;

            // if(rd1!=rd2)
            //	break;

            rem0[i] = rd2;
            sum += rd2 * down_factor;
        }

        // Find the simplex we are in and store it in rank (where rank describes what position coorinate i has
        // in the sorted order of the features values)
        for (int i = 0; i <= d_; i++) rank[i] = 0;
        for (int i = 0; i < d_; i++) {
            double di = elevated[i] - rem0[i];
            for (int j = i + 1; j <= d_; j++)
                if (di < elevated[j] - rem0[j])
                    rank[i]++;
                else
                    rank[j]++;
        }

        // If the point doesn't lie on the plane (sum != 0) bring it back
        for (int i = 0; i <= d_; i++) {
            rank[i] += sum;
            if (rank[i] < 0) {
                rank[i] += d_ + 1;
                rem0[i] += d_ + 1;
            } else if (rank[i] > d_) {
                rank[i] -= d_ + 1;
                rem0[i] -= d_ + 1;
            }
        }

        // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
        for (int i = 0; i <= d_ + 1; i++) barycentric[i] = 0;
        for (int i = 0; i <= d_; i++) {
            float v = (elevated[i] - rem0[i]) * down_factor;
            barycentric[d_ - rank[i]] += v;
            barycentric[d_ - rank[i] + 1] -= v;
        }
        // Wrap around
        barycentric[0] += 1.0 + barycentric[d_ + 1];

        // Compute all vertices and their offset
        for (int remainder = 0; remainder <= d_; remainder++) {
            for (int i = 0; i < d_; i++) key[i] = rem0[i] + canonical[remainder * (d_ + 1) + rank[i]];
            offset_[k * (d_ + 1) + remainder] = hash_table.find(key, true);
            rank_[k * (d_ + 1) + remainder] = rank[remainder];
            barycentric_[k * (d_ + 1) + remainder] = barycentric[remainder];
        }
    }

    // Find the Neighbors of each lattice point

    // Get the number of vertices in the lattice
    M_ = hash_table.size();

    if (with_blur_) {
        // Create the neighborhood structure
        blur_neighbors_.resize((d_ + 1) * M_);

        short n1[d_ + 1];
        short n2[d_ + 1];

        // For each of d+1 axes,
        for (int j = 0; j <= d_; j++) {
            for (int i = 0; i < M_; i++) {
                const short* key = hash_table.getKey(i);
                for (int k = 0; k < d_; k++) {
                    n1[k] = key[k] - 1;
                    n2[k] = key[k] + 1;
                }
                n1[j] = key[j] + d_;
                n2[j] = key[j] - d_;

                blur_neighbors_[j * M_ + i].n1 = hash_table.find(n1);
                blur_neighbors_[j * M_ + i].n2 = hash_table.find(n2);
            }
        }
    }*/
}
#endif