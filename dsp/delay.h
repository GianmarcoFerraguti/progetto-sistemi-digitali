#ifndef SIGNALSMITH_DSP_DELAY_H
#define SIGNALSMITH_DSP_DELAY_H
#include <stdlib.h>
#include <immintrin.h>
#include "../util/simd_consts.h"
/// Holds a view for a particular position in the buffer
class View {
	public:
		float *buffer = nullptr;
		int bufferMask;
		unsigned bufferIndex = 0;
		View(float* buffer, int bufferIndex, int bufferMask, int offset=0) : buffer(buffer), bufferIndex(bufferIndex + (unsigned)offset), bufferMask(bufferMask) {}
			
		float & operator[](int offset) {
			return buffer[(bufferIndex + (unsigned)offset)&bufferMask];
		}
};
class MultiBuffer {
	public:
		int channels, stride;
		int bufferIndex;
		int bufferMask;
		int bufferLength; 
		float* buffer;

		MultiBuffer(int channels=0, int capacity=0) : channels(channels), stride(capacity), buffer((float*)malloc(sizeof(float)*channels*capacity)) {}

		void resize(int nChannels, int capacity, float value) {
			channels = nChannels;
			stride = capacity;
			//buffer.resize(channels*capacity, value);
			bufferLength = 1;
			while (bufferLength < channels*capacity) 
				bufferLength *= 2;
			buffer=(float*)malloc(sizeof(float)*bufferLength);
			__m512* pBuffer = (__m512*)buffer;
			__m512 XMM_BUFF;
			for(int i=0;i<bufferLength/AVX_DATA_LANE; i++)
			{
				XMM_BUFF = _mm512_set1_ps(value);
				_mm512_storeu_ps(pBuffer + i, XMM_BUFF);
			}
			bufferMask = bufferLength - 1;
			bufferIndex = 0;
		}
		View operator[](int channel) {
			return View(buffer, bufferIndex, bufferMask,channel*stride);
		}
};
#endif // include guard
