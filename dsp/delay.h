#ifndef SIGNALSMITH_DSP_DELAY_H
#define SIGNALSMITH_DSP_DELAY_H
#include <stdlib.h>
#include <immintrin.h>
/// Holds a view for a particular position in the buffer
class View {
	public:
		__bfloat16 *buffer = nullptr;
		int bufferMask;
		unsigned bufferIndex = 0;
		View(__bfloat16* buffer, int bufferIndex, int bufferMask, int offset=0) : buffer(buffer), bufferIndex(bufferIndex + (unsigned)offset), bufferMask(bufferMask) {}
			
		__bfloat16 & operator[](int offset) {
			return buffer[(bufferIndex + (unsigned)offset)&bufferMask];
		}
};
class MultiBuffer {
	public:
		int channels, stride;
		int bufferIndex;
		int bufferMask;
		int bufferLength; 
		__bfloat16* buffer;

		MultiBuffer(int channels=0, int capacity=0) : channels(channels), stride(capacity), buffer((__bfloat16*)malloc(sizeof(__bfloat16)*channels*capacity)) {}

		void resize(int nChannels, int capacity, __bfloat16 value) {
			channels = nChannels;
			stride = capacity;
			//buffer.resize(channels*capacity, value);
			bufferLength = 1;
			while (bufferLength < channels*capacity) 
				bufferLength *= 2;
			buffer=(__bfloat16*)malloc(sizeof(__bfloat16)*bufferLength);
			//Loop di assegnamento potenzialmente parallelizzabile in SIMD
			for(int i=0;i<bufferLength; i++)
				buffer[i]=value;
			bufferMask = bufferLength - 1;
			bufferIndex = 0;
		}
		View operator[](int channel) {
			return View(buffer, bufferIndex, bufferMask,channel*stride);
		}
};
#endif // include guard
