#ifndef SIGNALSMITH_DSP_DELAY_H
#define SIGNALSMITH_DSP_DELAY_H
#include <stdlib.h>
/// Holds a view for a particular position in the buffer
class View {
	public:
		double *buffer = nullptr;
		int bufferMask;
		unsigned bufferIndex = 0;
		View(double* buffer, int bufferIndex, int bufferMask, int offset=0) : buffer(buffer), bufferIndex(bufferIndex + (unsigned)offset), bufferMask(bufferMask) {}
			
		double & operator[](int offset) {
			return buffer[(bufferIndex + (unsigned)offset)&bufferMask];
		}
};
class MultiBuffer {
	public:
		int channels, stride;
		int bufferIndex;
		int bufferMask;
		int bufferLength; 
		double* buffer;

		MultiBuffer(int channels=0, int capacity=0) : channels(channels), stride(capacity), buffer((double*)malloc(sizeof(double)*channels*capacity)) {}

		void resize(int nChannels, int capacity, double value) {
			channels = nChannels;
			stride = capacity;
			//buffer.resize(channels*capacity, value);
			bufferLength = 1;
			while (bufferLength < channels*capacity) 
				bufferLength *= 2;
			buffer=(double*)malloc(sizeof(double)*bufferLength);
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
