#ifndef SIGNALSMITH_DSP_DELAY_H
#define SIGNALSMITH_DSP_DELAY_H
#include <stdlib.h>
//Riscrivere queste classi in modo meno OOP: togliere gli operatori, togliere la classe innestata View
class Buffer {
		int bufferIndex;
		int bufferMask;
		int bufferLength; 
		double* buffer;
	public:
		Buffer(int minCapacity=0) {
			resize(minCapacity, 0);
		}
		// But moving one is fine
		Buffer(Buffer &&other) = default;
		Buffer & operator =(Buffer &&other) = default;

		void resize(int minCapacity, double value=0) {
			bufferLength = 1;
			while (bufferLength < minCapacity) 
				bufferLength *= 2;
			buffer=(double*)malloc(sizeof(double)*bufferLength);
			//Loop di assegnamento potenzialmente parallelizzabile in SIMD
			for(int i=0;i<bufferLength; i++)
				buffer[i]=value;
			bufferMask = bufferLength - 1;
			bufferIndex = 0;
		}
		void reset(double value) {
			if(buffer!=NULL)
				free(buffer);
			buffer=(double*)malloc(sizeof(double)*bufferLength);
			//Idem come nel metodo precedente
			for(int i=0;i<bufferLength; i++)
				buffer[i]=value;
		}

		/// Holds a view for a particular position in the buffer
		class View {
			Buffer *buffer = nullptr;
			unsigned bufferIndex = 0;
		public:
			View(Buffer &buffer, int offset=0) : buffer(&buffer), bufferIndex(buffer.bufferIndex + (unsigned)offset) {}
			View(const View &other, int offset=0) : buffer(other.buffer), bufferIndex(other.bufferIndex + (unsigned)offset) {}
			View & operator =(const View &other) {
				buffer = other.buffer;
				bufferIndex = other.bufferIndex;
				return *this;
			}
			
			double & operator[](int offset) {
				return buffer->buffer[(bufferIndex + (unsigned)offset)&buffer->bufferMask];
			}

			View operator +(int offset) {
				return View(*this, offset);
			}
			View operator -(int offset) const {
				return View(*this, -offset);
			}
		};

		double & operator[](int offset) {
			return buffer[(bufferIndex + (unsigned)offset)&bufferMask];
		}
		
		Buffer & operator ++() {
			++bufferIndex;
			return *this;
		}
		Buffer & operator +=(int i) {
			bufferIndex += (unsigned)i;
			return *this;
		}
		Buffer & operator --() {
			--bufferIndex;
			return *this;
		}
		Buffer & operator -=(int i) {
			bufferIndex -= (unsigned)i;
			return *this;
		}

		View operator ++(int) {
			View view(*this);
			++bufferIndex;
			return view;
		}
		View operator +(int i) {
			return View(*this, i);
		}
		View operator --(int) {
			View view(*this);
			--bufferIndex;
			return view;
		}
		View operator -(int i) {
			return View(*this, -i);
		}
};
class MultiBuffer {
	public:
		int channels, stride;
		Buffer buffer;

		MultiBuffer(int channels=0, int capacity=0) : channels(channels), stride(capacity), buffer(channels*capacity) {}

		void resize(int nChannels, int capacity, double value) {
			channels = nChannels;
			stride = capacity;
			buffer.resize(channels*capacity, value);
		}

		Buffer::View operator[](int channel) {
			return buffer + channel*stride;
		}
		
		MultiBuffer & operator ++() {
			++buffer;
			return *this;
		}
		MultiBuffer & operator +=(int i) {
			buffer += i;
		return *this;
	}
};
#endif // include guard
