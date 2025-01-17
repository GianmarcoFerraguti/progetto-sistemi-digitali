#include "./common.h"

#ifndef SIGNALSMITH_DSP_DELAY_H
#define SIGNALSMITH_DSP_DELAY_H

#include <vector>
#include "./fft.h"
#include "./windows.h"
template<typename Sample>
class Buffer {
		unsigned bufferIndex;
		unsigned bufferMask;
		std::vector<Sample> buffer;
	public:
		Buffer(int minCapacity=0) {
			resize(minCapacity);
		}
		// We shouldn't accidentally copy a delay buffer
		Buffer(const Buffer &other) = delete;
		Buffer & operator =(const Buffer &other) = delete;
		// But moving one is fine
		Buffer(Buffer &&other) = default;
		Buffer & operator =(Buffer &&other) = default;

		void resize(int minCapacity, Sample value=Sample()) {
			int bufferLength = 1;
			while (bufferLength < minCapacity) 
				bufferLength *= 2;
			buffer.assign(bufferLength, value); //Replaces vector content and size
			bufferMask = unsigned(bufferLength - 1);
			bufferIndex = 0;
		}
		void reset(Sample value=Sample()) {
			buffer.assign(buffer.size(), value);
		}

		/// Holds a view for a particular position in the buffer
		template<bool isConst>
		class View {
			using CBuffer = typename std::conditional<isConst, const Buffer, Buffer>::type;
			using CSample = typename std::conditional<isConst, const Sample, Sample>::type;
			CBuffer *buffer = nullptr;
			unsigned bufferIndex = 0;
		public:
			View(CBuffer &buffer, int offset=0) : buffer(&buffer), bufferIndex(buffer.bufferIndex + (unsigned)offset) {}
			View(const View &other, int offset=0) : buffer(other.buffer), bufferIndex(other.bufferIndex + (unsigned)offset) {}
			View & operator =(const View &other) {
				buffer = other.buffer;
				bufferIndex = other.bufferIndex;
				return *this;
			}
			
			CSample & operator[](int offset) {
				return buffer->buffer[(bufferIndex + (unsigned)offset)&buffer->bufferMask];
			}
			const Sample & operator[](int offset) const {
				return buffer->buffer[(bufferIndex + (unsigned)offset)&buffer->bufferMask];
			}

			/// Write data into the buffer
			template<typename Data>
			void write(Data &&data, int length) {
				for (int i = 0; i < length; ++i) {
					(*this)[i] = data[i];
				}
			}
			/// Read data out from the buffer
			template<typename Data>
			void read(int length, Data &&data) const {
				for (int i = 0; i < length; ++i) {
					data[i] = (*this)[i];
				}
			}

			View operator +(int offset) const {
				return View(*this, offset);
			}
			View operator -(int offset) const {
				return View(*this, -offset);
			}
		};
		using MutableView = View<false>;
		using ConstView = View<true>;
		
		MutableView view(int offset=0) {
			return MutableView(*this, offset);
		}
		ConstView view(int offset=0) const {
			return ConstView(*this, offset);
		}
		ConstView constView(int offset=0) const {
			return ConstView(*this, offset);
		}

		Sample & operator[](int offset) {
			return buffer[(bufferIndex + (unsigned)offset)&bufferMask];
		}
		const Sample & operator[](int offset) const {
			return buffer[(bufferIndex + (unsigned)offset)&bufferMask];
		}

		/// Write data into the buffer
		template<typename Data>
		void write(Data &&data, int length) {
			for (int i = 0; i < length; ++i) {
				(*this)[i] = data[i];
			}
		}
		/// Read data out from the buffer
		template<typename Data>
		void read(int length, Data &&data) const {
			for (int i = 0; i < length; ++i) {
				data[i] = (*this)[i];
			}
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

		MutableView operator ++(int) {
			MutableView view(*this);
			++bufferIndex;
			return view;
		}
		MutableView operator +(int i) {
			return MutableView(*this, i);
		}
		ConstView operator +(int i) const {
			return ConstView(*this, i);
		}
		MutableView operator --(int) {
			MutableView view(*this);
			--bufferIndex;
			return view;
		}
		MutableView operator -(int i) {
			return MutableView(*this, -i);
		}
		ConstView operator -(int i) const {
			return ConstView(*this, -i);
		}
};
	
template<typename Sample>
class MultiBuffer {
		int channels, stride;
		Buffer<Sample> buffer;
	public:
		using ConstChannel = typename Buffer<Sample>::ConstView;
		using MutableChannel = typename Buffer<Sample>::MutableView;

		MultiBuffer(int channels=0, int capacity=0) : channels(channels), stride(capacity), buffer(channels*capacity) {}

		void resize(int nChannels, int capacity, Sample value=Sample()) {
			channels = nChannels;
			stride = capacity;
			buffer.resize(channels*capacity, value);
		}
		void reset(Sample value=Sample()) {
			buffer.reset(value);
		}

		/// A reference-like multi-channel result for a particular sample index
		template<bool isConst>
		class Stride {
			using CChannel = typename std::conditional<isConst, ConstChannel, MutableChannel>::type;
			using CSample = typename std::conditional<isConst, const Sample, Sample>::type;
			CChannel view;
			int channels, stride;
		public:
			Stride(CChannel view, int channels, int stride) : view(view), channels(channels), stride(stride) {}
			Stride(const Stride &other) : view(other.view), channels(other.channels), stride(other.stride) {}
			
			CSample & operator[](int channel) {
				return view[channel*stride];
			}
			const Sample & operator[](int channel) const {
				return view[channel*stride];
			}

			/// Reads from the buffer into a multi-channel result
			template<class Data>
			void get(Data &&result) const {
				for (int c = 0; c < channels; ++c) {
					result[c] = view[c*stride];
				}
			}
			/// Writes from multi-channel data into the buffer
			template<class Data>
			void set(Data &&data) {
				for (int c = 0; c < channels; ++c) {
					view[c*stride] = data[c];
				}
			}
			template<class Data>
			Stride & operator =(const Data &data) {
				set(data);
				return *this;
			}
			Stride & operator =(const Stride &data) {
				set(data);
				return *this;
			}
		};
		
		Stride<false> at(int offset) {
			return {buffer.view(offset), channels, stride};
		}
		Stride<true> at(int offset) const {
			return {buffer.view(offset), channels, stride};
		}

		/// Holds a particular position in the buffer
		template<bool isConst>
		class View {
			using CChannel = typename std::conditional<isConst, ConstChannel, MutableChannel>::type;
			CChannel view;
			int channels, stride;
		public:
			View(CChannel view, int channels, int stride) : view(view), channels(channels), stride(stride) {}
			
			CChannel operator[](int channel) {
				return view + channel*stride;
			}
			ConstChannel operator[](int channel) const {
				return view + channel*stride;
			}

			Stride<isConst> at(int offset) {
				return {view + offset, channels, stride};
			}
			Stride<true> at(int offset) const {
				return {view + offset, channels, stride};
			}
		};
		using MutableView = View<false>;
		using ConstView = View<true>;

		MutableView view(int offset=0) {
			return MutableView(buffer.view(offset), channels, stride);
		}
		ConstView view(int offset=0) const {
			return ConstView(buffer.view(offset), channels, stride);
		}
		ConstView constView(int offset=0) const {
			return ConstView(buffer.view(offset), channels, stride);
		}

		MutableChannel operator[](int channel) {
			return buffer + channel*stride;
		}
		ConstChannel operator[](int channel) const {
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
