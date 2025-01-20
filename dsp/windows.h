#include "./common.h"

#ifndef SIGNALSMITH_DSP_WINDOWS_H
#define SIGNALSMITH_DSP_WINDOWS_H

	class Kaiser {
		inline static double bessel0(double x) {
			const double significanceLimit = 1e-4;
			double result = 0;
			double term = 1;
			double m = 0;
			while (term > significanceLimit) {
				result += term;
				++m;
				term *= (x*x)/(4*m*m);
			}

			return result;
		}
		double beta;
		double invB0;
		
		static double heuristicBandwidth(double bandwidth) {
			return bandwidth + 8/((bandwidth + 3)*(bandwidth + 3)) + 0.25*std::max(3 - bandwidth, 0.0);
		}
	public:
		Kaiser(double beta) : beta(beta), invB0(1/bessel0(beta)) {}
		static Kaiser withBandwidth(double bandwidth, bool heuristicOptimal=false) {
			return Kaiser(bandwidthToBeta(bandwidth, heuristicOptimal));
		}

		static double bandwidthToBeta(double bandwidth, bool heuristicOptimal=false) {
			if (heuristicOptimal) {
				bandwidth = heuristicBandwidth(bandwidth);
			}
			bandwidth = std::max(bandwidth, 2.0);
			double alpha = std::sqrt(bandwidth*bandwidth*0.25 - 1);
			return alpha*M_PI;
		}
		
		static double betaToBandwidth(double beta) {
			double alpha = beta*(1.0/M_PI);
			return 2*std::sqrt(alpha*alpha + 1);
		}

		static double bandwidthToEnergyDb(double bandwidth, bool heuristicOptimal=false) {
			if (heuristicOptimal) {
				if (bandwidth < 3) bandwidth += (3 - bandwidth)*0.5;
				return 12.9 + -3/(bandwidth + 0.4) - 13.4*bandwidth + (bandwidth < 3)*-9.6*(bandwidth - 3);
			}
			return 10.5 + 15/(bandwidth + 0.4) - 13.25*bandwidth + (bandwidth < 2)*13*(bandwidth - 2);
		}
		static double energyDbToBandwidth(double energyDb, bool heuristicOptimal=false) {
			double bw = 1;
			while (bw < 20 && bandwidthToEnergyDb(bw, heuristicOptimal) > energyDb) {
				bw *= 2;
			}
			double step = bw/2;
			while (step > 0.0001) {
				if (bandwidthToEnergyDb(bw, heuristicOptimal) > energyDb) {
					bw += step;
				} else {
					bw -= step;
				}
				step *= 0.5;
			}
			return bw;
		}
		static double bandwidthToPeakDb(double bandwidth, bool heuristicOptimal=false) {
			// Horrible heuristic fits
			if (heuristicOptimal) {
				return 14.2 - 20/(bandwidth + 1) - 13*bandwidth + (bandwidth < 3)*-6*(bandwidth - 3) + (bandwidth < 2.25)*5.8*(bandwidth - 2.25);
			}
			return 10 + 8/(bandwidth + 2) - 12.75*bandwidth + (bandwidth < 2)*4*(bandwidth - 2);
		}
		static double peakDbToBandwidth(double peakDb, bool heuristicOptimal=false) {
			double bw = 1;
			while (bw < 20 && bandwidthToPeakDb(bw, heuristicOptimal) > peakDb) {
				bw *= 2;
			}
			double step = bw/2;
			while (step > 0.0001) {
				if (bandwidthToPeakDb(bw, heuristicOptimal) > peakDb) {
					bw += step;
				} else {
					bw -= step;
				}
				step *= 0.5;
			}
			return bw;
		}
		static double bandwidthToEnbw(double bandwidth, bool heuristicOptimal=false) {
			if (heuristicOptimal) 
				bandwidth = heuristicBandwidth(bandwidth);
			double b2 = std::max<double>(bandwidth - 2, 0);
			return 1 + b2*(0.2 + b2*(-0.005 + b2*(-0.000005 + b2*0.0000022)));
		}

		double operator ()(double unit) {
			double r = 2*unit - 1;
			double arg = std::sqrt(1 - r*r);
			return bessel0(beta*arg)*invB0;
		}

		void fill(double *&data, int size) const { //Data = double *&
			double invSize = 1.0/size;
			for (int i = 0; i < size; ++i) {
				double r = (2*i + 1)*invSize - 1;
				double arg = std::sqrt(1 - r*r);
				data[i] = bessel0(beta*arg)*invB0;
			}
		}
	};

	void forcePerfectReconstruction(double *&data, int windowLength, int interval) {
		for (int i = 0; i < interval; ++i) { //Data = double *&
			double sum2 = 0;
			for (int index = i; index < windowLength; index += interval) {
				sum2 += data[index]*data[index];
			}
			double factor = 1/std::sqrt(sum2);
			for (int index = i; index < windowLength; index += interval) {
				data[index] *= factor;
			}
		}
	}
#endif
