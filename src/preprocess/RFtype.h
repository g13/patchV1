#ifndef RFTYPE_H
#define RFTYPE_H
#include <vector>
#include <numeric>
#include "../types.h"
#include "../util/util.h"
#include <cassert>
#include <type_traits>
#include <algorithm>

#define nRFtype 5
#define nOutputType 4
#define nInputType 6

enum class RFtype: PosInt {
	nonOppopent_gabor = 0,  // different cone input correlates, i.e., have the same sign, in gabor form, V1 4Calpha
	nonOppopent_cs = 1,  // different cone input correlates, i.e., have the same sign, in center surround form not necessarily concentric, magno LGN
	singleOppopent = 2, // center surround cone opponency, parvo LGN and V1 4Cbeta
	doubleOppopent_cs = 3, // center surround cone and spatial opponency, V1 4Cbeta
    doubleOppopent_gabor = 4, // cone and spatial opponency in Gabor profile, V1
};

enum class OutputType: PosInt { // V1 local RF subregion
	// if RF has double peaks, choose the first on the left
	// surround ignored when RF have double peaks
	// non-opponent
	LonMon = 0,
	LoffMoff = 1,
	// opponent
	LonMoff = 2,
	LoffMon = 3
};

struct InputActivation {
    Float actPercent[nInputType];
    __host__ __device__
    __forceinline__ InputActivation() {};

    __host__ __device__
    __forceinline__ InputActivation(Float percent[]) {
        for (PosInt i=0; i<nInputType; i++) {
            actPercent[i] = percent[i];
        }
    }

    __host__ __device__
    __forceinline__ void assign(Float percent[]) {
        for (PosInt i=0; i<nInputType; i++) {
            actPercent[i] = percent[i];
        }
    }
};

enum class InputType: PosInt { // LGN
    // center-surround
    LonMoff = 0, // parvocellular
    LoffMon = 1,
    MonLoff = 2,
    MoffLon = 3,
    OnOff = 4, // magnocellular
    OffOn = 5
    // ignore mixed surround, almost always from a different cone type
};

typedef std::underlying_type<RFtype>::type RFtype_t;
typedef std::underlying_type<InputType>::type InputType_t;
typedef std::underlying_type<OutputType>::type OutputType_t;

// normalize x to [-1,1], y to baRatio*[-1,1]
inline std::pair<Float, Float> transform_coord_to_unitRF(Float x, Float y, const Float mx, const Float my, const Float theta, const Float a) {
    // a is half-width at the x-axis
    x = (x - mx)/a;
    y = (y - my)/a;
    Float new_x, new_y;
    new_x = cos(theta) * x + sin(theta) * y;
	new_y = -sin(theta) * x + cos(theta) * y;
    return std::make_pair(new_x, new_y);
}

// TODO: document type meanings
inline Int match_OnOff(InputType iType, OutputType oType, Float &modulation) {
	Int match;
   	switch (oType) {
   	    case OutputType::LonMon:
   			switch (iType) {
                case InputType::LonMoff: case InputType::MonLoff: case InputType::OnOff: match = 1;
   					break;
                case InputType::LoffMon: case InputType::MoffLon: case InputType::OffOn: match = -1;
					break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
   	        break;
   	    case OutputType::LoffMoff:
   	        switch (iType) {
   				case InputType::LoffMon: case InputType::MoffLon: case InputType::OffOn: match = 1;
   					break;                                                               
   				case InputType::LonMoff: case InputType::MonLoff: case InputType::OnOff: match = -1;
					break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
			break;
   	    default: throw("There's no implementation of such non-opponent RF");
	}
   	return match;
}

inline Int oppose_Cone_OnOff_double(InputType iType, OutputType oType, Float &modulation) {
	Int opponent;
   	switch (oType) {
   	    case OutputType::LonMoff:
   			switch (iType) {
   				case InputType::LonMoff: case InputType::MoffLon: opponent = 1;
   					break;
   				case InputType::LoffMon: case InputType::MonLoff: opponent = -1;
					break;
                case InputType::OnOff: case InputType::OffOn: opponent = 0;
                    break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
   	        break;
   	    case OutputType::LoffMon:
   	        switch (iType) {
   				case InputType::LoffMon: case InputType::MonLoff: opponent = 1;
   					break;
   				case InputType::LonMoff: case InputType::MoffLon: opponent = -1;
					break;
                case InputType::OnOff: case InputType::OffOn: opponent = 0;
                    break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
			break;
   	    default: throw("There's no implementation of such opponent RF");
	}
   	return opponent;
}

inline Int oppose_Cone_OnOff_single(InputType iType, OutputType oType, Float &modulation) {
	Int opponent;
   	switch (oType) {
   	    case OutputType::LonMoff:
   			switch (iType) {
   				case InputType::LonMoff: opponent = 1;
   					break;
                case InputType::LoffMon: case InputType::MoffLon: case InputType::MonLoff: case InputType::OffOn: opponent = -1;
					break;
                case InputType::OnOff: opponent = 0;
                    break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
   	        break;
   	    case OutputType::LoffMon:
   	        switch (iType) {
   				case InputType::LoffMon: opponent = 1;
   					break;
                case InputType::LonMoff: case InputType::MonLoff: case InputType::MoffLon: case InputType::OnOff: opponent = -1;
					break;
                case InputType::OffOn: opponent = 0;
                    break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
			break;
   	    default: throw("There's no implementation of such opponent RF");
	}
   	return opponent;
}

struct LinearReceptiveField { // RF sample without implementation of check_opponency
    RFtype rfType; // class identifier for pointer
	std::vector<Float> prob;
    Size n;
    //                        a is the minor-axis
    Float sfreq, phase, amp, theta, a, baRatio, sig;
    OutputType oType;
	bool strictStrength;
	/* not needed
    // default constructor
    LinearReceptiveField(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, Float sig = 1.177):
        n(n),
        sfreq(sfreq),
        phase(phase),
        amp(amp),
        theta(theta),
        a(a),
        baRatio(baRatio),
        oType(oType),
        sig(sig)
    {} */
    virtual void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, bool strictStrength, Float sig = 1.177) {
		assert(amp <= 1);
        this->n = n;
        this->sfreq = sfreq;
        this->phase = phase;
        this->amp = amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
		this->strictStrength = strictStrength;
	}

	virtual void clear() {
		prob.clear();
	}
 
    virtual Size construct_connection_N(std::vector<Float> &x, std::vector<Float> &y, std::vector<InputType> &iType, std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen, Float fnLGNeff, bool p_n, Size max_nCon, bool top_pick, Float thres = 0.5) {
        Size nConnected;
        if (n > 0) {
            if (fnLGNeff > 0) {
		        prob.reserve(n);
                // putative RF center
			    Float cx, cy;
                std::tie(cx, cy) = average2D<Float>(x, y);
		        for (Size i = 0; i < n; i++) {
                    Float norm_x, norm_y;
                    // orient and normalize LGN coord
		            std::tie(norm_x, norm_y) = transform_coord_to_unitRF(x[i], y[i], cx, cy, theta, a);
                    // calc. prob. dist. distance-dependent envelope at coord.
                    Float envelope = get_envelope(norm_x, norm_y, top_pick, thres, sig);
                    // calc. modulation of prob at coord.
                    Float modulation = modulate(norm_x, norm_y);
                    /* TEST with no modulation: bool RefShift = false;
                    if (modulation < 0.5) {
                        RefShift = true;
                    } */
                    // calc. cone and position opponency at coord.
	                Float opponent = check_opponency(iType[i], modulation);
					Float _prob = get_prob(opponent, modulation, envelope);
					assert(_prob >= 0); 
                    prob.push_back(_prob);
                    /* TEST with no modulation: if (opponent < 0.0 && RefShift || (opponent < 0.0 && (rfType == RFtype::doubleOppopent_cs || rfType == RFtype::singleOppopent || rfType == RFtype::nonOppopent_cs))) {
                        if (prob.back() > 0.0) {
                            std::cout << envelope << " * (0.5 + " << amp << " * " << opponent << " * " << modulation <<  ") = " << prob.back() << "\n";
                            assert(prob.back() == 0.0);
                        }
                    } */
                }
				// set the number of connections
                Size sSum = normalize(fnLGNeff, p_n);
                // make connection and update ID and strength list
                nConnected = connect_N(idList, strengthList, rGen, max_nCon, sSum, top_pick);
            } else {
                idList = std::vector<Size>();
                idList.shrink_to_fit();
                nConnected = 0;
            }
        }  else {
            nConnected = 0;
        }
        return nConnected;
    }
    // Probability envelope based on distance
    //virtual Float get_envelope(Float x, Float y, bool top_pick, Float thres = 0.5, Float sig = 1.1775) {
    virtual Float get_envelope(Float x, Float y, bool top_pick, Float thres = 0.5, Float sig = 0.6) {
        Float v = exp(-0.5*(pow(x/sig,2)+pow(y/(sig*baRatio),2)));
		//if (top_pick) {
		//	if (v > thres) {
		//		v = 1.0;
		//	}
		//}
        return v;
        //return 1.0;
        // baRatio comes from Dow et al., 1981
    }
    // Full cosine modulation, modulation is on cone and on-off types
    virtual Float modulate(Float x, Float y) {
        return 0.5 + 0.5 * cos(sfreq*(x + phase)*M_PI);
		//assert(abs(x) <= 1.0);
		//assert(sfreq*x * M_PI + phase > 0 && sfreq*x * M_PI + phase < 2*M_PI);
        // sfreq should be given as a percentage of the width
        // when sfreq == 1, the RF contain a full cycle of cos(x), x\in[-pi, pi]
    }
    // To be implemented by derived class
    virtual Float check_opponency(InputType iType, Float &modulation) = 0;
    // produce connection prob. at coord. for i-th LGN
    virtual Float get_prob(Float opponent, Float modulation, Float envelope) {
        return envelope * (1.0 + amp * opponent * modulation);
    }
    // normalize prob.
    Size normalize(Float fnLGNeff, bool p_n) {
	    // average connection probability is controlled at fnLGNeff.
		Float norm;
		Size sSum;
        if (p_n) { // if percentage
			sSum = static_cast<Size>(round(fnLGNeff*prob.size()));
        } else { // number restriction
			sSum = static_cast<Size>(round(fnLGNeff));
        }
		if (sSum > 0) {
    		norm = std::accumulate(prob.begin(), prob.end(), 0.0) / sSum;
	    	//std::cout << "norm = " << norm << "\n";
	    	//print_list<Float>(prob);
        	//Float sum = 0.0;
        	if (norm == 0) {
				sSum = 0;
			} else {
				Size nAvail = 0;
        		for (PosInt i=0; i<prob.size(); i++) {
					if (prob[i] > 0) {
						nAvail++;
					}
        		    prob[i] = prob[i] / norm;
        		}
				sSum = nAvail < sSum? nAvail: sSum;
			}
		}
    	//std::cout << fnLGNeff*prob.size() << " ~ " << sum << "\n";
		return sSum;
    }
    // make connections
    virtual Size connect_N(std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen, Size max_nCon, Size sSum, bool top_pick) {
		// make connections and normalized strength i.e., if prob > 1 then s = 1 else s = prob
        std::uniform_real_distribution<Float> uniform(0,1);
        std::normal_distribution<Float> normal(0,0.05);
		strengthList.reserve(n);
		std::vector<Size> newList;
		newList.reserve(n);
		if (sSum > 0) {
			Size count = 0; 
			Size n_to_connect = sSum;
			assert(n_to_connect <= max_nCon);
			if (!top_pick) {
				do  { // variable connection
					newList.clear();
					strengthList.clear();
					for (PosInt i = 0; i < n; i++) {
						if (uniform(rGen) < prob[i]) {
							newList.push_back(idList[i]);
							if (prob[i] > 1) {
								strengthList.push_back(prob[i]);
							} else {
								strengthList.push_back(1);
							}
						}
					}
					count++;
					if (count > 20) {
						std::cout << "too many iters, sum over prob #" << prob.size() << " = " << std::accumulate(prob.begin(), prob.end(), 0.0) << "\n";
						assert(count <= 20);
					}
				} while (newList.size() == 0 || newList.size() > max_nCon);
			} else { // fixed number
				/*
				do  { // decide number to connect first
					n_to_connect = 0;
					for (PosInt i = 0; i < n; i++) {
						prob[i] *= 1+normal(rGen);
						if (uniform(rGen) < prob[i]) {
							n_to_connect++;
						}
					}
					count++;
					if (count > 20) {
						std::cout << "too many iters, sum over prob #" << prob.size() << " = " << std::accumulate(prob.begin(), prob.end(), 0.0) << "\n";
						assert(count <= 20);
					}
				} while (n_to_connect > max_nCon || n_to_connect == 0);
				*/
				// pick the tops
				do  {
					// get the LGN with max prob
					PosInt i = std::distance(prob.begin(), std::max_element(prob.begin(), prob.end()));
					newList.push_back(idList[i]);
					// put the max prob to min
					prob[i] = -prob[i];
					if (prob[i] > 1) {
						strengthList.push_back(prob[i]);
					} else {
						strengthList.push_back(1);
					}
				} while (newList.size() < n_to_connect || newList.size() > max_nCon);
			}

			if (strictStrength) {
				Float con_irl = std::accumulate(strengthList.begin(), strengthList.end(), 0.0);
        		Float ratio = sSum/con_irl;
        		for (PosInt i=0; i<strengthList.size(); i++) {
        		    strengthList[i] *= ratio; 
        		}
			}
		}
		idList = newList;
		idList.shrink_to_fit();
        return static_cast<Size>(idList.size());
    }

    virtual Size connect_thres(std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen, Float conThres, Size max_nCon, std::vector<Int> &iPick, std::vector<Float> &norm_x, std::vector<Float> &norm_y, Size n, PosInt iV1) {
		// make connections and normalized strength i.e., if prob > 1 then s = 1 else s = prob

		std::vector<Size> list0;
		list0.reserve(n);
		std::vector<Float> strList0;
		strList0.reserve(n);

		std::vector<Float> x;
		std::vector<Float> y;
		std::vector<Int> pick2;
		Int count = 0;
		Float probSum = 0.0;
		for (PosInt i = 0; i < n; i++) {
			if (prob[i] > conThres) {
				list0.push_back(idList[i]);
				//if (prob[i] > 1) {
					strList0.push_back(prob[i]);
				//} else {
				//	strList0.push_back(1);
				//}
				count++;
				pick2.push_back(iPick[i]);
				x.push_back(norm_x[i]);
				y.push_back(norm_y[i]);
				probSum += prob[i];
			}
		}
		Size m = list0.size();

		// pickout extra LGN cons with equal prob.
		std::vector<bool> ipick(m, true);
        std::uniform_real_distribution<Float> uniform(0,1);
		if (count > max_nCon) {
			while (count-max_nCon > 0) {
				PosInt ipicked = static_cast<int>(flooring(uniform(rGen) * m));
				if (ipick[ipicked]) {
					ipick[ipicked] = false;
					count--;
				}
			}
			strengthList.reserve(max_nCon);
			idList.clear();
			idList.reserve(max_nCon);
		} else {
			strengthList.reserve(count);
			idList.clear();
			idList.reserve(count);
		}
		// push to output vectors
		std::vector<Float> cx(2, 0), cy(2, 0);
		Size n0 = 0;
		Size n1 = 0;
		for (PosInt i=0; i<m; i++) {
			if (ipick[i]) {
				idList.push_back(list0[i]);
				strengthList.push_back(strList0[i]);
				if (pick2[i] > 0) {
					cx[1] += x[i];
					cy[1] += y[i];
					n1++;

				} else {
					cx[0] += x[i];
					cy[0] += y[i];
					n0++;
				}
			}
		}
		norm_x.clear();
		norm_y.clear();
		norm_x = x;
		norm_y = y;
		norm_x.shrink_to_fit();
		norm_y.shrink_to_fit();
		Size nConnected = idList.size();
		assert(n0 + n1 == nConnected);
		if (n0 > 0 && n1 > 0) {
			cx[0] /= n0;
			cy[0] /= n0;
			cx[1] /= n1;
			cy[1] /= n1;
			Float dis = square_root((cx[0] - cx[1]) * (cx[0] - cx[1]) + (cy[0] - cy[1]) * (cy[0] - cy[1]));
			if (dis >= 2*square_root(2)) {
				std::cout << "dis = " << dis  << "\n";
				std::cout << " c0 = [" << cx[0] << ", " << cy[0] << "]\n";
				std::cout << " c1 = [" << cx[1] << ", " << cy[1] << "]\n";
				//assert(dis < 2);
			}
			sfreq = 1/(dis*2/2*2*a);
		} else {
			sfreq = 0;
		}
		//std::cout << "sfreq = " << sfreq << " from " << nConnected << " LGN cells\n";

		assert(nConnected <= max_nCon);

        return nConnected;
    }

    virtual Size construct_connection_thres(std::vector<Float> &x, std::vector<Float> &y, std::vector<InputType> &iType, std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen, Size max_nCon, Float conThres, Float zero, Float &true_sfreq, Float vx, Float vy, PosInt iV1) {
        Size nConnected;
        if (n > 0) {
            if (zero > 0) {
        		std::uniform_real_distribution<Float> uniform(0,1);
		        prob.reserve(n);
                // putative RF center
			    Float cx, cy;
                std::tie(cx, cy) = average2D<Float>(x, y);
				//cx = vx;
				//cy = vy;
				std::vector<Float> norm_x;
				std::vector<Float> norm_y; 
				std::vector<Int> iPick;  // 0 Lon,Moff,On, 1 Loff,Mon,Off
				std::vector<Float> envelope_value; 
				std::vector<Size> newList; 
				std::vector<InputType> newType; 

				Int iOnOff;
				switch (oType) {
					case OutputType::LonMoff: case OutputType::LonMon: iOnOff = 1;
   						break;
					case OutputType::LoffMon: case OutputType::LoffMoff: iOnOff = -1;
   						break;
				}
		        for (Size i=0; i<n; i++) {
					Float temp_x, temp_y;
		        	std::tie(temp_x, temp_y) = transform_coord_to_unitRF(x[i], y[i], cx, cy, theta, a);

					Float temp_value = get_envelope(temp_x, temp_y, false);
                    if (temp_value > 0.0) {
						norm_x.push_back(temp_x);
						norm_y.push_back(temp_y);
						envelope_value.push_back(temp_value);
						//envelope_value.push_back(1.0);
   						switch (iType[i]) {
							case InputType::LonMoff: case InputType::MoffLon: case InputType::OnOff: iPick.push_back(1);
   								break;
   							case InputType::LoffMon: case InputType::MonLoff: case InputType::OffOn: iPick.push_back(-1);
								break;
   	        			}
						newList.push_back(idList[i]);
						newType.push_back(iType[i]);
					}
				}
				// get a LGN cell's position as the phase, randomly chose based on the con prob from their amplitude modulation over VF distance
				Float rand = uniform(rGen);
				PosInt m = iPick.size();
				PosInt iPhase = static_cast<PosInt>(flooring(rand*m));

				phase = norm_x[iPhase] + iPick[iPhase]*iOnOff/2;
				//phase = M_PI/2.0;
                
				Float max_prob[2] = {0, 0};
		        for (Size i=0; i<m; i++) {
                    // calc. modulation of prob at coord.
                    Float modulation = modulate(norm_x[i], norm_y[i]); // sfreq, phase
                    // calc. cone and position opponency at coord.
	                Float opponent = check_opponency(newType[i], modulation);
					Float _prob = get_prob(opponent, modulation, envelope_value[i]);
					assert(_prob >= 0);
                    prob.push_back(_prob);
					if (iPick[i] == 1) {
						if (_prob > max_prob[0]) max_prob[0] = _prob;
					} else {
						if (_prob > max_prob[1]) max_prob[1] = _prob;
					}
                }
		        for (Size i=0; i<m; i++) {
					if (iPick[i] == 1) {
						prob[i] /= max_prob[0];
					} else {
						prob[i] /= max_prob[1];
					}
				}
                // make connection and update ID and strength list
                nConnected = connect_thres(newList, strengthList, rGen, conThres, max_nCon, iPick, norm_x, norm_y, m, iV1);
				idList = newList; 
				idList.shrink_to_fit();
				true_sfreq = sfreq;
            } else {
                idList = std::vector<Size>();
                idList.shrink_to_fit();
                nConnected = 0;
				true_sfreq = 0;
            }
        } else {
            nConnected = 0;
			true_sfreq = 0;
        }
        return nConnected;
    }
    virtual Size construct_connection_opt(std::vector<Float> &x, std::vector<Float> &y, std::vector<InputType> &iType, std::vector<Size> &idList, std::vector<Float> &strengthList, Float zero, Float &true_sfreq, Float vx, Float vy, PosInt iV1, Float ori_tol, Float disLGN, Float sSum) {
        Size nConnected;
        if (n > 0) {
            if (zero > 0) {
                // putative RF center
			    Float cx, cy;
                std::tie(cx, cy) = average2D<Float>(x, y);
				//cx = vx;
				//cy = vy;
				std::vector<Float> norm_x;
				std::vector<Float> norm_y; 
				std::vector<Int> biPick;  // 0 Lon,Moff,On, 1 Loff,Mon,Off
				std::vector<Float> envelope_value; 

				Int iOnOff;
				switch (oType) {
					case OutputType::LonMoff: case OutputType::LonMon: iOnOff = 1;
   						break;
					case OutputType::LoffMon: case OutputType::LoffMoff: iOnOff = -1;
   						break;
				}
		        for (Size i=0; i<n; i++) {
					Float temp_x, temp_y;
		        	std::tie(temp_x, temp_y) = transform_coord_to_unitRF(x[i], y[i], cx, cy, theta, a);
					envelope_value.push_back(get_envelope(temp_x, temp_y, false));
					norm_x.push_back(temp_x);
					norm_y.push_back(temp_y);
   					switch (iType[i]) {
						case InputType::LonMoff: case InputType::MoffLon: case InputType::OnOff: biPick.push_back(1);
   							break;
   						case InputType::LoffMon: case InputType::MonLoff: case InputType::OffOn: biPick.push_back(-1);
							break;
   	        		}
				}

                // make connection and update ID and strength list
                nConnected = connect_opt(idList, strengthList, iType, biPick, envelope_value, norm_x, norm_y, n, iOnOff, iV1, ori_tol, disLGN, sSum);
				idList.shrink_to_fit();
				true_sfreq = sfreq;
            } else {
                idList = std::vector<Size>();
                idList.shrink_to_fit();
                nConnected = 0;
				true_sfreq = 0;
            }
        }  else {
            nConnected = 0;
			true_sfreq = 0;
        }
        return nConnected;
    }

    virtual Size connect_opt(std::vector<Size> &idList, std::vector<Float> &strengthList, std::vector<InputType> &iType, std::vector<Int> &biPick, std::vector<Float> envelope_value, std::vector<Float> &norm_x, std::vector<Float> &norm_y, Size n, Int iOnOff, PosInt iV1, Float ori_tol, Float disLGN, Float sSum, Float dmax = 2) {
		// make connections and normalized strength i.e., if prob > 1 then s = 1 else s = prob
        ori_tol = ori_tol/180*M_PI;

		std::vector<Float> xon;
		std::vector<Float> yon;
		std::vector<PosInt> ion;
		std::vector<Float> xoff;
		std::vector<Float> yoff;
		std::vector<PosInt> ioff;
        std::vector<std::vector<PosInt>> onComponent;
        std::vector<std::vector<PosInt>> offComponent;
        bool pInfo = false;
        //if (iV1 == 3*1024 + 744) {
        //if (iV1 == 49*1024 + 946 || iV1 == 4*1024 + 675) {
        if (iV1 == 1183) {
        //if (iV1 ==  5460 || iV1 == 3816) {
            pInfo = true;
            //pInfo = false;
        }
        
		for (PosInt i = 0; i < n; i++) {
            if (biPick[i] > 0) {
                ion.push_back(i);
                xon.push_back(norm_x[i]);
                yon.push_back(norm_y[i]);
            } else {
                ioff.push_back(i);
                xoff.push_back(norm_x[i]);
                yoff.push_back(norm_y[i]);
            }
        }
        int anyLGN = 0;
        Float min_tan = tangent(ori_tol);

        // try multiple on
        std::vector<Float> phaseOn; 
        std::vector<Float> disOnY; 
        std::vector<Float> meanOnY; 
        std::vector<Size> nExtraOn; 
        Size non = ion.size();
        if (pInfo) {
            std::cout << iV1 << " have " << non << " red LGNs\n";
		    for (PosInt i = 0; i < non; i++) {
                printf("%i: (%.4f, %.4f)\n", i, xon[i], yon[i]);
            }
        }
		for (PosInt i = 0; i < non; i++) {
            for (PosInt j = i+1; j < non; j++) {
                if (abs((xon[i]-xon[j])/(yon[i]-yon[j])) < min_tan && abs(yon[i] - yon[j]) < disLGN) {
                    bool newComp = true;
                    if (pInfo) {
                        printf("picked: %i, %i (%.4f, %.4f)", i, j, xon[i]-xon[j], yon[i]-yon[j]);
                    }
                    for (PosInt ic=0; ic<onComponent.size(); ic++) {
                        bool new_j = true;
                        for (PosInt k=0; k<onComponent[ic].size(); k++) {
                            if (onComponent[ic][k] == i)  {
                                newComp = false;
                            }
                            if (onComponent[ic][k] == j)  {
                                newComp = false;
                                new_j = false;
                            }
                        }
                        if (!newComp) {
                            disOnY[ic] += abs(yon[i] - yon[j]);
                            if (new_j) {
                                onComponent[ic].push_back(j);
                                phaseOn[ic] += xon[j];
                                meanOnY[ic] += yon[j];
                            } else {
                                nExtraOn[ic]++;
                            }
                        }
                        if (pInfo) {
                            if (!newComp) {
                                printf(" for component %i\n", ic);
                            }
                        }
                        if (!newComp) {
                            break;
                        }
                    }
                    if (newComp) {
                        onComponent.push_back(std::vector<PosInt>());
                        onComponent.back().push_back(i);
                        onComponent.back().push_back(j);
                        phaseOn.push_back(xon[i] + xon[j]);
                        meanOnY.push_back(yon[i] + yon[j]);
                        disOnY.push_back(abs(yon[i] - yon[j]));
                        nExtraOn.push_back(0);
                        if (pInfo) {
                            printf(" for component %i\n", onComponent.size()-1);
                        }
                    }
                }
            }
        }
        
        Size m = onComponent.size();
        if (pInfo) {
            std::cout << "  found " << m << " on components\n";
        }
        if (m > 0) {
            anyLGN += 1;
        }
        Float minPhaseVar = 1;
        PosInt ionC;
        for (PosInt i=0; i<m; i++) {
            Size nC = onComponent[i].size();
            phaseOn[i] /= nC;
            meanOnY[i] /= nC;
            disOnY[i] /= nC-1 + nExtraOn[i];
            Float var_x = 0;
            for (PosInt j=0; j<nC; j++) {
                var_x += power(xon[onComponent[i][j]] - phaseOn[i],2);
            }
            var_x /= onComponent[i].size();
            if (var_x < minPhaseVar) {
                minPhaseVar = var_x;
                ionC = i;
            }
        }
        if (m > 0) {
            if (pInfo) {
                printf("on component %i is chosen\n", ionC);
            }
        }

        // try multiple on
        std::vector<Float> phaseOff; 
        std::vector<Float> meanOffY; 
        std::vector<Float> disOffY; 
        std::vector<Size> nExtraOff; 
        Size noff = ioff.size();
        if (pInfo) {
            std::cout << iV1 << " have " << noff << " green LGNs\n";
		    for (PosInt i = 0; i < noff; i++) {
                printf("%i: (%.4f, %.4f)\n", i, xoff[i], yoff[i]);
            }
        }
		for (PosInt i = 0; i < noff; i++) {
            for (PosInt j = i+1; j < noff; j++) {
                if (abs((xoff[i]-xoff[j])/(yoff[i]-yoff[j])) < min_tan && abs(yoff[i] - yoff[j]) < disLGN) {
                    bool newComp = true;
                    if (pInfo) {
                        printf("picked: %i, %i (%.4f, %.4f) ", i, j, xoff[i]-xoff[j], yoff[i]-yoff[j]);
                    }
                    for (PosInt ic=0; ic<offComponent.size(); ic++) {
                        bool new_j = true;
                        for (PosInt k=0; k<offComponent[ic].size(); k++) {
                            if (offComponent[ic][k] == i)  {
                                newComp = false;
                            }
                            if (offComponent[ic][k] == j)  {
                                newComp = false;
                                new_j = false;
                            }
                        }
                        if (!newComp) {
                            disOffY[ic] += abs(yoff[i] - yoff[j]);
                            if (new_j) {
                                offComponent[ic].push_back(j);
                                phaseOff[ic] += xoff[j];
                                meanOffY[ic] += yoff[j];
                            } else {
                                nExtraOff[ic]++;
                            }
                        }
                        if (pInfo) {
                            if (!newComp) {
                                printf(" for component %i\n", ic);
                            }
                        }
                        if (!newComp) {
                            break;
                        }
                    }
                    if (newComp) {
                        offComponent.push_back(std::vector<PosInt>());
                        offComponent.back().push_back(i);
                        offComponent.back().push_back(j);
                        phaseOff.push_back(xoff[i] + xoff[j]);
                        meanOffY.push_back(yoff[i] + yoff[j]);
                        disOffY.push_back(abs(yoff[i] - yoff[j]));
                        nExtraOff.push_back(0);
                        if (pInfo) {
                            printf(" for component %i\n", offComponent.size()-1);
                        }
                    }
                }
            }
        }
        
        m = offComponent.size();
        if (pInfo) {
            std::cout << "  found " << m << " off components\n";
        }
        if (m > 0) {
            anyLGN += 2;
        }
        minPhaseVar = 1;
        PosInt ioffC;
        for (PosInt i=0; i<m; i++) {
            Size nC = offComponent[i].size();
            phaseOff[i] /= nC;
            meanOffY[i] /= nC;
            disOffY[i] /= nC-1 + nExtraOff[i];
            Float var_x = 0;
            for (PosInt j=0; j<offComponent[i].size(); j++) {
                var_x += power(xoff[offComponent[i][j]] - phaseOff[i],2);
            }
            var_x /= offComponent[i].size();
            if (var_x < minPhaseVar) {
                minPhaseVar = var_x;
                ioffC = i;
            }
        }
        if (m > 0) {
            if (pInfo) {
                printf(" off component %i is chosen\n", ioffC);
            }
        }

		bool try_again = true;
		while (try_again) {
			switch (anyLGN) {
        	    case 0: {// two LGN 
					try_again = false;
        	        if (pInfo) {
        	            std::cout << "try pairs...";
        	        }
        	        PosInt kon, koff;
        	        for (PosInt i = 0; i < non; i++) {
        	            for (PosInt j = 0; j < noff; j++) {
                            if ((xon[i]-xoff[j]) >= disLGN && (xon[i]-xoff[j]) < dmax*disLGN) {
        	                    Float tan0 = abs((yon[i]-yoff[j])/(xon[i]-xoff[j]));
        	                    if (tan0 < min_tan) {
        	                        min_tan = tan0;
        	                        kon = i;
        	                        koff = j;
        	                        anyLGN = 4;
        	                    }
                            }
        	            }
        	        }
        	        if (anyLGN) {
        	            onComponent.push_back(std::vector<PosInt>());
                        ionC = onComponent.size()-1;
        	            onComponent[ionC].push_back(kon);
        	            phaseOn.push_back(xon[kon]);
                        if (ionC != phaseOn.size() - 1) {
                            printf("ionC = %i, size of phaseOn: %i, size of onComponent%i\n", ionC, phaseOn.size(), onComponent.size());
                            assert(ionC == phaseOn.size() - 1);
                        }

        	            offComponent.push_back(std::vector<PosInt>());
                        ioffC = offComponent.size()-1;
        	            offComponent[ioffC].push_back(koff);
        	            phaseOff.push_back(xoff[koff]);
        	            if (pInfo) {
        	                std::cout << "succeeded with pair (" << kon << ", " << koff << ")\n";
        	            }
        	        } else {
        	            if (pInfo) {
        	                std::cout << "failed\n";
        	            }
        	        }
        	    }
        	    break;
        	    case 1: {// on components only, get one off LGN
        	        int imin = -1;
        	        Float rangeOnY[2] = {100, 0};
        	        for (PosInt i=0; i<onComponent[ionC].size(); i++) {
        	            Float temp_y = yon[onComponent[ionC][i]];
        	            if (rangeOnY[1] < temp_y) {
        	                rangeOnY[1] = temp_y;
        	            }
        	            if (rangeOnY[0] > temp_y) {
        	                rangeOnY[0] = temp_y;
        	            }
        	        }
        	        Float centerDis = (rangeOnY[1] - rangeOnY[0]) * (1 + min_tan)/2;
        	        if (pInfo) {
        	            std::cout << "centerDis = " << centerDis << "\n";
        	        }
        	        for (PosInt i=0; i<noff; i++) {
                        Float dx = abs(xoff[i] - phaseOn[ionC]);
        	            if (dx >= disLGN && dx < dmax*disLGN && dx > disOnY[ionC]) {
        	                Float cD = abs(2*yoff[i] - (rangeOnY[0] + rangeOnY[1]));
        	                if (pInfo) {
        	                    std::cout << i << "th off LGN cD = " << cD << "\n";
        	                }
        	                if (cD < centerDis) {
        	                    centerDis = cD;
        	                    imin = i;
        	                } 
        	            } else {
        	                if (pInfo) {
        	                    std::cout << i << "th off LGN dropped for disLGN\n";
        	                }
        	            }
        	        }
        	        if (imin >= 0) {
        	            offComponent.push_back(std::vector<PosInt>()) ;
        	            offComponent.back().push_back(imin);
        	            phaseOff.push_back(xoff[imin]);
        	            ioffC = offComponent.size()-1;
        	            if (pInfo) {
        	                std::cout << " chose " << imin << " th off LGN\n";
        	            }
					    try_again = false;
        	        } else {
        	            if (pInfo) {
        	                std::cout << " all other off LGNs dropped for range, try two LGN next before falling back on using only the " << ionC << "th on component.\n";
        	            }
                        anyLGN = 0;
        	        }
        	    }
        	    break;
        	    case 2: {// off components only, get one on LGN
        	        if (pInfo) {
        	            std::cout << "try find one on LGN\n";
        	        }
        	        int imin = -1;
        	        Float rangeOffY[2] = {100,0};
        	        for (PosInt i=0; i<offComponent[ioffC].size(); i++) {
        	            Float temp_y = yoff[offComponent[ioffC][i]];
        	            if (rangeOffY[1] < temp_y) {
        	                rangeOffY[1] = temp_y;
        	            }
        	            if (rangeOffY[0] > temp_y) {
        	                rangeOffY[0] = temp_y;
        	            }
        	        }
        	        Float centerDis = (rangeOffY[1] - rangeOffY[0]) * (1 + min_tan)/2;
        	        if (pInfo) {
        	            std::cout << "centerDis = " << centerDis << "\n";
        	        }
        	        for (PosInt i=0; i<non; i++) {
                        Float dx = abs(xon[i] - phaseOff[ioffC]);
        	            if (dx >= disLGN && dx < dmax*disLGN && dx > disOffY[ioffC]) {
        	                Float cD = abs(2*yon[i] - (rangeOffY[0] + rangeOffY[1]));
        	                if (pInfo) {
        	                    std::cout << i << "th on LGN cD = " << cD << "\n";
        	                }
        	                if (cD < centerDis) {
        	                    centerDis = cD;
        	                    imin = i;
        	                }
        	            } else {
        	                if (pInfo) {
        	                    std::cout << i << "th on LGN dropped for disLGN\n";
        	                }
        	            }
        	        }
        	        if (imin >= 0) {
        	            onComponent.push_back(std::vector<PosInt>()) ;
        	            onComponent.back().push_back(imin);
        	            phaseOn.push_back(xon[imin]);
        	            ionC = onComponent.size()-1;
        	            if (pInfo) {
        	                std::cout << " chose " << imin << " th on LGN\n";
        	            }
                        try_again = false;
        	        } else {
        	            if (pInfo) {
        	                std::cout << " all other on LGNs dropped for range, try two LGN next before falling back on using only the" << ioffC << "th off component.\n";
        	            }
                        anyLGN = 0;
        	        }
        	    }
        	    break;
        	    case 3: {// on and off components
        	        Float minDisY = disLGN;
        	        if (pInfo) {
        	            std::cout << " pick pairs of on and off components\n";
        	        }
					bool nopair = true;
        	        for (PosInt i=0; i<onComponent.size(); i++) {
        	            for (PosInt j=0; j<offComponent.size(); j++) {
                            Float dx = abs(phaseOn[i] - phaseOff[j]);
        	                Float dydx_ratio = std::max(disOnY[i], disOffY[j])/dx;
        	                if (dx >= disLGN && dx < dmax*disLGN && dydx_ratio < 1) {
                                Float disY = abs(meanOnY[i] - meanOffY[j]);
        	                    if (disY < minDisY) {
                                    minDisY = disY;
        	                        ionC = i;
        	                        ioffC = j;
									nopair = false;
        	                    }
        	                } else {
        	                    if (pInfo) {
        	                        std::cout << "pair: " << i << "th on and " << j << "th off is droppped, dy:dx = " << dydx_ratio << ", dx = " << dx << "\n";
        	                    }
        	                }
        	            }
        	        }
					if (nopair) {
						if (iOnOff > 0) {
							anyLGN = 1;
                            offComponent.clear();
                            phaseOff.clear();
        	                if (pInfo) {
        	                    std::cout << "try again with only " << ionC << "th on component.\n";
        	                }
						} else {
							anyLGN = 2;
                            onComponent.clear();
                            phaseOn.clear();
        	                if (pInfo) {
        	                    std::cout << "try again with only " << ioffC << "th off component.\n";
        	                }
						}
					} else {
						try_again = false;
        	            if (pInfo) {
        	                std::cout << "pair: " << ionC << "th on and " << ioffC << "th off is picked\n";
        	            }
					}
        	    }
        	    break;
        	}
		}

        std::vector<PosInt> added;
        if (anyLGN == 0) { // one LGN
            PosInt j;
            Float max_env = 0.5;
            bool zero = true;
            for (PosInt i = 0; i < n; i++) {
                // find largest matching LGN
                if (envelope_value[i] > max_env) {
                    j = i; 
                    max_env = envelope_value[i];
                    zero = false;
                }
            }
            if (!zero) {
                if (iOnOff*biPick[j] < 0) {
                    if (biPick[j] > 0) {
                        oType = OutputType::LonMoff;
                    } else {
                        oType = OutputType::LoffMon;
                    }
                }
                phase = -norm_x[j];
                sfreq = 1/(2*disLGN);
                added.push_back(j);
                anyLGN = 5;
                j = idList[j];
                idList.clear();
			    idList.push_back(j);
            } else {
                sfreq = 0;
                idList.clear();
            }
        } else {
            std::vector<PosInt> newList;

            if (onComponent.size() > 0) {
                Size m = onComponent[ionC].size();
                for (PosInt i = 0; i<m; i++) {
                    PosInt id = ion[onComponent[ionC][i]];
                    added.push_back(id);
                    newList.push_back(idList[id]);
                }
            }
            if (offComponent.size() > 0) {
                m = offComponent[ioffC].size();
                for (PosInt i = 0; i<m; i++) {
                    PosInt id = ioff[offComponent[ioffC][i]];
                    added.push_back(id);
                    newList.push_back(idList[id]);
                }
            }

            idList.assign(newList.begin(), newList.end());

            if (offComponent.size() > 0 && onComponent.size() > 0) {
                sfreq = 1/abs(phaseOn[ionC] - phaseOff[ioffC]);
                if (sfreq > 1/disLGN) {
                    std::cout << sfreq << " = 1/(" << phaseOn[ionC] << " - " << phaseOff[ioffC] << ")\n";
                    assert(sfreq <= 1/disLGN);
                }
                assert(sfreq > 1/std::max(dmax,2.0f)/disLGN);
                if (iOnOff > 0) {
                    phase = -phaseOn[ionC];
                } else {
                    phase = -phaseOff[ioffC];
                }
            } else {
                sfreq = 1/(2*disLGN);
                if (onComponent.size() > 0) {
                    oType = OutputType::LonMoff;
                    phase = -phaseOn[ionC];
                } else {
                    oType = OutputType::LoffMon;
                    phase = -phaseOff[ioffC];
                }
            }
        }
        if (pInfo) {
            std::cout << iV1 << " try connect " << idList.size() << ", sfreq = " << sfreq << ", phase = " << phase << "\n";
            if (onComponent.size() > 0) {
                std::cout << phaseOn[ionC] << "\n";
            }
            if (offComponent.size() > 0) {
                std::cout << phaseOff[ioffC] << "\n";
            }
        }

		// assign strengths.
        Size nErase = 0;
        PosInt j = 0;
        m = idList.size();
		for (Size i=0; i<m; i++) {
            // calc. modulation of prob at coord.
            Float modulation = modulate(norm_x[added[i]], norm_y[added[i]]); // sfreq, phase
            // calc. cone and position opponency at coord.
	        Float opponent = check_opponency(iType[added[i]], modulation);
			Float _prob = get_prob(opponent, modulation, envelope_value[added[i]]);
            if (_prob > 0) {
                strengthList.push_back(_prob);
                j++;
				if (pInfo) {
					printf("accepted (%.4f, %.4f) with modulation of %.2f and opponency: %.1f, envelope = %.3f\n", norm_x[added[i]], norm_y[added[i]], modulation, opponent, envelope_value[added[i]]);
				}
            } else {
                idList.erase(idList.begin() + j);
                nErase++;
				if (pInfo) {
					printf("rejected (%.4f, %.4f) with modulation of %.2f and opponency: %.1f, envelope = %.3f\n", norm_x[added[i]], norm_y[added[i]], modulation, opponent, envelope_value[added[i]]);
				}
            }
            //if (one4one) {
            //    std::cout << iV1 << ": modulation = " << modulation << ", opponent = " << opponent << ", envelope_value = " << envelope_value[i] << ", prob = " << _prob << ", sfreq = " << sfreq << ", phase = " << phase << "\n";
            //}
        }
        if (pInfo) {
            std::cout << " erased " << nErase << ".\n";
        }
        assert(idList.size() == strengthList.size());
		if (strictStrength && idList.size() > 0) {
			Float con_irl = std::accumulate(strengthList.begin(), strengthList.end(), 0.0);
        	Float ratio = sSum/con_irl;
        	for (PosInt i=0; i<strengthList.size(); i++) {
        	    strengthList[i] *= ratio; 
        	}
		}
        return idList.size();
    }
};

struct SingleOpponent: LinearReceptiveField {
	// center-surround
    SingleOpponent() {
        rfType = RFtype::singleOppopent;
    }
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, bool strictStrength, Float sig = 1.177) override {
        this->n = n;
        this->sfreq = 0.0;
        this->phase = 0.0;
        this->amp = 0.5*amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}

    Float modulate(Float x, Float y) override {
        return 1.0;
    }

    Float check_opponency(InputType iType, Float &modulation) override {
		return 1.0 * oppose_Cone_OnOff_single(iType, oType, modulation);
	}

    Float get_prob(Float opponent, Float modulation, Float envelope) override {
        return envelope * (0.5 + amp * opponent * modulation);
    }
};

struct DoubleOpponent_Gabor: LinearReceptiveField {
	// Gabor
    DoubleOpponent_Gabor() {
        rfType = RFtype::doubleOppopent_gabor;
    }
    Float check_opponency(InputType iType, Float &modulation) override {
        Float opponent;
        OutputType currType;
		if (modulation < 0.5) {
			// switch to the dominant type
			modulation = 1 - modulation;
			currType = static_cast<OutputType> (5 - static_cast<Size>(oType));
			assert(currType == OutputType::LonMoff || currType == OutputType::LoffMon);
        } else {
            currType = oType;
        }
		opponent = 1.0 * oppose_Cone_OnOff_double(iType, currType, modulation);
        return opponent;
    }

    Float get_prob(Float opponent, Float modulation, Float envelope) {
        Float v = (1.0 + amp * opponent * modulation)/2;
        if (v < 0.5) {
            v = 0;
        }
        v = envelope * v;

	    assert(!isnan(v));
        return v;
    }
};

struct DoubleOpponent_CS: LinearReceptiveField {
	// center-surround
    DoubleOpponent_CS() {
        rfType = RFtype::doubleOppopent_cs;
    }
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, bool strictStrength, Float sig = 1.177) override {
        this->n = n;
        this->sfreq = 0.0;
        this->phase = 0.0;
        this->amp = 0.5*amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}

    Float modulate(Float x, Float y) override {
        return 1.0;
    }

    Float check_opponency(InputType iType, Float &modulation) override {
		return 1.0 * oppose_Cone_OnOff_double(iType, oType, modulation);
	}

    Float get_prob(Float opponent, Float modulation, Float envelope) override {
        return envelope * (0.5 + amp * opponent * modulation);
    }
};

struct NonOpponent_Gabor: LinearReceptiveField {
	// Gabor
    NonOpponent_Gabor() {
        rfType = RFtype::nonOppopent_gabor;
    }
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, bool strictStrength, Float sig = 1.177) override {
        this->n = n;
        this->sfreq = sfreq;
        this->phase = 0;
        this->amp = 0.5*amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}
    Float check_opponency(InputType iType, Float &modulation) override {
        Float opponent;
        OutputType currType;
		if (modulation < 0.5) {
			// switch to the dominant type
		    modulation = 1 - modulation;
			currType = static_cast<OutputType> ( 1- static_cast<Size>(oType) );
			assert(currType == OutputType::LonMon || currType == OutputType::LoffMoff);
		} else {
            currType = oType;
        }
		opponent = 1.0 * match_OnOff(iType, currType, modulation);
        return opponent;
    }

    Float get_prob(Float opponent, Float modulation, Float envelope) override {
        return envelope * (0.5 + amp * opponent * modulation);
    }
};

struct NonOpponent_CS: LinearReceptiveField {
	// center-surround
    NonOpponent_CS() {
        rfType = RFtype::nonOppopent_cs;
    }
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, bool strictStrength, Float sig = 1.177) override {
        this->n = n;
        this->sfreq = 0.0;
        this->phase = 0.0;
        this->amp = 0.5*amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}

    Float modulate(Float x, Float y) override {
        return 1.0;
    }

    Float check_opponency(InputType iType, Float &modulation) override {
		return 1.0 * match_OnOff(iType, oType, modulation);
	}

    Float get_prob(Float opponent, Float modulation, Float envelope) override {
        return envelope * (0.5 + amp * opponent * modulation);
    }
};

struct load_prop {

};

#endif
