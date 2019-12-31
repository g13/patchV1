#ifndef RFTYPE_H
#define RFTYPE_H
#include <vector>
#include <numeric>
#include "../types.h"
#include "../util/util.h"
#include <cassert>
#include <type_traits>
//#include <algorithm>

enum class RFtype: Size {
	nonOppopent_gabor = 0,  // different cone input correlates, i.e., have the same sign, in gabor form, find in V1
	nonOppopent_cs = 1,  // different cone input correlates, i.e., have the same sign, in center surround form not necessarily concentric, find in V1
	singleOppopent = 2, // center surround cone opponency, LGN and V1
	doubleOppopent_cs = 3, // center surround cone and spatial opponency, V1
    doubleOppopent_gabor = 4 // cone and spatial opponency in Gabor profile, V1
};

enum class OutputType: Size { // V1 local RF 
	// if RF has double peaks, choose the first on the left
	// surround ignored when RF have double peaks
	// non-opponent
	LonMon = 0,
	LoffMoff = 1,
	// opponent
	LonMoff = 2,
	LoffMon = 3
};

enum class InputType: Size { // LGN
    // center-surround
    LonMoff = 0,
    LoffMon = 1,
    MonLoff = 2,
    MoffLon = 3
	// even -> on, odd -> off
    // ignore mixed surround, almost always from a different cone type
};

typedef std::underlying_type<RFtype>::type RFtype_t;
typedef std::underlying_type<InputType>::type InputType_t;
typedef std::underlying_type<OutputType>::type OutputType_t;

// normalize x to [-1,1], y to baRatio*[-1,1]
auto transform_coord_to_unitRF(Float x, Float y, const Float mx, const Float my, const Float theta, const Float a) {
    // a is half-width at the x-axis
    x = (x - mx)/a;
    y = (y - my)/a;
    Float new_x, new_y;
    new_x = cos(theta) * x + sin(theta) * y;
	new_y = -sin(theta) * x + cos(theta) * y;
    return std::make_pair(new_x, new_y);
}

inline Int match_OnOff(InputType iType, OutputType oType, Float &modulation) {
	Int match;
   	switch (oType) {
   	    case OutputType::LonMon:
   			switch (iType) {
   				case InputType::LonMoff: case InputType::MonLoff: match = 1;
   					break;
   				case InputType::LoffMon: case InputType::MoffLon: match = -1;
					break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
   	        break;
   	    case OutputType::LoffMoff:
   	        switch (iType) {
   				case InputType::LoffMon: case InputType::MoffLon: match = 1;
   					break;
   				case InputType::LonMoff: case InputType::MonLoff: match = -1;
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
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
   	        break;
   	    case OutputType::LoffMon:
   	        switch (iType) {
   				case InputType::LoffMon: case InputType::MonLoff: opponent = 1;
   					break;
   				case InputType::LonMoff: case InputType::MoffLon: opponent = -1;
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
				case InputType::LoffMon: case InputType::MoffLon: case InputType::MonLoff: opponent = -1;
					break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
   	        break;
   	    case OutputType::LoffMon:
   	        switch (iType) {
   				case InputType::LoffMon: opponent = 1;
   					break;
   				case InputType::LonMoff: case InputType::MonLoff: case InputType::MoffLon: opponent = -1;
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
    virtual void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, Float sig = 1.177) {
        this->n = n;
        this->sfreq = sfreq;
        this->phase = phase;
        this->amp = amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}

	virtual void clear() {
		prob.clear();
	}
 
    virtual Size construct_connection(std::vector<Float> &x, std::vector<Float> &y, std::vector<InputType> &iType, std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen, Float percent) {
        Size nConnected;
        if (n > 0) {
		    prob.reserve(n);
            // putative RF center
			Float cx, cy;
            std::tie(cx, cy) = average(x, y);
		    for (Size i = 0; i < n; i++) {
                Float norm_x, norm_y;
                // orient and normalize LGN coord
		        std::tie(norm_x, norm_y) = transform_coord_to_unitRF(x[i], y[i], cx, cy, theta, a);
                // calc. prob. dist. distance-dependent envelope at coord.
                Float envelope = get_envelope(norm_x, norm_y, amp, baRatio, sig);
                // calc. modulation of prob at coord.
                Float modulation = modulate(norm_x, norm_y);
                /* TEST with no modulation: bool RefShift = false;
                if (modulation < 0.5) {
                    RefShift = true;
                } */
                // calc. cone and position opponency at coord.
	            Float opponent = check_opponency(iType[i], modulation);
                prob.push_back(get_prob(opponent, modulation, envelope));
                /* TEST with no modulation: if (opponent < 0.0 && RefShift || (opponent < 0.0 && (rfType == RFtype::doubleOppopent_cs || rfType == RFtype::singleOppopent || rfType == RFtype::nonOppopent_cs))) {
                    if (prob.back() > 0.0) {
                        std::cout << envelope << " * (0.5 + " << amp << " * " << opponent << " * " << modulation <<  ") = " << prob.back() << "\n";
                        assert(prob.back() == 0.0);
                    }
                } */
            }
            normalize(percent);
            // make connection and update ID and strength list
            nConnected = connect(idList, strengthList, rGen);
        }  else {
            nConnected = 0;
        }
        return nConnected;
    }
    // Probability envelope based on distance
    virtual Float get_envelope(Float x, Float y, Float amp, Float baRatio, Float sig = 1.1775) {
        return exp(-0.5*(pow(x/sig,2)+pow(y/(sig*baRatio),2)));
        //return 1.0;
        // baRatio comes from Dow et al., 1981
    }
    // Full cosine modulation, modulation is on cone and on-off types
    virtual Float modulate(Float x, Float y) {
        return 0.5 + 0.5 * cos(sfreq * x * M_PI + phase);
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
    void normalize(Float percent) {
	    // average connection probability is controlled at percent.
        const Float norm = std::accumulate(prob.begin(), prob.end(), 0.0) / (percent * prob.size());
	    //std::cout << "norm = " << norm << "\n";
	    //print_list<Float>(prob);
	    assert(!isnan(norm));
        //Float sum = 0.0;
        for (Size i=0; i<prob.size(); i++) {
            prob[i] = prob[i] / norm;
            //sum += prob[i];
        }
    //std::cout << percent*prob.size() << " ~ " << sum << "\n";
    }
    // make connections
    virtual Size connect(std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen) {
		// make connections and normalized strength i.e., if prob > 1 then s = 1 else s = prob
        std::uniform_real_distribution<Float> uniform(0,1);
		strengthList.reserve(n);
		std::vector<Size> newList;
		newList.reserve(n);
		for (Size i = 0; i < n; i++) {
			if (uniform(rGen) < prob[i]) {
				newList.push_back(idList[i]);
				//if (prob[i] > 1) {
					strengthList.push_back(prob[i]);
				//} else {
					//strengthList.push_back(1);
				//}
			}
		}
		idList = newList;
		idList.shrink_to_fit();
        return idList.size();
    }
};

struct SingleOpponent: LinearReceptiveField {
	// center-surround
    SingleOpponent() {
        rfType = RFtype::singleOppopent;
    }
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, Float sig = 1.177) {
        this->n = n;
        this->sfreq = 0.0;
        this->phase = 0.0;
        this->amp = amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}

    Float modulate(Float x, Float y) override {
        return 0.5;
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
        return envelope * (1.0 + amp * opponent * modulation);
    }
};

struct DoubleOpponent_CS: LinearReceptiveField {
	// center-surround
    DoubleOpponent_CS() {
        rfType = RFtype::doubleOppopent_cs;
    }
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, Float sig = 1.177) override {
        this->n = n;
        this->sfreq = 0.0;
        this->phase = 0.0;
        this->amp = amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}

    Float modulate(Float x, Float y) override {
        return 0.5;
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
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, Float sig = 1.177) override {
        this->n = n;
        this->sfreq = sfreq;
        this->phase = sfreq;
        this->amp = 0.5;
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
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, Float sig = 1.177) override {
        this->n = n;
        this->sfreq = 0.0;
        this->phase = 0.0;
        this->amp = amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}

    Float modulate(Float x, Float y) override {
        return 0.5;
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
