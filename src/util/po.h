#ifndef PO_H
#define PO_H

#include "../types.h"
#include <regex>
#include <vector>
#include <boost/program_options.hpp>
#include <string>

namespace boost {
	namespace program_options {
		template<>
		void validate<Float, char>(boost::any& v,
			const std::vector<std::basic_string<char> >& s,
			std::vector<Float>*,
			int)
		{
			if (v.empty()) {
				v = boost::any(std::vector<Float>());
			}
			std::vector<Float>* tv = boost::any_cast<std::vector<Float>>(&v);
			assert(NULL != tv);

			std::vector<std::basic_string<char>> strs;
			static std::regex r("\\s?(;|,|\\{|\\}|\\s)\\s?");
			std::vector<std::string>  str;
			for (unsigned i = 0; i < s.size(); ++i) {
				std::vector<std::string> substr(
					std::sregex_token_iterator(s[i].begin(), s[i].end(), r, -1),
					std::sregex_token_iterator()
				);
				for (unsigned j = 0; j < substr.size(); ++j) {
					str.push_back(substr[j]);
				}
			}

			for (unsigned i = 0; i < str.size(); ++i)
			{
				try {
					/* We call validate so that if user provided
					   a validator for class T, we use it even
					   when parsing vector<T>.  */
					boost::any a;
					std::vector<std::basic_string<char> > cv;
					cv.push_back(str[i]);
					validate(a, cv, (Float*)0, 0);
					tv->push_back(boost::any_cast<Float>(a));
				}
				catch (const bad_lexical_cast& /*e*/) {
					boost::throw_exception(invalid_option_value(s[i]));
				}
			}
		}

		template<>
		void validate<Size, char>(boost::any& v,
			const std::vector<std::basic_string<char> >& s,
			std::vector<Size>*,
			int)
		{
			if (v.empty()) {
				v = boost::any(std::vector<Size>());
			}
			std::vector<Size>* tv = boost::any_cast<std::vector<Size>>(&v);
			assert(NULL != tv);

			std::vector<std::basic_string<char>> strs;
			static std::regex r("\\s?(;|,|\\{|\\}|\\s)\\s?");
			std::vector<std::string>  str;
			for (unsigned i = 0; i < s.size(); ++i) {
				std::vector<std::string> substr(
					std::sregex_token_iterator(s[i].begin(), s[i].end(), r, -1),
					std::sregex_token_iterator()
				);
				for (unsigned j = 0; j < substr.size(); ++j) {
					str.push_back(substr[j]);
				}
			}

			for (unsigned i = 0; i < str.size(); ++i)
			{
				try {
					/* We call validate so that if user provided
					   a validator for class T, we use it even
					   when parsing vector<T>.  */
					boost::any a;
					std::vector<std::basic_string<char> > cv;
					cv.push_back(str[i]);
					validate(a, cv, (Size*)0, 0);
					tv->push_back(boost::any_cast<Size>(a));
				}
				catch (const bad_lexical_cast& /*e*/) {
					boost::throw_exception(invalid_option_value(s[i]));
				}
			}
		}
	}
}
#endif
