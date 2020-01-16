#ifndef PO_H
#define PO_H
#include <boost/program_options.hpp>
namespace boost {
	namespace program_options {
		template<>
		void validate<float, char>(boost::any& v,
			const std::vector<std::basic_string<char> >& s,
			std::vector<float>*,
			int)
		{
			if (v.empty()) {
				v = boost::any(std::vector<float>());
			}
			std::vector<float>* tv = boost::any_cast<std::vector<float>>(&v);
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
					validate(a, cv, (float*)0, 0);
					tv->push_back(boost::any_cast<float>(a));
				}
				catch (const bad_lexical_cast& /*e*/) {
					boost::throw_exception(invalid_option_value(s[i]));
				}
			}
		}
	}
}
#endif
