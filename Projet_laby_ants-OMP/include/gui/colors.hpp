/**
 * @brief Color models
 */
#ifndef _GUI_COLORS_HPP_
#define _GUI_COLORS_HPP_
#include <initializer_list>
#include <algorithm>
#include <cmath>

namespace gui
{
	namespace color
	{
		/**
		 * @brief      RGB color model + alpha channel
		 */
		struct rgba
		{
			// Red, Green, Blue and Alpha channel
			double r,g,b,a;// Between 0. and 1.
			rgba() : r(0.),g(0.),b(0.),a{1.} {}

			rgba( const std::initializer_list<double>& lc ) {
				auto it = lc.begin();
				r = *it; ++it;
				g = *it; ++it;
				b = *it;
				if ( lc.size() > 3 ) {
					++it; a = *it;
				} else a = 1.;
			}
			rgba( const std::initializer_list<int>& lc ) {
				auto it = lc.begin();
				r = (*it)/255.; ++it;
				g = (*it)/255; ++it;
				b = (*it)/255.;
				if ( lc.size() > 3 ) {
					++it; a = (*it)/255.;
				} else a = 1.;
			}
			rgba( const rgba& ) = default;
		};

		/**
		 * @brief      HSV color model + alpha channel
		 */
		struct hsva
		{
			// Hue, Saturation, Value and Alpha channel
			double h, s, v, a; // 0. <= h < 360; 0. <= s,v,a <= 1.
			hsva( const std::initializer_list<double>& lc ) {
				auto it = lc.begin();
				h = *it; ++it;
				s = *it; ++it;
				v = *it;
				if ( lc.size() > 3){
					++it; a = *it;
				} else a = 1.;
			}
			hsva( const std::initializer_list<int>& lc ) {
				auto it = lc.begin();
				h = (*it); ++it;
				s = (*it)/255; ++it;
				v = (*it)/255.;
				if ( lc.size() > 3 ) {
					++it; a = (*it)/255.;
				} else a = 1.;
			}
			hsva( const hsva& ) = default;
			hsva( const rgba& col ) {
				double Cmax = std::max({col.r, col.g, col.b});
				double Cmin = std::min({col.r, col.g, col.b});
				double delta = Cmax - Cmin;
				h = ( delta = 0. ? 0. : 
					  Cmax == col.r ? 60*std::fmod((col.g-col.b)/delta,6.) :
					  Cmax == col.g ? 60*((col.b-col.r)/delta+2.) :
					                  60*((col.r-col.g)/delta+4.) );
				s = ( Cmax == 0. ? 0. : delta/Cmax );
				v = Cmax;
				a = col.a;
			}
			explicit operator rgba() const
			{
				double c = v*s;
				double x = c*(1-std::abs(std::fmod(h/60.,2)-1));
				double m = v - c;
				double rp, gp, bp;
				if ( h < 60.) {
					rp = c; gp = x; bp = 0.;
				} else if ( h < 120 ) {
					rp = x; gp = c; bp = 0.;
				} else if ( h < 180. ) {
					rp = 0.; gp = c; bp = x;
				} else if ( h < 240. ) {
					rp = 0.; gp = x; bp = c;
				} else if ( h < 300. ) {
					rp = x; gp = 0.; bp = c;
				} else // if ( h < 360. ) 
				{
					rp = c; gp = 0; bp = x;
				}
				return rgba{rp+m, gp+m, bp+m, a};
			}

		};
	}
}

#endif