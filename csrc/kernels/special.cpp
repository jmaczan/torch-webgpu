#include "kernels/special.h"

#include <c10/util/Exception.h>

#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <sstream>

namespace torch_webgpu::kernels
{
    namespace
    {
        // Float literal, parenthesized so it can follow any operator.
        std::string lit(double v)
        {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "(%.9e)", v);
            return buf;
        }

        // Horner evaluation of c[0]*x^n + ... + c[n] (Cephes polevl order).
        std::string horner(const std::string &x, const std::vector<double> &c)
        {
            std::string e = lit(c[0]);
            for (size_t i = 1; i < c.size(); ++i)
                e = "(" + e + " * " + x + " + " + lit(c[i]) + ")";
            return e;
        }

        // fn NAME(x: f32) -> f32: Clenshaw evaluation of a Chebyshev series
        // (Cephes chbevl), unrolled.
        std::string chbevl_fn(const std::string &name, const std::vector<double> &c)
        {
            std::ostringstream s;
            s << "fn " << name << "(x: f32) -> f32 {\n"
              << "    var b0 = " << lit(c[0]) << ";\n    var b1 = 0.0;\n    var b2 = 0.0;\n";
            for (size_t i = 1; i < c.size(); ++i)
                s << "    b2 = b1; b1 = b0; b0 = x * b1 - b2 + " << lit(c[i]) << ";\n";
            s << "    return 0.5 * (b0 - b2);\n}\n";
            return s.str();
        }

        // ------------------------------------------------ Chebyshev tables
        const std::vector<double> I0E_A = {
            -4.41534164647933937950E-18, 3.33079451882223809783E-17, -2.43127984654795469359E-16,
            1.71539128555513303061E-15, -1.16853328779934516808E-14, 7.67618549860493561688E-14,
            -4.85644678311192946090E-13, 2.95505266312963983461E-12, -1.72682629144155570723E-11,
            9.67580903537323691224E-11, -5.18979560163526290666E-10, 2.65982372468238665035E-9,
            -1.30002500998624804212E-8, 6.04699502254191894932E-8, -2.67079385394061173391E-7,
            1.11738753912010371815E-6, -4.41673835845875056359E-6, 1.64484480707288970893E-5,
            -5.75419501008210370398E-5, 1.88502885095841655729E-4, -5.76375574538582365885E-4,
            1.63947561694133579842E-3, -4.32430999505057594430E-3, 1.05464603945949983183E-2,
            -2.37374148058994688156E-2, 4.93052842396707084878E-2, -9.49010970480476444210E-2,
            1.71620901522208775349E-1, -3.04682672343198398683E-1, 6.76795274409476084995E-1};
        const std::vector<double> I0E_B = {
            -7.23318048787475395456E-18, -4.83050448594418207126E-18, 4.46562142029675999901E-17,
            3.46122286769746109310E-17, -2.82762398051658348494E-16, -3.42548561967721913462E-16,
            1.77256013305652638360E-15, 3.81168066935262242075E-15, -9.55484669882830764870E-15,
            -4.15056934728722208663E-14, 1.54008621752140982691E-14, 3.85277838274214270114E-13,
            7.18012445138366623367E-13, -1.79417853150680611778E-12, -1.32158118404477131188E-11,
            -3.14991652796324136454E-11, 1.18891471078464383424E-11, 4.94060238822496958910E-10,
            3.39623202570838634515E-9, 2.26666899049817806459E-8, 2.04891858946906374183E-7,
            2.89137052083475648297E-6, 6.88975834691682398426E-5, 3.36911647825569408990E-3,
            8.04490411014108831608E-1};
        // Float-precision tables, as PyTorch uses for float.
        const std::vector<double> I1E_A = {
            9.38153738649577178388E-9, -4.44505912879632808065E-8, 2.00329475355213526229E-7,
            -8.56872026469545474066E-7, 3.47025130813767847674E-6, -1.32731636560394358279E-5,
            4.78156510755005422638E-5, -1.61760815825896745588E-4, 5.12285956168575772895E-4,
            -1.51357245063125314899E-3, 4.15642294431288815669E-3, -1.05640848946261981558E-2,
            2.47264490306265168283E-2, -5.29459812080949914269E-2, 1.02643658689847095384E-1,
            -1.76416518357834055153E-1, 2.52587186443633654823E-1};
        const std::vector<double> I1E_B = {
            -3.83538038596423702205E-9, -2.63146884688951950684E-8, -2.51223623787020892529E-7,
            -3.88256480887769039346E-6, -1.10588938762623716291E-4, -9.76109749136146840777E-3,
            7.78576235018280120474E-1};

        const double ERFCX_TABLE[100][7] = {
#include "kernels/erfcx_table.inc"
        };

        // ------------------------------------------------- function bodies
        struct Fn
        {
            std::vector<std::string> deps;
            std::function<std::string()> source;
        };

        const std::map<std::string, Fn> &registry()
        {
            static const std::map<std::string, Fn> r = {
                {"lgamma_f", {{}, []
                              {
                                  // x > 0: shift to z >= 8 (exact product), then Stirling.
                                  // x <= 0: reflection with an accurately reduced sin(pi x).
                                  return std::string(R"(
fn lgamma_pos(x: f32) -> f32 {
    var z = x;
    var p = 1.0;
    for (var i = 0; i < 8; i++) {
        if (z >= 8.0) { break; }
        p = p * z;
        z = z + 1.0;
    }
    let iz = 1.0 / z;
    let iz2 = iz * iz;
    let series = iz * (0.083333333333333333 + iz2 * (-0.0027777777777777778 + iz2 * (0.00079365079365079365 + iz2 * -0.00059523809523809524)));
    return (z - 0.5) * log(z) - z + 0.91893853320467274 + series - log(p);
}
fn lgamma_f(x: f32) -> f32 {
    if (isnan_f(x)) { return x; }
    if (isinf_f(x)) { return inf_f(); }
    if (x > 0.0) { return lgamma_pos(x); }
    let r = x - floor(x);
    let s = sin_f(3.14159265358979323846 * min(r, 1.0 - r));
    return log(3.14159265358979323846 / s) - lgamma_pos(1.0 - x);
}
)");
                              }}},
                {"digamma_f", {{}, []
                               {
                                   return "fn digamma_f(xin: f32) -> f32 {\n"
                                          "    var x = xin;\n"
                                          "    if (x == 0.0) { return select(-inf_f(), inf_f(), signbit_f(x)); }\n"
                                          "    var result = 0.0;\n"
                                          "    if (x < 0.0) {\n"
                                          "        if (x == trunc(x)) { return nan_f(); }\n"
                                          "        let r = x - trunc(x);\n"
                                          "        result = -3.14159265358979323846 / tan_f(3.14159265358979323846 * r);\n"
                                          "        x = 1.0 - x;\n"
                                          "    }\n"
                                          "    for (var i = 0; i < 10; i++) {\n"
                                          "        if (!(x < 10.0)) { break; }\n"
                                          "        result = result - 1.0 / x;\n"
                                          "        x = x + 1.0;\n"
                                          "    }\n"
                                          "    if (x == 10.0) { return result + 2.25175258906672110764; }\n"
                                          "    var y = 0.0;\n"
                                          "    if (x < 1.0e17) {\n"
                                          "        let z = 1.0 / (x * x);\n"
                                          "        y = z * " +
                                          horner("z", {8.33333333333333333333E-2, -2.10927960927960927961E-2,
                                                       7.57575757575757575758E-3, -4.16666666666666666667E-3,
                                                       3.96825396825396825397E-3, -8.33333333333333333333E-3,
                                                       8.33333333333333333333E-2}) +
                                          ";\n"
                                          "    }\n"
                                          "    return log(x) - (0.5 / x) - y + result;\n"
                                          "}\n";
                               }}},
                {"trigamma_f", {{}, []
                                {
                                    return std::string(R"(
fn trigamma_f(xin: f32) -> f32 {
    var x = xin;
    var sgn = 1.0;
    var result = 0.0;
    if (x < 0.5) {
        sgn = -1.0;
        let s = sin_f(3.14159265358979323846 * x);
        result = result - (3.14159265358979323846 * 3.14159265358979323846) / (s * s);
        x = 1.0 - x;
    }
    for (var i = 0; i < 6; i++) {
        result = result + 1.0 / (x * x);
        x = x + 1.0;
    }
    let ixx = 1.0 / (x * x);
    result = result + (1.0 + 1.0 / (2.0 * x) + ixx * (1.0 / 6.0 - ixx * (1.0 / 30.0 - ixx * (1.0 / 42.0)))) / x;
    return sgn * result;
}
)");
                                }}},
                {"zeta_f", {{}, []
                            {
                                // Hurwitz zeta (Cephes); A[i] = (2i+2)! / B_{2i+2}.
                                const double A[12] = {12.0, -720.0, 30240.0, -1209600.0, 47900160.0,
                                                      -1.8924375803183791606e9, 7.47242496e10,
                                                      -2.950130727918164224e12, 1.1646782814350067249e14,
                                                      -4.5979787224074726105e15, 1.8152105401943546773e17,
                                                      -7.1661652561756670113e18};
                                std::ostringstream s;
                                s << R"(
fn zeta_f(x: f32, q: f32) -> f32 {
    let MACHEP = 1.11022302462515654042e-16;
    if (x == 1.0) { return inf_f(); }
    if (x < 1.0) { return nan_f(); }
    if (q <= 0.0) {
        if (q == floor(q)) { return inf_f(); }
        if (x != floor(x)) { return nan_f(); }
    }
    var s = pow_f(q, -x);
    var a = q;
    var b = 0.0;
    var i = 0;
    loop {
        if (!((i < 9) || (a <= 9.0))) { break; }
        i = i + 1;
        a = a + 1.0;
        b = pow_f(a, -x);
        s = s + b;
        if ((-MACHEP * s < b) && (b < MACHEP * s)) { return s; }
        if (i > 100000) { break; }
    }
    let w = a;
    s = s + b * w / (x - 1.0);
    s = s - 0.5 * b;
    a = 1.0;
    var k = 0.0;
    var t = 0.0;
)";
                                for (int j = 0; j < 12; ++j)
                                {
                                    // Once b underflows (f32 flushes denormals) the
                                    // remaining terms are 0; stop before a overflows.
                                    s << "    a = a * (x + k); b = b / w; if (b == 0.0) { return s; }\n"
                                      << "    t = a * b / " << lit(A[j]) << "; s = s + t;\n"
                                      << "    if (abs(t / s) < MACHEP) { return s; }\n"
                                      << "    k = k + 1.0; a = a * (x + k); b = b / w; k = k + 1.0;\n";
                                }
                                s << "    return s;\n}\n";
                                return s.str();
                            }}},
                {"polygamma_f", {{"zeta_f"}, []
                                 {
                                     // n >= 2: fac = (-1)^(n+1) n!, np1 = n + 1. Poles and
                                     // NaN are returned directly (see xlog1py_f).
                                     return std::string(R"(
fn polygamma_f(np1: f32, fac: f32, x: f32) -> f32 {
    if (isnan_f(x)) { return x; }
    if (x <= 0.0 && x == floor(x)) { return select(-inf_f(), inf_f(), fac > 0.0); }
    return fac * zeta_f(np1, x);
}
)");
                                 }}},
                {"frexp_f", {{}, []
                             {
                                 // Integer-only, so exact for denormals too (float ops
                                 // flush them to zero on GPUs). 0/inf/nan: (x, 0).
                                 return std::string(R"(
fn frexp_f(x: f32) -> vec2<u32> {
    let u = bitcast<u32>(x);
    let e = (u >> 23u) & 0xffu;
    let m = u & 0x7fffffu;
    let sgn = u & 0x80000000u;
    if (e == 255u || (e == 0u && m == 0u)) { return vec2<u32>(u, 0u); }
    if (e == 0u) {
        let p = 31u - countLeadingZeros(m);
        let mant = sgn | (126u << 23u) | ((m << (23u - p)) & 0x7fffffu);
        return vec2<u32>(mant, bitcast<u32>(i32(p) - 148));
    }
    return vec2<u32>(sgn | (126u << 23u) | m, bitcast<u32>(i32(e) - 126));
}
)");
                             }}},
                {"i0e_f", {{}, []
                           {
                               return chbevl_fn("chb_i0e_a", I0E_A) + chbevl_fn("chb_i0e_b", I0E_B) +
                                      "fn i0e_f(xin: f32) -> f32 {\n"
                                      "    let x = abs(xin);\n"
                                      "    if (x <= 8.0) { return chb_i0e_a(x / 2.0 - 2.0); }\n"
                                      "    return chb_i0e_b(32.0 / x - 2.0) / sqrt(x);\n"
                                      "}\n";
                           }}},
                {"i0_f", {{"i0e_f"}, []
                          {
                              return std::string("fn i0_f(xin: f32) -> f32 {\n"
                                                 "    let x = abs(xin);\n"
                                                 "    if (x <= 8.0) { return exp(x) * chb_i0e_a(x / 2.0 - 2.0); }\n"
                                                 "    return exp(x) * chb_i0e_b(32.0 / x - 2.0) / sqrt(x);\n"
                                                 "}\n");
                          }}},
                {"i1e_f", {{}, []
                           {
                               return chbevl_fn("chb_i1e_a", I1E_A) + chbevl_fn("chb_i1e_b", I1E_B) +
                                      "fn i1e_f(xin: f32) -> f32 {\n"
                                      "    let x = abs(xin);\n"
                                      "    var out: f32;\n"
                                      "    if (x <= 8.0) { out = chb_i1e_a(x / 2.0 - 2.0) * x; }\n"
                                      "    else { out = chb_i1e_b(32.0 / x - 2.0) / sqrt(x); }\n"
                                      "    return select(out, -out, xin < 0.0);\n"
                                      "}\n";
                           }}},
                {"i1_f", {{"i1e_f"}, []
                          {
                              return std::string("fn i1_f(xin: f32) -> f32 {\n"
                                                 "    let x = abs(xin);\n"
                                                 "    var out: f32;\n"
                                                 "    if (x <= 8.0) { out = exp(x) * x * chb_i1e_a(x / 2.0 - 2.0); }\n"
                                                 "    else { out = exp(x) * chb_i1e_b(32.0 / x - 2.0) / sqrt(x); }\n"
                                                 "    return select(out, -out, xin < 0.0);\n"
                                                 "}\n");
                          }}},
                {"ndtri_f", {{}, []
                             {
                                 const std::vector<double> P0 = {-5.99633501014107895267E1, 9.80010754185999661536E1,
                                                                 -5.66762857469070293439E1, 1.39312609387279679503E1,
                                                                 -1.23916583867381258016E0};
                                 const std::vector<double> Q0 = {1.0, 1.95448858338141759834E0, 4.67627912898881538453E0,
                                                                 8.63602421390890590575E1, -2.25462687854119370527E2,
                                                                 2.00260212380060660359E2, -8.20372256168333339912E1,
                                                                 1.59056225126211695515E1, -1.18331621121330003142E0};
                                 const std::vector<double> P1 = {4.05544892305962419923E0, 3.15251094599893866154E1,
                                                                 5.71628192246421288162E1, 4.40805073893200834700E1,
                                                                 1.46849561928858024014E1, 2.18663306850790267539E0,
                                                                 -1.40256079171354495875E-1, -3.50424626827848203418E-2,
                                                                 -8.57456785154685413611E-4};
                                 const std::vector<double> Q1 = {1.0, 1.57799883256466749731E1, 4.53907635128879210584E1,
                                                                 4.13172038254672030440E1, 1.50425385692907503408E1,
                                                                 2.50464946208309415979E0, -1.42182922854787788574E-1,
                                                                 -3.80806407691578277194E-2, -9.33259480895457427372E-4};
                                 const std::vector<double> P2 = {3.23774891776946035970E0, 6.91522889068984211695E0,
                                                                 3.93881025292474443415E0, 1.33303460815807542389E0,
                                                                 2.01485389549179081538E-1, 1.23716634817820021358E-2,
                                                                 3.01581553508235416007E-4, 2.65806974686737550832E-6,
                                                                 6.23974539184983293730E-9};
                                 const std::vector<double> Q2 = {1.0, 6.02427039364742014255E0, 3.67983563856160859403E0,
                                                                 1.37702099489081330271E0, 2.16236993594496635890E-1,
                                                                 1.34204006088543189037E-2, 3.28014464682127739104E-4,
                                                                 2.89247864745380683936E-6, 6.79019408009981274425E-9};
                                 return "fn ndtri_f(y0: f32) -> f32 {\n"
                                        "    if (y0 == 0.0) { return -inf_f(); }\n"
                                        "    if (y0 == 1.0) { return inf_f(); }\n"
                                        "    if (y0 < 0.0 || y0 > 1.0 || isnan_f(y0)) { return nan_f(); }\n"
                                        "    var code = true;\n"
                                        "    var y = y0;\n"
                                        "    if (y > 1.0 - 0.13533528323661269189) { y = 1.0 - y; code = false; }\n"
                                        "    if (y > 0.13533528323661269189) {\n"
                                        "        y = y - 0.5;\n"
                                        "        let y2 = y * y;\n"
                                        "        let x = y + y * (y2 * " + horner("y2", P0) + " / " + horner("y2", Q0) + ");\n"
                                        "        return x * 2.50662827463100050242;\n"
                                        "    }\n"
                                        "    let x = sqrt(-2.0 * log(y));\n"
                                        "    let x0 = x - log(x) / x;\n"
                                        "    let z = 1.0 / x;\n"
                                        "    var x1: f32;\n"
                                        "    if (x < 8.0) { x1 = z * " + horner("z", P1) + " / " + horner("z", Q1) + "; }\n"
                                        "    else { x1 = z * " + horner("z", P2) + " / " + horner("z", Q2) + "; }\n"
                                        "    let r = x0 - x1;\n"
                                        "    return select(-r, r, !code);\n"
                                        "}\n";
                             }}},
                {"erfcx_f", {{}, []
                             {
                                 std::ostringstream s;
                                 s << "fn erfcx_y100(y100: f32) -> f32 {\n"
                                   << "    switch (i32(y100)) {\n";
                                 for (int k = 0; k < 100; ++k)
                                 {
                                     std::vector<double> c(ERFCX_TABLE[k], ERFCX_TABLE[k] + 7);
                                     std::vector<double> rev(c.rbegin(), c.rend());
                                     s << "        case " << k << ": { let t = 2.0 * y100 - " << (2 * k + 1)
                                       << ".0; return " << horner("t", rev) << "; }\n";
                                 }
                                 s << "        default: { return 1.0; }\n    }\n}\n";
                                 s << R"(
fn erfcx_f(x: f32) -> f32 {
    if (isnan_f(x)) { return x; }
    if (x >= 0.0) {
        if (x > 50.0) {
            let ispi = 0.56418958354775628694807945156;
            if (x > 5.0e7) { return ispi / x; }
            return ispi * ((x * x) * (x * x + 4.5) + 2.0) / (x * ((x * x) * (x * x + 5.0) + 3.75));
        }
        return erfcx_y100(400.0 / (4.0 + x));
    }
    if (x < -26.7) { return inf_f(); }
    if (x < -6.1) { return 2.0 * exp(x * x); }
    return 2.0 * exp(x * x) - erfcx_y100(400.0 / (4.0 - x));
}
)";
                                 return s.str();
                             }}},
                {"log_ndtr_f", {{"erfcx_f"}, []
                                {
                                    return std::string(R"(
fn log_ndtr_f(x: f32) -> f32 {
    let t = x * 0.707106781186547524400844362104849039;
    if (x < -1.0) { return log(erfcx_f(-t) / 2.0) - t * t; }
    return log1p_f(-erfc_f(t) / 2.0);
}
)");
                                }}},
                {"entr_f", {{}, []
                            {
                                return std::string(R"(
fn entr_f(a: f32) -> f32 {
    if (isnan_f(a)) { return a; }
    if (a > 0.0) { return -a * log(a); }
    if (a == 0.0) { return 0.0; }
    return -inf_f();
}
)");
                            }}},
                {"xlog1py_f", {{}, []
                               {
                                   return std::string(R"(
fn xlog1py_f(x: f32, y: f32) -> f32 {
    // NaN results are returned directly, never fed into arithmetic: under
    // driver fast-math, `x * NaN-constant` may fold to anything.
    if (isnan_f(y)) { return nan_f(); }
    if (x == 0.0) { return 0.0; }
    if (y < -1.0) { return nan_f(); }
    if (y == -1.0) { return select(-inf_f(), inf_f(), x < 0.0); }
    return x * log1p_f(y);
}
)");
                               }}},
            };
            return r;
        }
    }

    std::string special_wgsl(const std::vector<std::string> &names)
    {
        const auto &reg = registry();
        std::set<std::string> done;
        std::string out;
        std::function<void(const std::string &)> add = [&](const std::string &n)
        {
            if (done.count(n))
                return;
            auto it = reg.find(n);
            TORCH_INTERNAL_ASSERT(it != reg.end(), "unknown special function ", n);
            for (const auto &d : it->second.deps)
                add(d);
            done.insert(n);
            out += it->second.source();
        };
        for (const auto &n : names)
            add(n);
        return out;
    }
}
