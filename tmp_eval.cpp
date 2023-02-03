#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

// Type list impl
namespace meta
{
    template <typename T> struct wrap { using type = T; };

    template <typename... Ts>
    class type_list
    {
        template <size_t I = 0, size_t N, size_t C,
            typename T, typename... Us>
        static consteval inline auto get_at() {
            static_assert(N < C, "Out of bounds");
            if constexpr(I == N) return wrap<T>{};
            else return get_at<I + 1, N, C, Us...>();
        };

        template <typename U, size_t I = 0, size_t C>
        static consteval inline size_t locate_impl() {
            return std::numeric_limits<size_t>::max();
        };

        template <typename U, size_t I = 0, size_t C,
            typename T, typename... Us>
        static consteval inline size_t locate_impl() {
            if constexpr (std::same_as<U, T>) return I;
            else return locate_impl<U, I + 1, C, Us...>();
        };

        static inline constexpr size_t count_impl = sizeof...(Ts);

    public:
        static inline consteval size_t count()
        { return count_impl; };

        template <typename... Us> struct append
        { using type = type_list <Ts..., Us...>; };

        template <typename... Us> struct prepend
        { using type = type_list <Us..., Ts...>; };

        template <size_t N>
        static inline consteval auto get()
        {
            return get_at<0, N, type_list<Ts...>::count_impl, Ts...>();
        };

        template <typename T>
        static inline consteval bool exists()
        {
            return locate_impl<T, 0, type_list<Ts...>::count_impl, Ts...>()
                    != std::numeric_limits<size_t>::max();
        };

        template <typename T>
        static inline consteval size_t locate()
        {
            return locate_impl<T, 0, type_list<Ts...>::count_impl, Ts...>();
        };

        template <bool B, typename... Us>
        struct append_if
        { using type = type_list <Ts..., Us...>; };

        template <typename... Us>
        struct append_if <false, Us...>
        { using type = type_list <Ts...>; };
    };

    template <> class type_list<>
    {
    public:
        static inline consteval size_t count() { return 0; };

        template <typename T>
        static inline consteval bool exists() { return false; };

        template <typename T>
        static inline consteval size_t locate()
        { return std::numeric_limits<size_t>::max(); };

        template <typename... Us> struct append
        { using type = type_list <Us...>; };

        template <typename... Us> struct prepend
        { using type = type_list <Us...>; };

        template <bool B, typename... Us>
        struct append_if
        { using type = type_list <Us...>; };

        template <typename... Us>
        struct append_if <false, Us...>
        { using type = type_list <>; };
    };

    template <typename List, typename T, typename... Ts>
    consteval inline auto tl_flip(meta::type_list<T, Ts...>)
    {
        return tl_flip<typename List::template prepend<T>::type>(meta::type_list<Ts...>{});
    };

    template <typename List>
    consteval inline auto tl_flip(meta::type_list<>) { return List{}; };
}

template <typename... Ts>
struct tl_flip
{
    using type = decltype(meta::tl_flip<meta::type_list<>>(meta::type_list<Ts...>{}));
};

template <size_t N> struct Size_t
{ static inline constexpr size_t value = N; };

template <typename> struct is_idxls : std::false_type {};
template <size_t...Is>
struct is_idxls <meta::type_list<Size_t<Is>...>> : std::true_type {};
template <size_t...Is>
struct is_idxls <const meta::type_list<Size_t<Is>...>> : std::true_type {};
template <> struct is_idxls <meta::type_list<>> : std::true_type {};

template <typename T> concept idxls_c = is_idxls<T>::value;



// token impl
enum E_TOKEN : size_t
{
    CONSTANT,
    VARIABLE,
    OPERATOR,
    EOF,
    WHITESPACE
};

template <char C> struct token
{
    static inline constexpr char tok = C;
    static inline constexpr auto T_t = E_TOKEN::VARIABLE;
};

template <> struct token<'\0'> {
    static inline constexpr char tok = '\0';
    static inline constexpr auto T_t = E_TOKEN::EOF;
};

template <> struct token<' '> {
    static inline constexpr char tok = ' ';
    static inline constexpr auto T_t = E_TOKEN::WHITESPACE;
};

template <> struct token<'+'>  {
    static inline constexpr char tok = '+';
    static inline constexpr auto T_t = E_TOKEN::OPERATOR;
    static inline constexpr auto expr =
        [](double _0, double _1) { return _0 + _1; };
};

template <> struct token<'-'> {
    static inline constexpr char tok = '-';
    static inline constexpr auto T_t = E_TOKEN::OPERATOR;
    static inline constexpr auto expr =
        [](double _0, double _1) { return _0 - _1; };
};

template <> struct token<'*'> {
    static inline constexpr char tok = '*';
    static inline constexpr auto T_t = E_TOKEN::OPERATOR;
    static inline constexpr auto expr =
        [](double _0, double _1) { return _0 * _1; };
};

template <> struct token<'/'> {
    static inline constexpr char tok = '/';
    static inline constexpr auto T_t = E_TOKEN::OPERATOR;
    static inline constexpr auto expr =
        [](double _0, double _1) { return _0 / _1; };
};

template <> struct token<'%'> {
    static inline constexpr char tok = '%';
    static inline constexpr auto T_t = E_TOKEN::OPERATOR;
    static inline constexpr auto expr =
        [](double _0, double _1) { return (long)_0 % (long)_1; };
};

#include <cmath>

template <> struct token<'^'> {
    static inline constexpr char tok = '^';
    static inline constexpr auto T_t = E_TOKEN::OPERATOR;
    static inline constexpr auto expr =
        [](double _0, double _1) { return pow(_0, _1); };
};

template <typename> struct is_tokls : std::false_type {};
template <char... Ts>
struct is_tokls <meta::type_list<token<Ts>...>> : std::true_type {};
template <char... Ts>
struct is_tokls <const meta::type_list<token<Ts>...>> : std::true_type {};
template <> struct is_tokls <meta::type_list<>> : std::true_type {};

template <typename T> concept tokls_c = is_tokls<T>::value;



// String interning : AlexPolt (http://alexpolt.github.io/intern.html)
#define N3599
namespace intern
{
  template<char... Is> struct string {
    static constexpr char const value[ sizeof...(Is) ]{Is...};
    using tokens = meta::type_list<token<Is>...>;
    static_assert( value[ sizeof...(Is) - 1 ] == '\0', "interned string was too long, see $(...) macro" );
    static inline constexpr auto data() { return value; }
  };

  template<char... N> constexpr char const string<N...>::value[];
  template<int N> constexpr char ch ( char const(&s)[N], int i ) { return i < N ? s[i] : '\0'; }
  template<typename T> struct is_string { static const bool value = false; };
  template<char... Is> struct is_string< string<Is...> > { static const bool value = true; };
}

template<typename T, T... C>
auto operator ""_intern() { return intern::string<C..., T{}>{}; }
#define $( s ) decltype( s ## _intern )

template <typename String> concept string_c = intern::is_string<String>::value;



// Lexer / tokenization
template <tokls_c Vs, tokls_c Os, idxls_c Is>
struct Lex_dict
{
    using Vars = Vs;
    using Ops = Os;
    using Idxs = Is;

    using flip =
        Lex_dict<Vs,
        decltype(meta::tl_flip<meta::type_list<>>(Os{})),
        decltype(meta::tl_flip<meta::type_list<>>(Is{}))>;
};

using Empty_lex = Lex_dict<meta::type_list<>, meta::type_list<>, meta::type_list<>>;

template <typename> struct is_lxdc : std::false_type {};
template <tokls_c Vs, tokls_c Os, idxls_c Is>
struct is_lxdc <Lex_dict<Vs, Os, Is>> : std::true_type {};
template <tokls_c Vs, tokls_c Os, idxls_c Is>
struct is_lxdc <const Lex_dict<Vs, Os, Is>> : std::true_type {};

template <typename T> concept lxdc_c = is_lxdc<T>::value;



// Lexer append impl
namespace meta
{
    template <lxdc_c, char, size_t> struct lex_append;

    template <lxdc_c Lex, char C> struct lex_append <Lex, C, E_TOKEN::VARIABLE>
    {
        static inline constexpr bool do_append
            = not (Lex::Vars::template exists<token<C>>());

        static inline constexpr size_t idx_v =
            (do_append) ? Lex::Vars::count() :
            Lex::Vars::template locate<token<C>>();

        using type = Lex_dict<
            typename Lex::Vars::append_if<do_append, token<C>>::type,
            typename Lex::Ops,
            typename Lex::Idxs::append<Size_t<idx_v>>::type
        >;
    };

    template <lxdc_c Lex, char C> struct lex_append <Lex, C, E_TOKEN::OPERATOR>
    {
        using type = Lex_dict<
            typename Lex::Vars,
            typename Lex::Ops::append<token<C>>::type,
            typename Lex::Idxs
        >;
    };

    template <lxdc_c Lex, char C> struct lex_append <Lex, C, E_TOKEN::WHITESPACE>
    { using type = Lex; };

    template <lxdc_c Lex, char C> struct lex_append <Lex, C, E_TOKEN::EOF>
    { using type = Lex; };

}

template <lxdc_c, typename> struct lex_append;
template <lxdc_c Lex, char C> struct lex_append <Lex, token<C>>
{
    using type = typename meta::lex_append<Lex, C, token<C>::T_t>::type;
};

class tokenize
{
    template <lxdc_c Lex>
    static inline consteval auto
        gen_tokens_impl(meta::type_list<>)
    {
        return Lex{};
    };

    template <lxdc_c Lex, char C, char...CC>
    static inline consteval auto
        gen_tokens_impl(meta::type_list<token<C>, token<CC>...>)
    {
        return gen_tokens_impl
            <typename lex_append<Lex, token<C>>::type>
            (meta::type_list<token<CC>...>{});
    };

public:
    template <string_c expr>
    static consteval auto gen_tokens()
    {
        return gen_tokens_impl<Empty_lex>(typename expr::tokens{});
    };
};



// Pointer attributes
namespace meta
{
    template <typename> struct Pointer_attributes {};

    template <typename R, typename...Args>
    struct Pointer_attributes <R(*)(Args...)>
    {
        using Ret_t = R;
        using Arg_t = type_list<Args...>;
        static constexpr inline size_t count = sizeof...(Args);
    };

    template <typename R, class C, typename...Args>
    struct Pointer_attributes <R(C::*)(Args...) const>
    {
        using Ret_t = R;
        using Class_t = C;
        using Arg_t = type_list<Args...>;
        static constexpr inline size_t count = sizeof...(Args);
    };
}

template <auto Lambda>
struct Lambda_wrapper
    : meta::Pointer_attributes<decltype(&decltype(Lambda)::operator ())>
{
    static constexpr inline auto function = Lambda;
};

template <auto...Lambdas>
struct Lambda_list
{
    using value = meta::type_list<Lambda_wrapper<Lambdas>...>;
    static constexpr inline size_t count = sizeof...(Lambdas);
    static constexpr inline size_t arg_count = (Lambda_wrapper<Lambdas>::count + ...) - 1;
};



#include <tuple>

// Lambda constructor
namespace meta
{
    template <auto, typename Input>
    struct Unpack_as
    { using type = Input; };

    template <auto L, size_t _1, size_t _2>
    struct Final
    {
        template <typename... Ts>
        constexpr static inline auto call(std::tuple<Ts...>& t)
        {
            return L(std::get<_1>(t), std::get<_2>(t));
        };
    };

    template <typename, typename> struct Expand;

    template <auto L, auto...LL, size_t N, size_t...Is>
    struct Expand <Lambda_list<L, LL...>, type_list<Size_t<N>, Size_t<Is>...>>
    {
        template <typename... Ts>
        constexpr static inline auto call(std::tuple<Ts...>& t)
        {
            if constexpr(sizeof...(Is) == 1) return Final<L, Is..., N>::call(t);
            else return L(
                Expand<Lambda_list<LL...>,
                type_list<Size_t<Is>...>>::call(t),
                std::get<N>(t)
            );
        };
    };
}

template <char... Vs, char... Os, size_t... Is>
consteval inline auto
get_as_lambda(
    Lex_dict<
        meta::type_list<token<Vs>...>,
        meta::type_list<token<Os>...>,
        meta::type_list<Size_t<Is>...>>)
{
    using lambda_pack = Lambda_list<token<Os>::expr...>;
    using index_pack = meta::type_list<Size_t<Is>...>;
    using tuple_in = std::tuple<typename meta::Unpack_as<Vs, double>::type&&...>;

    return [] (tuple_in t) constexpr
    {
        return meta::Expand<lambda_pack, index_pack>::call(t);
    };
};


// expression builder
template <string_c expr> struct eval;

template <string_c expr>
class builder
{
    static consteval inline auto evaluate()
    {
        using flipped = typename decltype(tokenize::gen_tokens<expr>())::flip;
        return get_as_lambda(flipped{});
    };

    static inline constexpr auto value = evaluate();
    friend struct eval<expr>;
};

#include <string_view>

template <string_c expr>
struct eval
{
protected:
    static inline constexpr auto expression = builder<expr>::value;
    static inline constexpr std::string_view s_name = expr::value;
public:
    template <typename... Ts>
    constexpr double operator()(Ts&&...tt) const
    {
        return expression(std::make_tuple<Ts&&...>(static_cast<Ts&&>(tt)...));
    };

    std::string_view name() const { return eval<expr>::s_name; }
};

auto e = eval<$("a - b * c / d")>{}(1.0, 2.0, 7.0, 3.9);

#include <iostream>

int main() {
    std::cout << "e = " << e << std::endl;
    std::cout << "(1.0 - 2.0 * 7.0 / 3.9) = " << (1.0 - 2.0 * 7.0 / 3.9) << std::endl;
    std::cout << "(((1.0 - 2.0) * 7.0) / 3.9) = " << (((1.0 - 2.0) * 7.0) / 3.9);
}
