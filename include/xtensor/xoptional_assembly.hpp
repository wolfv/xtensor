/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XOPTIONAL_ASSEMBLY_HPP
#define XOPTIONAL_ASSEMBLY_HPP

#include "xconcepts.hpp"
#include "xoptional.hpp"
#include "xoptional_assembly_base.hpp"
#include "xsemantic.hpp"

namespace xt
{

    /**********************************
     * xoptional_assembly declaration *
     **********************************/

    template <class VE, class FE, bool EXISTS = true>
    class xoptional_assembly;

    template <class VE, class FE, bool EXISTS>
    struct xcontainer_inner_types<xoptional_assembly<VE, FE, EXISTS>>
    {
        using value_expression = VE;
        using flag_expression = FE;
        using temporary_type = xoptional_assembly<VE, FE, EXISTS>;

        constexpr static bool has_value_truthy = EXISTS;
    };

    template <class VE, class FE, bool EXISTS>
    constexpr bool xcontainer_inner_types<xoptional_assembly<VE, FE, EXISTS>>::has_value_truthy;

    template <class VE, class FE, bool EXISTS>
    struct xiterable_inner_types<xoptional_assembly<VE, FE, EXISTS>>
    {
        using assembly_type = xoptional_assembly<VE, FE, EXISTS>;
        using inner_shape_type = typename VE::inner_shape_type;
        using stepper = xoptional_assembly_stepper<assembly_type, false>;
        using const_stepper = xoptional_assembly_stepper<assembly_type, true>;
    };

    /**
     * @class xoptional_assembly
     * @brief Dense multidimensional container holding optional values,
     * optimized for tensor operations.
     *
     * The xoptional_assembly class implements a dense multidimensional container
     * hoding optional values. This container is optimized of tensor operations:
     * contrary to xarray_optional, xoptional_assembly holds two separated expressions,
     * one for the values, the other for the missing mask.
     *
     * @tparam VE The type of expression holding the values.
     * @tparam FE The type of expression holding the missing mask.
     */
    template <class VE, class FE, bool EXISTS>
    class xoptional_assembly : public xoptional_assembly_base<xoptional_assembly<VE, FE, EXISTS>>,
                               public xcontainer_semantic<xoptional_assembly<VE, FE, EXISTS>>
    {
    public:

        using self_type = xoptional_assembly<VE, FE, EXISTS>;
        using base_type = xoptional_assembly_base<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using value_expression = typename base_type::value_expression;
        using flag_expression = typename base_type::flag_expression;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        xoptional_assembly();
        explicit xoptional_assembly(const shape_type& shape, layout_type l = base_type::static_layout);
        explicit xoptional_assembly(const shape_type& shape, const value_type& value, layout_type l = base_type::static_layout);
        explicit xoptional_assembly(const shape_type& shape, const strides_type& strides);
        explicit xoptional_assembly(const shape_type& shape, const strides_type& strides, const value_type& value);

        xoptional_assembly(const VE& ve);
        xoptional_assembly(VE&& ve);

        template <class OVE, class OFE, XTENSOR_REQUIRE<is_xexpression<OVE>::value && is_xexpression<OFE>::value>>
        xoptional_assembly(OVE&& ove, OFE&& ofe);

        xoptional_assembly(const value_type& value);
        xoptional_assembly(nested_initializer_list_t<value_type, 1> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 2> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 3> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 4> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 5> t);

        template <class S = shape_type>
        static xoptional_assembly from_shape(S&& s);

        ~xoptional_assembly() = default;

        xoptional_assembly(const xoptional_assembly&) = default;
        xoptional_assembly& operator=(const xoptional_assembly&) = default;

        xoptional_assembly(xoptional_assembly&&) = default;
        xoptional_assembly& operator=(xoptional_assembly&&) = default;

        template <class E>
        xoptional_assembly(const xexpression<E>& e);

        template <class E>
        xoptional_assembly& operator=(const xexpression<E>& e);

    private:

        value_expression& value_impl() noexcept;
        const value_expression& value_impl() const noexcept;

        flag_expression& flag_impl() noexcept;
        const flag_expression& flag_impl() const noexcept;

        value_expression m_value;
        flag_expression m_flag;

        friend class xoptional_assembly_base<xoptional_assembly<VE, FE, EXISTS>>;
    };

    /******************************************
     * xoptional_assembly_adaptor declaration *
     ******************************************/

    template <class VEC, class FEC, bool EXISTS = true>
    class xoptional_assembly_adaptor;

    template <class VEC, class FEC, bool EXISTS>
    struct xcontainer_inner_types<xoptional_assembly_adaptor<VEC, FEC, EXISTS>>
    {
        using value_expression = std::remove_reference_t<VEC>;
        using flag_expression = std::remove_reference_t<FEC>;
        using temporary_type = xoptional_assembly<value_expression, flag_expression>;

        constexpr static bool has_value_truthy = EXISTS;
    };

    template <class VEC, class FEC, bool EXISTS>
    constexpr bool xcontainer_inner_types<xoptional_assembly_adaptor<VEC, FEC, EXISTS>>::has_value_truthy;

    template <class VEC, class FEC, bool EXISTS>
    struct xiterable_inner_types<xoptional_assembly_adaptor<VEC, FEC, EXISTS>>
    {
        using assembly_type = xoptional_assembly_adaptor<VEC, FEC, EXISTS>;
        using inner_shape_type = typename std::decay_t<VEC>::inner_shape_type;
        using stepper = xoptional_assembly_stepper<assembly_type, false>;
        using const_stepper = xoptional_assembly_stepper<assembly_type, true>;
    };

    /**
     * @class xoptional_assembly_adaptor
     * @brief Dense multidimensional adaptor holding optional values,
     * optimized for tensor operations.
     *
     * The xoptional_assembly_adaptor class implements a dense multidimensional adaptor
     * hoding optional values. It is used to provide an optional expression semantic
     * to two tensor expressions, one holding the value, the hother holding the missing mask.
     *
     * @tparam VEC The closure for the type of expression holding the values.
     * @tparam FE The closure for the type of expression holding the missing mask.
     */
    template <class VEC, class FEC, bool EXISTS>
    class xoptional_assembly_adaptor : public xoptional_assembly_base<xoptional_assembly_adaptor<VEC, FEC, EXISTS>>,
                                       public xcontainer_semantic<xoptional_assembly_adaptor<VEC, FEC, EXISTS>>
    {
    public:

        using self_type = xoptional_assembly_adaptor<VEC, FEC, EXISTS>;
        using base_type = xoptional_assembly_base<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using value_expression = typename base_type::value_expression;
        using flag_expression = typename base_type::flag_expression;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using temporary_type = typename semantic_base::temporary_type;

        constexpr static bool has_value_truthy = EXISTS;

        template <class OVE, class OFE>
        xoptional_assembly_adaptor(OVE&& ve, OFE&& fe);

        ~xoptional_assembly_adaptor() = default;

        xoptional_assembly_adaptor(const xoptional_assembly_adaptor&) = default;
        xoptional_assembly_adaptor& operator=(const xoptional_assembly_adaptor&);

        xoptional_assembly_adaptor(xoptional_assembly_adaptor&&) = default;
        xoptional_assembly_adaptor& operator=(xoptional_assembly_adaptor&&);
        xoptional_assembly_adaptor& operator=(temporary_type&&);

        template <class E>
        xoptional_assembly_adaptor& operator=(const xexpression<E>& e);

    private:

        value_expression& value_impl() noexcept;
        const value_expression& value_impl() const noexcept;

        flag_expression& flag_impl() noexcept;
        const flag_expression& flag_impl() const noexcept;

        VEC m_value;
        FEC m_flag;

        friend class xoptional_assembly_base<xoptional_assembly_adaptor<VEC, FEC, EXISTS>>;
    };

    /*************************************
     * xoptional_assembly implementation *
     *************************************/

    namespace detail
    {
        template <class T, class S>
        inline void nested_optional_copy(T&& iter, const S& s)
        {
            iter->value() = s.value();
            iter->flag() = s.flag();
            ++iter;
        }

        template <class T, class S>
        inline void nested_optional_copy(T&& iter, std::initializer_list<S> s)
        {
            for (auto it = s.begin(); it != s.end(); ++it)
            {
                nested_optional_copy(std::forward<T>(iter), *it);
            }
        }
    }

    /**
     * @name Constructors
     */
    //@{
    /**
     * Allocates an uninitialized xoptional_assembly that holds 0 element.
     */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly()
        : m_value(), m_flag()
    {
    }

    /**
     * Allocates an uninitialized xoptional_assembly with the specified shape and
     * layout_type.
     * @param shape the shape of the xoptional_assembly
     * @param l the layout_type of the xoptional_assembly
     */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(const shape_type& shape, layout_type l)
        : m_value(shape, l), m_flag(shape, l)
    {
    }

    /**
     * Allocates an xoptional_assembly with the specified shape and layout_type. Elements
     * are initialized to the specified value.
     * @param shape the shape of the xoptional_assembly
     * @param value the value of the elements
     * @param l the layout_type of the xoptional_assembly
     */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(const shape_type& shape, const value_type& value, layout_type l)
        : m_value(shape, value.value(), l), m_flag(shape, value.flag(), l)
    {
    }

    /**
     * Allocates an uninitialized xoptional_assembly with the specified shape and strides.
     * @param shape the shape of the xoptional_assembly
     * @param strides the strides of the xoptional_assembly
     */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(const shape_type& shape, const strides_type& strides)
        : m_value(shape, strides), m_flag(shape, strides)
    {
    }

    /**
     * Allocates an uninitialized xoptional_assembly with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the xoptional_assembly
     * @param strides the strides of the xoptional_assembly
     * @param value the value of the elements
     */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(const shape_type& shape, const strides_type& strides, const value_type& value)
        : m_value(shape, strides, value.value()), m_flag(shape, strides, value.flag())
    {
    }

    /**
     * Allocates an xoptional_assembly that holds a single element initialized to the
     * specified value.
     * @param value the value of the element
     */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(const value_type& value)
        : m_value(value.value()), m_flag(value.flag())
    {
    }

    /**
     * Allocates an xoptional_assembly from the specified value expression. The flag
     * expression is initialized as if no value is missing.
     * @param ve the expression holding the values
     */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(const VE& ve)
        : m_value(ve), m_flag(ve.shape(), EXISTS, ve.layout())
    {
    }

    /**
     * Allocates an xoptional_assembly from the specified value expression. The flag
     * expression is initialized as if no value is missing. The value expression is
     * moved inside the xoptional_assembly and is therefore not available after
     * the xoptional_assembly has been constructed.
     * @param ve the expression holding the values
     */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(VE&& ve)
        : m_value(std::move(ve)), m_flag(ve.shape(), EXISTS, ve.layout())
    {
    }

    /**
     * Allocates an xoptional_assembly from the specified value expression and missing
     * mask expression.
     * @param ove the expression holding the values
     * @param ofe the expression holding the missing mask
     */
    template <class VE, class FE, bool EXISTS>
    template <class OVE, class OFE, typename>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(OVE&& ove, OFE&& ofe)
        : m_value(std::forward<OVE>(ove)), m_flag(std::forward<OFE>(ofe))
    {
    }
    //@}

    /**
    * @name Constructors from initializer list
    */
    //@{
    /**
    * Allocates a one-dimensional xoptional_assembly.
    * @param t the elements of the xoptional_assembly
    */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(nested_initializer_list_t<value_type, 1> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->storage_begin(), t)
                  : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
    * Allocates a two-dimensional xoptional_assembly.
    * @param t the elements of the xoptional_assembly
    */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(nested_initializer_list_t<value_type, 2> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->storage_begin(), t)
                  : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
    * Allocates a three-dimensional xoptional_assembly.
    * @param t the elements of the xoptional_assembly
    */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(nested_initializer_list_t<value_type, 3> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->storage_begin(), t)
                  : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
    * Allocates a four-dimensional xoptional_assembly.
    * @param t the elements of the xoptional_assembly
    */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(nested_initializer_list_t<value_type, 4> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->storage_begin(), t)
                  : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
    * Allocates a five-dimensional xoptional_assembly.
    * @param t the elements of the xoptional_assembly
    */
    template <class VE, class FE, bool EXISTS>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(nested_initializer_list_t<value_type, 5> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->storage_begin(), t)
                  : nested_copy(this->template begin<layout_type::row_major>(), t);
    }
    //@}

    /**
     * Allocates and returns an xoptional_assembly with the specified shape.
     * @param s the shape of the xoptional_assembly
     */
    template <class VE, class FE, bool EXISTS>
    template <class S>
    inline xoptional_assembly<VE, FE, EXISTS> xoptional_assembly<VE, FE, EXISTS>::from_shape(S&& s)
    {
        shape_type shape = xtl::forward_sequence<shape_type>(s);
        return self_type(shape);
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended copy constructor.
     */
    template <class VE, class FE, bool EXISTS>
    template <class E>
    inline xoptional_assembly<VE, FE, EXISTS>::xoptional_assembly(const xexpression<E>& e)
        : base_type()
    {
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class VE, class FE, bool EXISTS>
    template <class E>
    inline auto xoptional_assembly<VE, FE, EXISTS>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class VE, class FE, bool EXISTS>
    inline auto xoptional_assembly<VE, FE, EXISTS>::value_impl() noexcept -> value_expression&
    {
        return m_value;
    }

    template <class VE, class FE, bool EXISTS>
    inline auto xoptional_assembly<VE, FE, EXISTS>::value_impl() const noexcept -> const value_expression&
    {
        return m_value;
    }

    template <class VE, class FE, bool EXISTS>
    inline auto xoptional_assembly<VE, FE, EXISTS>::flag_impl() noexcept -> flag_expression&
    {
        return m_flag;
    }

    template <class VE, class FE, bool EXISTS>
    inline auto xoptional_assembly<VE, FE, EXISTS>::flag_impl() const noexcept -> const flag_expression&
    {
        return m_flag;
    }

    /*********************************************
     * xoptional_assembly_adaptor implementation *
     *********************************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xoptional_assembly_adaptor of the given value and
     * missing mask expressions.
     * @param ve the expression holding the values
     * @param fe the expression holding the missing mask
     */
    template <class VEC, class FEC, bool EXISTS>
    template <class OVE, class OFE>
    inline xoptional_assembly_adaptor<VEC, FEC, EXISTS>::xoptional_assembly_adaptor(OVE&& ve, OFE&& fe)
        : m_value(std::forward<OVE>(ve)), m_flag(std::forward<OFE>(fe))
    {
    }
    //@}

    template <class VEC, class FEC, bool EXISTS>
    inline auto xoptional_assembly_adaptor<VEC, FEC, EXISTS>::operator=(const self_type& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_value = rhs.m_value;
        m_flag = rhs.m_flag;
        return *this;
    }

    template <class VEC, class FEC, bool EXISTS>
    inline auto xoptional_assembly_adaptor<VEC, FEC, EXISTS>::operator=(self_type&& rhs) -> self_type&
    {
        base_type::operator=(std::move(rhs));
        m_value = rhs.m_value;
        m_flag = rhs.m_flag;
        return *this;
    }

    template <class VEC, class FEC, bool EXISTS>
    inline auto xoptional_assembly_adaptor<VEC, FEC, EXISTS>::operator=(temporary_type&& tmp) -> self_type&
    {
        m_value = std::move(tmp.value());
        m_flag = std::move(tmp.flag());
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class VEC, class FEC, bool EXISTS>
    template <class E>
    inline auto xoptional_assembly_adaptor<VEC, FEC, EXISTS>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class VEC, class FEC, bool EXISTS>
    inline auto xoptional_assembly_adaptor<VEC, FEC, EXISTS>::value_impl() noexcept -> value_expression&
    {
        return m_value;
    }

    template <class VEC, class FEC, bool EXISTS>
    inline auto xoptional_assembly_adaptor<VEC, FEC, EXISTS>::value_impl() const noexcept -> const value_expression&
    {
        return m_value;
    }

    template <class VEC, class FEC, bool EXISTS>
    inline auto xoptional_assembly_adaptor<VEC, FEC, EXISTS>::flag_impl() noexcept -> flag_expression&
    {
        return m_flag;
    }

    template <class VEC, class FEC, bool EXISTS>
    inline auto xoptional_assembly_adaptor<VEC, FEC, EXISTS>::flag_impl() const noexcept -> const flag_expression&
    {
        return m_flag;
    }
}

#endif
