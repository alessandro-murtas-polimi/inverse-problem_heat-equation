#pragma once
#include <deal.II/base/utilities.h>

using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim>
{
public:
    RightHandSide() {}

    virtual double value(const Point<dim>  &p, const unsigned int component = 0) const override;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim>  &/*p*/,
                                 const unsigned int /*component*/) const
{
    return 0;
}