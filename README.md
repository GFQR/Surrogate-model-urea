# SURROGATE MODEL FOR UREA FIRST HYPERPOLARIZABILITY

I model the hyperpolarizability β (nonlinear optical response) of urea subjected to specific charge distributions (dipolar, quadrupolar, octupolar), charge values, and angle between molecule and charge distribution.

Features are extracted from Dalton quantum chemistry outputs using custom bash and SQL pipelines.

The regression attempts to approximate the normalized β (with respect to gas, or no charges) from structural + field-type features.

## OBJECTIVES

- Fit smoothly the hyperpolarizability β within the region of parameter space that was calculated using discrete Dalton experiments.
- Find the maximum value of β

## REGRESSION

The regression is done using harmonics on the angle $\theta$ and polynomials on the charge $q$:

$$
\beta(q,\theta)
=
a_0
+
a_1 q
+
a_2 q^2
+
\sum_{n=1}^{N}
\left[
(b_n + c_n q + d_n q^2)\cos(n\theta)
+
(e_n + f_n q + g_n q^2)\sin(n\theta)
\right]
$$
