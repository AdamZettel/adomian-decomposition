import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbols
x, z, _lambda = sp.symbols('x z _lambda')
order = 5

# Given equation parameters
P = sp.Rational(-1, 3)  # P(x) = 1/3
Q = sp.Rational(1, 6) * x  # Q(x) = x/6
y0 = -2  # Initial condition
#mod = 0.0
v = sp.symbols(f'v0:{order}')  # Define v_0, v_1, ..., v_(order-1) as symbols

# Define the nonlinear function
def f(y):
    return y**4  # The nonlinearity in the Bernoulli equation

# Compute Adomian polynomials
def adomian_polynomials(order=8):
    """Compute Adomian polynomials up to the given order."""
    f_expanded = f(sum(v[i] * (_lambda**i) for i in range(order)))
    
    A = []
    for i in range(order):
        Ai = sp.diff(f_expanded, _lambda, i).subs(_lambda, 0) / sp.factorial(i)
        A.append(Ai.simplify())
    
    return A, v

# Adomian decomposition iteration
def bernoulli_adomian_iteration(P, Q, y0, order=8):
    """Solve the Bernoulli equation using the Adomian Decomposition Method."""
    A, v = adomian_polynomials(order)
    print("Adomian Polynomials")
    for a in A:
        print(sp.latex(a))
    terms = []
    
    # First term (integral of Q * A_0)
    v0_x = y0 
    vk_x = v0_x
    terms.append(vk_x)

    for i in range(0, order):
        vk_plus1_x = sp.integrate(Q * A[i] - P * vk_x, (x, 0, z))
        vk_x = vk_plus1_x.simplify()
        terms.append(vk_x)

    return terms

# Compute terms of the Adomian series
adomian_terms = bernoulli_adomian_iteration(P, Q, y0, order=5)
print("ADOMIAN TERMS")
#for idx, at in enumerate(adomian_terms):
#    print(idx, ": ", at)
subst = []
#subst.append(y0)
for idx, val in enumerate(adomian_terms):
    for j in range(idx):
        adomian_terms[idx] = adomian_terms[idx].subs(v[j], subst[j])
    subst.append(adomian_terms[idx])
    
print("Substitutions")
#print(subst)

# Convert to numerical functions for plotting
adomian_sum = sum(subst).simplify()
adomian_sum = sum(subst)
for s in subst:
    if type(s) is int:
        print(sp.latex(s))
        print()
    else:
        print(sp.latex(s.simplify().subs(z, x)))
        print()

print("Full Summation")
print(sp.latex(adomian_sum.simplify().subs(z, x)))
f_adomian = sp.lambdify(z, adomian_sum, 'numpy') 


# Analytical solution
def analytical_solution(x):
    return -2 / (4*x - 4 + 5*np.exp(-x))**(0.33)

# Plotting


def set_publication_style1():
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'axes.titlepad': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'axes.linewidth': 1.2,
        'xtick.major.size': 6,
        'xtick.major.width': 1.2,
        'ytick.major.size': 6,
        'ytick.major.width': 1.2,
        'figure.figsize': (6, 4),
        'savefig.dpi': 600,
        'savefig.format': 'png'
    })


set_publication_style1()
x_vals = np.linspace(0, 2, 200)
y_analytical = analytical_solution(x_vals)
y_adomian = f_adomian(x_vals)

x_vals_table = np.linspace(0, 0.6, 10)
y_analytical_table = analytical_solution(x_vals_table)
y_adomian_table = f_adomian(x_vals_table)
abs_diff = np.absolute(y_analytical_table - y_adomian_table)
print(x_vals_table)
print(abs_diff)


plt.plot(x_vals, y_adomian, label="Adomian Approximation", linestyle='solid')
plt.plot(x_vals, y_analytical, label="Analytical Solution", linestyle='dashed')
plt.ylim(-4, 1)
plt.xlim(0, 0.8)

plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.title("Comparison of Analytical and Adomian Decomposition Method")
plt.savefig("bernoulli_adomian.png", transparent=True, bbox_inches='tight')


plt.show()
