from pylab import *
import current as pc

rc("figure", figsize=[8.,4.])
rc("figure.subplot", left=.08, top=.95, right=.98)
seed(1000)

dist = pc.Laplace(pi, .5)
f = pc.lazy_eval(dist.pdf)
#end

q0, w0 = pc.quadgen(2, [-10,10])
print q0
#  [[-10.   0.  10.]]
#end

comp = [-10, 0, 10]
q, w = pc.quadgen(2, [-10,10], composite=comp)
print q
#  [[-10.  -5.   0.   5.  10.]]
#end

def eval_errors(q, f):

    errors = [0.]
    for i in xrange(1, len(q[0])-1):

        a,b,c = q[0,i-1:i+2]
        f_approx = ((f(c)-f(a))*b + f(a)*c-f(c)*a)/(c-a)

        f_diff = abs(f(q[0,i]) - f_approx)
        q_diff = q[0,i+1]-q[0,i-1]

        errors.append(f_diff*q_diff)
    errors.append(0.)

    return errors
#end

def add_composite(i, q, comp):

    if q[0,i] in comp:
        if q[0,i]-q[0,i-1]>q[0,i+1]-q[0,i]:
            comp.append(q[0,i-1])
        else:
            comp.append(q[0,i+1])
    else:
        comp.append(q[0,i])

    comp.sort()
#end

tol = 1e-2
error = inf
while error>=tol:

    q, w = pc.quadgen(2, [-10,10], composite=comp)
    errors = eval_errors(q, f)
    add_composite(argmax(errors), q, comp)
    error = mean(errors)
#end

subplot(121)
t = linspace(-10,10,500)
plot(t, f(t), "k")
xlabel("$q$")
ylabel("Standard Gaussian Density")

plot(q[0], [f(_) for _ in q[0]], "ko", alpha=.7)

tol = 1e-6
error = inf
errorsa,errorsb, errorsc = [], [], []
while error>=tol:

    q, w = pc.quadgen(2, [-10,10], composite=comp)
    errors = eval_errors(q, f)
    add_composite(argmax(errors), q, comp)
    error = np.mean(errors)

    errorsa.append(error)
    y = [f(_) for _ in q[0]]
    errorsb.append(abs(np.sum(w*y)-1))

for i in range(9):
    q, w = pc.quadgen(2**len(errorsc), [-10,10])
#      q, w = pc.quadgen(2, [-10,10], composite=len(errorsc))
    y = [f(_) for _ in q[0]]
    errorsc.append(abs(np.sum(w*y)-1))

subplot(122)
#  semilogy(range(len(errorsb)), errorsb, "k-",
#          label=r"Adaptive approx")
semilogy(2**np.arange(9), errorsc, "k-",
        label=r"Non-adaptive")
semilogy(range(len(errorsa)), errorsa, "k--",
        label=r"Adaptive")
legend(loc="upper right")
xlim(0,200)
xlabel("Number iterations")
ylabel("Absolute error")


savefig("adaptive_quad.pdf" ); clf()
