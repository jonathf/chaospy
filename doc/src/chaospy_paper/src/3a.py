import current as pc

import numpy as np

## Model setup ##
#################



for order in xrange(O0, Of):

    print "order", order

    E,V = np.empty((2,len(Z)))
    N = 4*pc.terms(order, 2)
    q = Q.sample(N, "H")
    for i in xrange(len(Z)):

        print "i", i
        z = Z[i]
        solver = Solver(z)
        trans = lambda q: \
            [z/q[0]*(z<q[3]) + \
            (q[3]/q[0]+(z-q[3])/q[1])*(z>=q[3])*(z<q[4]) + \
            (q[3]/q[0]+(q[4]-q[3])/q[1]+(z-q[4])/q[2])*(z>=q[4]),
            q[3]/q[0]+(q[4]-q[3])/q[1]+(1-q[4])/q[2]]
        dist = pc.Dist(_length=2)
        dist._mom = pc.momgen(trans, 15, Q, rule="C",
                composit=[.15,.9,.15, z, z])

        orth = pc.orth_chol(order, dist, normed=0)
        y = np.array(map(solver, q.T))
        approx = pc.fitter_lr(orth, trans(q), y,
                rule="T", order=1, alpha=1e-8)

        E[i] = pc.E(approx, dist)
        V[i] = pc.Var(approx, dist)

    E = np.abs(E-E0)
    V = np.abs(V-V0)
    os.system('echo "%d\n%s\n%s" >> 3a.log' % (N, repr(E), repr(V)))
    os.system('echo "%s, %s" >> 3a.log' % (np.mean(E), np.mean(V)))
