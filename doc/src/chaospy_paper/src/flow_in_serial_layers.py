from Heaviside import PiecewiseConstant, IntegratedPiecewiseConstant
import numpy as np
from sets import Set
import current as pc
import matplotlib.pyplot as plt
from scipy.misc import comb
import os

plt.rc("figure", figsize=(6,4))
plt.rc("figure.subplot", hspace=.05, top=.9)

class SerialLayers:
    """
    b: coordinates of boundaries of layers, b[0] is left boundary
    and b[-1] is right boundary of the domain [0,L].
    a: values of the functions in each layer (len(a) = len(b)-1).
    U_0: u(x) value at left boundary x=0=b[0].
    U_L: u(x) value at right boundary x=L=b[0].
    """

    def __init__(self, a, b, U_0, U_L, eps=0):
        self.a, self.b = np.asarray(a), np.asarray(b)
        assert len(a) == len(b)-1, 'a and b do not have compatible lengths'
        self.eps = eps  # smoothing parameter for smoothed a
        self.U_0, self.U_L = U_0, U_L

        # Smoothing parameter must be less than half the smallest material
        if eps > 0:
            assert eps < 0.5*(self.b[1:] - self.b[:-1]).min(), 'too large eps'

        a_data = [[bi, ai] for bi, ai in zip(self.b, self.a)]
        domain = [b[0], b[-1]]
        self.a_func = PiecewiseConstant(domain, a_data, eps)

        # inv_a = 1/a is needed in formulas
        inv_a_data = [[bi, 1./ai] for bi, ai in zip(self.b, self.a)]
        self.inv_a_func = PiecewiseConstant(domain, inv_a_data, eps)
        self.integral_of_inv_a_func = \
             IntegratedPiecewiseConstant(domain, inv_a_data, eps)
        # Denominator in the exact formula is constant
        self.inv_a_0L = self.integral_of_inv_a_func(b[-1])

    def exact_solution(self, x):
        solution = self.U_0 + (self.U_L-self.U_0)*\
                   self.integral_of_inv_a_func(x)/self.inv_a_0L
        return solution

    __call__ = exact_solution

    def plot(self):
        x, y_a = self.a_func.plot()
        x = np.asarray(x); y_a = np.asarray(y_a)
        y_u = self.exact_solution(x)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x, y_u, 'b')
        plt.hold('on')  # Matlab style
        plt.plot(x, y_a, 'r')
        ymin = -0.1
        ymax = 1.2*max(y_u.max(), y_a.max())
        plt.axis([x[0], x[-1], ymin, ymax])
        plt.legend(['solution $u$', 'coefficient $a$'], loc='upper left')
        if self.eps > 0:
            plt.title('Smoothing eps: %s' % self.eps)
        plt.savefig('tmp.pdf')
        plt.savefig('tmp.png')
        plt.show()

def simulate_case1(
    mesh_resolution_factor=1.,
    uncertainty_type=Set(['material parameters',
                          'internal boundary 1',
                          'u at 0',
                          'u at L']),
    plot=True):  # include plot in simulator (ok when playing)
    """
    Return simulator that takes just the uncertain parameters
    as "unknowns" (i.e., works in parameter space), but uses
    other prescribed values for all remaining input.
    """

    def quantity_of_interest(solution, mesh):
        """Return gradient in last material."""
        return (solution[-1] - solution[-2])/(mesh[-1] - mesh[-2])

    def simulator(*args):
        """
        Function to be called from UQ libraries.
        `args` holds the uncertain parameters only.
        """
        # Note: args has different storing conventions for different
        # types of uncertainty parameters, see below.

        # *Initial all data needed for the simulation*

        # Should use a numerical mesh that has the boundaries of the
        # layers as mesh points. Therefore, the mesh is made first.
        L = 5
        n = 10 # no of mesh intervals with mesh_resolution_factor=1
        i = mesh_resolution_factor*n    # total no of intervals
        mesh = np.linspace(0, L, i+1)
        # Material boundaries matches mesh (as long as n=10)
        b = [0, 0.25, 0.5, 1]
        # Material values
        a = [0.2, 0.4, 4]
        # Boundary conditions
        U_0 = 0.5
        U_L = 5 # should approx match a for a nice plot...

        # Override with data from *args

        # Use Set to avoid a particular sequence of the elements
        # in the uncertainty_type set
        if uncertainty_type == Set(['material parameters']):
            a = np.asarray(*args)
        elif uncertainty_type == Set(['internal boundary 1']):
            b[1] = args[0]
        elif uncertainty_type == Set(['material parameters',
                                      'internal boundaries']):
            a = np.asarray(*args)[:-1]
            b[1] = args[-1]
        elif uncertainty_type == Set(['internal boundary 1', 'u at L']):
            b[1] = args[0]
            U_L = args[1]
        # else: no redefinition of data because of args

        # Create exact solution
        #eps=0.05
        eps = 0
        u = SerialLayers(a, b, U_0, U_L, eps=eps)
        if plot:
            u.plot()
        solution = u(mesh)
        q = quantity_of_interest(solution, mesh)
        return q
    return simulator


#  ## Distribution of a
#  X,Y,Z = pc.Uniform(1,2), pc.Uniform(8,10), pc.Uniform(.3,.7)
#  dist = pc.J(X,Y,Z)
#  
#  class aD(pc.Dist):
#  
#      def __init__(self, t):
#          pc.Dist.__init__(self, t=t)
#  
#      def _mom(self, K, t):
#          return X.mom(K)*(1-Z.cdf(t)) + Y.mom(K)*Z.cdf(t)
#  
#      def _rnd(self, s, t):
#          rx,ry,rz = X.rnd(s),Y.rnd(s),Z.rnd(s)
#          return rx*(rz>t) + ry*(rz<=t)
#  
#      def _bnd(self, s, t):
#          x_,y_ = X.bnd(s), Y.bnd(s)
#          return np.min([x_[0],y_[0]], 0), np.max(x_[1],y_[1], 0)
#  
#  
#  ## Distribution of A
#  class AD(pc.Dist):
#  
#      def __init__(self, t):
#          pc.Dist.__init__(self, t=t)
#  
#      def _rnd(self, s, t):
#          rx,ry,rz = X.rnd(s),Y.rnd(s),Z.rnd(s)
#          return rx*t*(t<rz) + (rx*rz + ry*t -ry*rz)*(t>=rz)
#  
#      def _mom(self, K, t):
#          if np.array(K).shape:
#              return np.array([self._mom(K[i], t) \
#                      for i in range(len(K))])
#  
#          out = X.mom(K)*t**K*(1-Z.cdf(t))
#          if t>.3:
#              t_ = min(t,.7)
#              for j in xrange(K+1):
#                  for i in xrange(j+1):
#                      _ = comb(K,j)*comb(j,i)
#                      _ *= X.mom(i)*Y.mom(K-i)
#                      _ *= (-1)**(j-i)*t**(K-j)
#                      _ *= 2.5*(t_**(j+1)-.3**(j+1))/(j+1)
#                      out += _
#          return out
#  
#      def _bnd(self, s, t):
#          data = self._rnd(10**2, t)
#          return np.min(data), np.max(data)
#  
#  
#  ## Distribution of X,Y,Z,AD
#  class XYZA(pc.Dist):
#  
#      def __init__(self, t):
#          pc.Dist.__init__(self, t=t)
#      def __len__(self):
#          return 4
#  
#      def _rnd(self, s, t):
#          rx,ry,rz = X.rnd(s),Y.rnd(s),Z.rnd(s)
#          return rx, ry, rz, rx*t*(t<rz) + (rx*rz + ry*t -ry*rz)*(t>=rz)
#  
#      def _mom(self, K, t):
#          if np.array(K[0]).shape:
#              out = np.array([self._mom(K[:,i], t) \
#                      for i in range(len(K[0]))])
#              return out
#  
#  
#          out = 0.
#          if t<.7:
#              out = dist.mom((K[-1]+K[0], K[1], 0))*t**K[-1]
#              out *= 2.5*(.7**(K[2]+1)-max(t,.3)**(K[2]+1))/(K[2]+1)
#          if t>.3:
#              t_ = min(t,.7)
#              for j in xrange(K[-1]+1):
#                  for i in xrange(j+1):
#                      _ = comb(K[-1],j)*comb(j,i)
#                      _ *= dist.mom((i+K[0], K[-1]+K[1]-i, 0))
#                      _ *= (-1)**(j-i)*t_**(K[-1]-j)
#                      _ *= 2.5*(t_**(j+K[2]+1)-.3**(j+K[2]+1))/(j+K[2]+1)
#                      out += _
#          return out
#  
#      def _bnd(self, s, t):
#          out = np.zeros((2,4,s))
#          data = self._rnd(10**2, t)
#          out[:,-1] = np.min(data), np.max(data)
#          out[:,:-1] = dist.bnd(s)
#  
#  class XYZB(pc.Dist):
#  
#      def __init__(self, t):
#          pc.Dist.__init__(self, t=t)
#  
#      def __len__(self):
#          return 4
#  
#      def _rnd(self, s, t):
#          rx,ry,rz = X.rnd(s),Y.rnd(s),Z.rnd(s)
#          return rx, ry, rz, t/rx*(t<rz) + (rz/rx + t/ry -rz/ry)*(t>=rz)
#  
#      def _mom(self, K, t):
#          if np.array(K[0]).shape:
#              out = np.array([self._mom(K[:,i], t) \
#                      for i in range(len(K[0]))])
#              return out
#  
#  
#          out = 0.
#          if t<.7:
#              if K[0]-K[3]+1:
#                  out = (2.**(K[0]-K[3]+1)-1)/(K[0]-K[3]+1.)
#              else:
#                  out = np.log(2)
#              out *= .5*(10.**(K[1]+1)-8.**(K[1]+1.))/(K[1]+1)
#              out *= t**K[3]
#              out *= 2.5*(.7**(K[2]+1)-max(t,.3)**(K[2]+1))/(K[2]+1.)
#          if t>.3:
#              t_ = min(t,.7)
#              for i in xrange(K[-1]+1):
#                  for j in xrange(i+1):
#                      _ = comb(K[-1],i)*comb(i,j)
#                      if K[0]+i-K[3]+1:
#                          _ *= (2.**(K[0]+i-K[3]+1)-1.)/(K[0]+i-K[3]+1)
#                      else:
#                          _ *= np.log(2)
#                      if K[1]-i+1:
#                          _ *= .5*(10.**(K[1]-i+1)-8.**(K[1]-i+1))/(K[1]-i+1)
#                      else:
#                          _ *= .5*np.log(1.25)
#                      _ *= (-1)**j*t**(i-j)
#                      _ *= 2.5*(t_**(K[2]+K[3]-i+j+1)-.3**(K[2]+K[3]-i+j+1))/\
#                              (K[2]+K[3]-i+j+1)
#                      out += _
#          return out
#  
#      def _bnd(self, s, t):
#          out = np.zeros((2,4,s))
#          data = self._rnd(10**2, t)
#          out[:,-1] = np.min(data), np.max(data)
#          out[:,:-1] = dist.bnd(s)
#  
#  
#  class XYA(pc.Dist):
#  
#      def __init__(self, t):
#          pc.Dist.__init__(self, t=t)
#      def __len__(self):
#          return 3
#  
#      def _rnd(self, s, t):
#          rx,ry,rz = X.rnd(s),Y.rnd(s),Z.rnd(s)
#          return rx, ry, rx*t*(t<rz) + (rx*rz + ry*t -ry*rz)*(t>=rz)
#  
#      def _mom(self, K, t):
#          if np.array(K[0]).shape:
#              out = np.array([self._mom(K[:,i], t) \
#                      for i in range(len(K[0]))])
#              return out
#  
#          out = dist.mom((K[-1]+K[0], K[1], 0))*t**K[-1]
#          if t<.7:
#              out *= 2.5*(.7-max(t,.3))
#          if t>.3:
#              t_ = min(t,.7)
#              for j in xrange(K[-1]+1):
#                  for i in xrange(j+1):
#                      _ = comb(K[-1],j)*comb(j,i)
#                      _ *= dist.mom((i+K[0], K[-1]+K[1]-i, 0))
#                      _ *= (-1)**(j-i)*t**(K[-1]-j)
#                      _ *= 2.5*(t_**(j+1)-.3**(j+1))/(j+1)
#                      out += _
#          return out
#  
#      def _bnd(self, s, t):
#          out = np.zeros((2,3,s))
#          data = self._rnd(10**2, t)
#          out[:,-1] = np.min(data), np.max(data)
#          out[:,:-1] = dist.bnd(s)[:,:-1]
#  
#  class B(pc.Dist):
#  
#      def __init__(self, t):
#          dist = XYZB(t)
#          pc.Dist.__init__(self, _B=dist, t=t)
#  
#      def _rnd(self, s, t, _B):
#          out = np.array(_B._rnd(s, t))[-1]
#          return out
#  
#      def _mom(self, k, t, _B):
#          zero = np.zeros((3, k.shape[-1]), dtype=int)
#          k = np.concatenate([zero, k], 0)
#          return _B._mom(k, t)
#  
#      def _bnd(self, s, t, _B):
#          return _B._bnd(s, t)[-1]
#  
#  class BC(pc.Dist):
#  
#      def __init__(self, t):
#          pc.Dist.__init__(self, t=t)
#  
#      def __len__(self):
#          return 2
#  
#      def _rnd(self, s, t):
#          rx,ry,rz = X.rnd(s),Y.rnd(s),Z.rnd(s)
#          return t/rx*(t<rz) + (rz/rx + t/ry -rz/ry)*(t>=rz), \
#                  rz/rx+(1-rz)/ry
#  
#      def _mom(self, k, t):
#          if np.array(k[0]).shape:
#              out = np.array([self._mom(k[:,i], t) \
#                      for i in range(len(k[0]))])
#              return out
#  
#          t_lo = min(t, .7)
#          t_up = max(t, .3)
#  
#          out = 0.
#          if t<.7:
#              for i in xrange(k[1]+1):
#                  for j in xrange(i+1):
#                      _ = comb(k[1],i)*comb(i,j)
#                      _ *= (-1)**j * t**k[0]
#  
#                      if i-k[0]-k[1]+1:
#                          _ *= (2.**(i-k[0]-k[1]+1)-1)/(i-k[0]-k[1]+1)
#                      else:
#                          _ *= np.log(2)
#  
#                      if -i+1:
#                          _ *= .5*(10.**(-i+1)-8.**(-i+1))/(-i+1)
#                      else:
#                          _ *= .5*np.log(1.25)
#  
#                      _ *= 2.5*(.7**(j+k[1]-i+1)-t_up**(j+k[1]-i+1))/\
#                              (j+k[1]-i+1.)
#                      out += _
#  
#          if t>.3:
#           for i1 in xrange(k[0]+1):
#            for i2 in xrange(i1+1):
#             for j1 in xrange(k[1]+1):
#              for j2 in xrange(j1+1):
#  
#                  _ = comb(k[0], i1)*comb(i1, i2)
#                  _ *= comb(k[1], j1)*comb(j1, j2)
#                  _ *= t**i2*(-1)**(i1-i2+j1-j2)
#  
#                  m = i1-k[0]+j1-k[1]+1.
#                  if m:
#                      _ *= (2.**m-1)/m
#                  else:
#                      _ *= np.log(2)
#  
#                  m = -i1-j1+1.
#                  if m:
#                      _ *= .5*(10.**m-8.**m)/m
#                  else:
#                      _ *= .5*np.log(1.25)
#  
#                  m = k[0]-i2+k[1]-j2+1.
#                  _ *= 2.5*(t_lo**m-.3**m)/m
#  
#                  out += _
#  
#          return out
#  
#  class XYB(pc.Dist):
#  
#      def __init__(self, t):
#          dist = XYZB(t)
#          pc.Dist.__init__(self, dist=dist, t=t)
#  
#      def __len__(self):
#          return 3
#  
#      def _rnd(self, s, t, dist):
#          out = np.empty((3,s))
#          out[:2] = dist[:2]
#          out[-1] = dist[-1]
#          return out
#  
#      def _mom(self, k, t, dist):
#          K = np.zeros((4,k.shape[-1]), dtype=int)
#          K[:2] = k[:2]
#          K[-1] = k[-1]
#          return dist._mom_func(K)
#  
#  class XYZE(pc.Dist):
#  
#      def __init__(self, t):
#          pc.Dist.__init__(self, t=t)
#  
#      def __len__(self):
#          return 4
#  
#      def _rnd(self, s, t):
#          x,y,z = dist.rnd(s)
#          e = z*(z<=t)
#          return x,y,z,e
#  
#      def _mom(self, k, t):
#          t += (.3-t)*(t<.3) + (.7-t)*(t>.7)
#          return 2.5*(t**(k[-1]+1)-.3**(k[-1]+1))/\
#                  (k[-1]+1.)*dist.mom(k[:-1])
#  
#  # Transform
#  def a_trans(x, t):
#      return np.array([x[0]*(x[2]>_) + x[1]*(x[2]<=_)\
#              for _ in t])
#  
#  def A_trans(x, t):
#      return np.array([x[0]*_*(_<x[2]) + (x[0]*x[2] + x[1]*_\
#          -x[1]*x[2])*(_>=x[2]) for _ in t])
#  
#  def B_trans(x, t):
#      return np.array([_/x[0]*(_<x[2]) + (x[2]/x[0] + _/x[1]\
#          -x[2]/x[1])*(_>=x[2]) for _ in t])
#  
#  
#  
#  if 0:
#      t = np.linspace(0,1,200)
#  
#      def solver(r,t):
#          b = [0, r[2], 1]
#          a = [r[0], r[1]]
#          U_0, U_L = 0.5, 5
#          sl = SerialLayers(a,b,U_0,U_L)
#          return sl(t)
#  
#      for i in range(3):
#          r = dist.rnd()
#          y1 = solver(r,t)
#          plt.plot(t, y1, "k-", label=r"$u(z;\xi)$")
#  
#      plt.xlabel("Depth $z$")
#      plt.ylabel("Pressure $u$")
#      plt.savefig("u_ex.pdf"); plt.clf()
#  
#      for i in range(3):
#          r = dist.rnd()
#          y1 = solver(r,t)
#          y2 = A_trans(r, t)
#          plt.plot(t, y1, "k-", label=r"$u(x;q)$")
#          plt.plot(t, y2, "k--", label=r"$\int_0^x a(s,q)\,ds$")
#          if not i:
#              plt.legend(loc="upper left")
#  
#      plt.xlabel("Depth $x$")
#      plt.ylabel("Pressure $u$")
#      plt.savefig("uinta.pdf")
#      fail
#  
#  
#  # Global variables
#  t = np.linspace(0,1,100)[1:-1]
#  def solver(r):
#          b = [0, r[2], 1]
#          a = [r[0], r[1]]
#          U_0, U_L = 0.5, 5
#          sl = SerialLayers(a,b,U_0,U_L)
#          return sl(t)
#  
#  # MC results
#  if os.path.isfile("data0"):
#      E0,V0 = np.loadtxt("data0")
#      N = np.loadtxt("N0").item()
#  else:
#      data = np.array([solver(_) for _ in dist.rnd(10**5).T])
#      E0,V0 = data = np.mean(data, 0), np.mean(data**2, 0)
#      np.savetxt("data0", data)
#      N = 1
#      np.savetxt("N0", [N])
#  
#  for i in range(0):
#      data = np.array([solver(_) for _ in dist.rnd(10**5).T])
#      print "run", i
#      _ = np.mean(data, 0)/(N+1.) + E0*N/(N+1.)
#      print np.sum((_-E0)**2)
#      E0 = _
#      V0 = np.mean(data**2, 0)/(N+1.) + V0*N/(N+1.)
#      N = N+1
#      np.savetxt("data0", (E0,V0))
#      np.savetxt("N0", [N])
#  
#  V0 = V0 - E0*E0
#  
#  x = pc.sampler.hh(500, 3)
#  x = dist.ppf(x.T).T
#  U = np.array(map(solver, x)).T
#  
#  for n in xrange(1,8):
#  
#      print "n", n
#  
#      N = np.zeros(7)
#      E = np.zeros((7, len(t)))
#      V = np.zeros((7, len(t)))
#      R = np.zeros((7, len(t)))
#  
#      for i in xrange(len(t)):
#  
#          def solver(q):
#  
#              b = [0, q[2], 1]
#              a = [q[0], q[1]]
#              U_0 = 0.5
#              U_L = 5
#              sl = SerialLayers(a, b, U_0, U_L)
#              return sl(t[i])
#  
#          R0, x0 = pc.pcm_gq(solver, n, dist, retall=1)
#          N[0] = len(x0)
#          u0 = np.array(map(solver, x0)).T
#  
#          E[0,i] = np.abs(pc.E(R0, dist)-E0[i])
#          V[0,i] = np.abs(pc.Var(R0, dist)-V0[i])
#          R[0,i] = np.sum((u0-R0(*x0.T))**2)
#  
#          orth = pc.orth_ttr(n, dist)
#          N[1] = 2*len(orth)
#          x1 = x[:N[1]]
#          U1 = U[i, :N[1]]
#          R1 = pc.lls_global(orth, x1.T, U1)
#  
#          E[1,i] = np.abs(pc.E(R1, dist)-E0[i])
#          V[1,i] = np.abs(pc.Var(R1, dist)-V0[i])
#          R[1,i] = np.sum((U1-R1(*x1.T))**2)
#  
#          D2 = XYZE(t[i])
#          orth = pc.orth_svd(n, D2)
#          N[2] = 2*len(orth)
#          x_ = A_trans(x.T, [t[i]]).T
#          x2 = np.concatenate([x, x_], 1)[:N[2]]
#          U2 = U[i, :N[2]]
#          R2 = pc.lls_global(orth, x2.T, U2)
#  
#          E[2,i] = np.abs(pc.E(R2, D2)-E0[i])
#          V[2,i] = np.abs(pc.Var(R2, D2)-V0[i])
#          R[2,i] = np.sum((U2-R2(*x2.T))**2)
#  
#          D3 = XYZB(t[i])
#          orth = pc.orth_pcd(n, D3)
#          N[3] = 2*len(orth)
#          x_ = B_trans(x.T, [t[i]]).T
#          x3 = np.concatenate([x, x_], 1)[:N[3]]
#          U3 = U[i, :N[3]]
#          R3 = pc.lls_global(orth, x3.T, U3)
#  
#          E[3,i] = np.abs(pc.E(R3, D3)-E0[i])
#          V[3,i] = np.abs(pc.Var(R3, D3)-V0[i])
#          R[3,i] = np.sum((U3-R3(*x3.T))**2)
#  
#  #          D4 = B(t[i])
#  #          orth = pc.orth_pcd(n, D4)
#  #          N[4] = 2*len(orth)
#  #          x4 = B_trans(x.T, [t[i]])[0,:N[4]]
#  #          U4 = U[i, :N[4]]
#  #          R4 = pc.lls_global(orth, x4.T, U4)
#  #  
#  #          E[4,i] = np.abs(pc.E(R4, D4)-E0[i])
#  #          V[4,i] = np.abs(pc.Var(R4, D4)-V0[i])
#  #          R[4,i] = np.sum((U4-R4(x4))**2)
#  #  
#  #          D5 = BC(t[i])
#  #          orth = pc.orth_pcd(n+1, D5)
#  #          N[5] = 2*len(orth)
#  #          x_1 = B_trans(x.T, [t[i]]).T
#  #          x_2 = B_trans(x.T, [1.]).T
#  #          x5 = np.concatenate([x_1,x_2], 1)[:N[5]]
#  #          U5 = U[i, :N[5]]
#  #          R5 = pc.lls_global(orth, x5.T, U5)
#  #  
#  #          E[5,i] = np.abs(pc.E(R5, D5)-E0[i])
#  #          V[5,i] = np.abs(pc.Var(R5, D5)-V0[i])
#  #          R[5,i] = np.sum((U5-R5(*x5.T))**2)
#  #  
#  #          D6 = XYB(t[i])
#  #          orth = pc.orth_pcd(n, D6)
#  #          N[6] = 2*len(orth)
#  #          x_ = B_trans(x.T, [t[i]]).T
#  #          x6 = np.concatenate([x[:,:2],x_], 1)[:N[6]]
#  #          U6 = U[i, :N[6]]
#  #          R6 = pc.lls_global(orth, x6.T, U6)
#  #  
#  #          E[6,i] = np.abs(pc.E(R6, D6)-E0[i])
#  #          V[6,i] = np.abs(pc.Var(R6, D6)-V0[i])
#  #          R[6,i] = np.sum((U6-R6(*x6.T))**2)
#  
#      print "gq"
#      print repr(N[0])
#      print repr(np.mean(E[0], 0))
#      print repr(np.mean(V[0], 0))
#      print repr(np.mean(R[0], 0))
#      print
#      print "xyz"
#      print repr(N[1])
#      print repr(np.mean(E[1], 0))
#      print repr(np.mean(V[1], 0))
#      print repr(np.mean(R[1], 0))
#      print
#      print "xyzE"
#      print repr(N[2])
#      print repr(np.mean(E[2], 0))
#      print repr(np.mean(V[2], 0))
#      print repr(np.mean(R[2], 0))
#      print
#      print "xyzB"
#      print repr(N[3])
#      print repr(np.mean(E[3], 0))
#      print repr(np.mean(V[3], 0))
#      print repr(np.mean(R[3], 0))
#      print
#  #      print "B"
#  #      print repr(N[4])
#  #      print repr(np.mean(E[4], 0))
#  #      print repr(np.mean(V[4], 0))
#  #      print repr(np.mean(R[4], 0))
#  #      print
#  #      print "BC"
#  #      print repr(N[5])
#  #      print repr(np.mean(E[5], 0))
#  #      print repr(np.mean(V[5], 0))
#  #      print repr(np.mean(R[5], 0))
#  #      print
#  #      print "xyB"
#  #      print repr(N[6])
#  #      print repr(np.mean(E[6], 0))
#  #      print repr(np.mean(V[6], 0))
#  #      print repr(np.mean(R[6], 0))
#  #      print
