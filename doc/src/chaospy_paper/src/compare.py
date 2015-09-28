import matplotlib.pyplot as plt

plt.rc("figure", figsize=[8,4])
plt.rc("legend",  fontsize="small")
plt.rc("figure.subplot", wspace=0.05, right=.99, bottom=.15)

N = [8, 20, 40, 70, 112, 168]

Ed1 = [0.0250178090717, 0.00356590064424, 0.000573816058922, 4.54902140695e-05,
        1.36592220924e-06, 4.51351538359e-07]
Vd1 = [0.0020748265305, 0.000216344465999, 2.01227490927e-05, 8.33331924694e-07,
        1.20736322417e-07, 1.02772079354e-08]

Nd2 = [1, 8, 27, 64, 125, 216]
Ed2 = [ 0.0294034957114, 0.00927094108166, 9.1447489912e-06, 5.59369392819e-08,
        3.92856247533e-10, 1.71539372984e-10, ]
Vd2 = [ 0.0205277443854, 0.000418292139831, 5.31918791728e-06,
        3.12655277516e-08, 1.12502596288e-10, 2.03874413256e-12, ]

Et = [0.0159899831815, 0.00951220745678, 0.000372064632437, 2.62568347698e-05,
        2.9565389999e-06, 3.39279716259e-07]
Vt = [0.000777347864465, 0.000427511736189, 4.20801086207e-05,
        2.11525744699e-06, 3.40893452975e-07, 2.72536011887e-09]

Ec1 = [0.053919527322, 0.00235199432281, 0.00037599552415, 3.69106519987e-05,
        1.38408481416e-06, 1.29001524751e-06]
Vc1 = [0.000459419453807, 0.000105519800305,  1.79490430099e-05,
        1.00937338069e-06, 1.41080185475e-07, 3.73183434739e-09]

Nc2 = [1, 8, 27, 64, 125, 216]

Ec2 = [ 0.0294034958026, 0.000858300187787, 9.14485451592e-06,
        5.57317255193e-08, 3.74102138156e-10, 1.33085188703e-10, ]

Vc2 = [ 0.0205277443995, 0.000440014386408, 5.31920882381e-06,
        3.12398767964e-08, 1.00277630259e-10, 1.56818208364e-11, ]

plt.subplot(121)

plt.semilogy(N, Ed1, "k^-")
plt.semilogy(N, Vd1, "k^--")

plt.semilogy(N, Et, "ko-")
plt.semilogy(N, Vt, "ko--")

plt.semilogy(N, Ec1, "ks-")
plt.semilogy(N, Vc1, "ks--")

plt.axis([0, 150, 1e-11, 0.06])
plt.title("Point collocation method")
plt.ylabel("Estimation error")
plt.xlabel("Number of samples")

plt.subplot(122)

plt.plot([-1], [1], "k-", label="Mean")
plt.plot([-1], [1], "k--", label="Variance")
plt.plot([-1], [1], "k^", label="Dakota")
plt.plot([-1], [1], "ko", label="Turns")
plt.plot([-1], [1], "ks", label="Chaospy")

plt.legend(loc="upper right")

plt.semilogy(Nc2, Ec2, "k^-")
plt.semilogy(Nc2, Vc2, "k^--")

plt.semilogy(Nd2, Ed2, "ks-")
plt.semilogy(Nd2, Vd2, "ks--")

plt.yticks([])
plt.axis([0, 150, 1e-11, 0.06])
plt.title("Pseudo-spectral projection")
plt.xlabel("Number of samples")

plt.savefig("compare.pdf")
