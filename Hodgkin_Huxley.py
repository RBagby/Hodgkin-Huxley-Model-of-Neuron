import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class HodgkinHuxley(object):

'''
Hodgkin-Huxley Model


	Parameters:

		C_m   : membrane capacitance in uF/cm**2.

		g_Na  : maximum sodium (Na) conductances in mS/cm**2.
		g_K   : maximum potassium (K) conductances in mS/cm**2.
		g_L   : maximum leak conductances in mS/cm**2.

		E_Na  : sodium (Na) Nernst reversal potentials in mV.
		E_K   : potassium (K) Nernst reversal potentials in mV.
		E_L   : leak Nernst reversal potentials in mV.

	Equations:

		# Voltage-Current relationship
		C_m(dV( t)/dt) = -sum([I( t,V) for i in range( time)])

		# Ohm's Law
		I(t,V) = g(t,V) * (V - V_eq)

		# Conductance
		g(t,V) = g_bar * m( t,V)**p * h( t,V)**q

		# Fractions ( m or h, but used m for example)
		dm(t,V)/dt = alpha_m( V)*( 1 - m) - beta_m( V)*m, where alpha & beta define gate fractions.
'''


	# Parameters.
	C_m = 1.0

	g_Na = 120.0
	g_K = 36.0
	g_L = 0.3

	E_Na = 50.0 
	E_K = -77.0
	E_L = -54.387


	# Time interval to integrate over.
	t = np.arange(0.0, 450.0, 0.01)

	# Equations.
	# Channel Gating Kinetics.
	def alpha_m(self, V):
		alpha = 0.1*(V + 40.0)/(1.0 - np.exp( -(V+40.0)/10.0))

		return alpha

	def beta_m(self, V):
		beta = 4.0*np.exp(-(V+65.0) / 18.0)
		
		return beta

	def alpha_h(self, V):
		alpha = 0.07*np.exp(-(V+65.0) / 20.0)

		return alpha

	def beta_h(self, V):
		beta = 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

		return beta

	def alpha_n(self, V):
		alpha = 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

		return alpha

	def beta_n(self, V):

		return 0.125*np.exp(-(V+65) / 80.0)


	# Membrane Currents.
	def I_Na(self, V, m, h):

		return self.g_Na * m**3 * h * (V - self.E_Na)

	def I_K(self, V, n):
		I = self.g_K * n**4 * (V - self.E_K)

		return I

	# Leak
	def I_L(self, V):
		I = self.g_L * (V - self.E_L)

		return I

	# External Current.
	def I_inj(self, t):
		I = 10*(t>100) - 10*(t>200) + 35*(t>300)

		return I


	def dALLdt(self, X, t):

		V, m, h, n = X

		# Calculate membrane potential & activation variables.
		dV_dt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
		dm_dt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
		dh_dt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
		dn_dt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n

		return dV_dt, dm_dt, dh_dt, dn_dt


	def main(self):
		# Demo for model neuron.
		X = odeint(self.dALLdt, [-65.0, 0.05, 0.6, 0.32], self.t)

		V, m, h, n = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

		i_na = self.I_Na(V, m, h)
		i_k  = self.I_K(V, n)
		i_l  = self.I_L(V)

		plt.figure()

		plt.subplot(4,1,1)
		plt.title('Hodgkin-Huxley Neuron')
		plt.plot(self.t, V, 'k')
		plt.ylabel('V (mV)')

		plt.subplot(4,1,2)
		plt.plot(self.t, i_na, 'c', label='$I_{Na}$')
		plt.plot(self.t, i_k, 'y', label='$I_{K}$')
		plt.plot(self.t, i_l, 'm', label='$I_{L}$')
		plt.ylabel('Current')
		plt.legend()

		plt.subplot(4,1,3)
		plt.plot(self.t, m, 'r', label='m')
		plt.plot(self.t, h, 'g', label='h')
		plt.plot(self.t, n, 'b', label='n')
		plt.ylabel('Gating Value')
		plt.legend()

		plt.subplot(4,1,4)
		i_inj_values = [self.I_inj(t) for t in self.t]
		plt.xlabel('t (ms)')
		plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
		plt.ylim(-1, 40)

		plt.show()


if __name__ == '__main__':
	runner = HodgkinHuxley()
	runner.main()










