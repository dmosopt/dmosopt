TITLE Delayed rectifier potassium channel

NEURON {
	SUFFIX Kdr
	USEION k READ ek WRITE ik
	RANGE gmax, ik, g
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S)  = (siemens)
}

PARAMETER {
	gmax 	= .8 	(mho/cm2)
}

ASSIGNED {
	v 		(mV)
	ek          (mV)
	ik		(mA/cm2)
	g (S/cm2)
	ninf ntau
}

STATE {
	n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gmax * n * n * n * n
	ik = g * (v - ek)
}

INITIAL {
    rates(v)
    n = ninf
}

DERIVATIVE states {
	rates(v)
	n' = (ninf - n) / ntau
}

PROCEDURE rates(v(mV)) {
	ntau = 7 / (exp((v + 40)/40) + exp(-(v + 40)/50))
	ninf = 1 / ((exp(-(v + 28)/15)) + 1)
}




