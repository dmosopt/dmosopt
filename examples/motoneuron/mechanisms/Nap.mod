TITLE Persistent Sodium Channel

NEURON {
	SUFFIX Nap
	USEION na READ ena WRITE ina
	RANGE gmax, ina, g, i
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S)  = (siemens)
}

PARAMETER {
	gmax	=0.0008 	(mho/cm2) <0,1e9>
}

ASSIGNED {
	v (mV)
	ena (mv)
	ina (mA/cm2)
	i (mA/cm2)
	g (S/cm2)
	minf mtau
}

STATE {
	m
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gmax * m * m * m
	i = g * (v - ena)
	ina = i
}

INITIAL { :Assume v has been constant for a long time
	rates(v)
	m = minf
}

DERIVATIVE states { :Computes state variable m and h at present v & t
	rates(v)
	m' = (minf - m)/mtau
}

PROCEDURE rates(v(mV)) {LOCAL a, b
	a = (-0.0353*(v+21.4))/(exp(-(v+21.4)/5)-1)
	b = (0.000883*(v+25.7))/(exp((v+25.7)/5)-1)
	mtau = 1/(a + b)
	minf = a/(a + b)
}
