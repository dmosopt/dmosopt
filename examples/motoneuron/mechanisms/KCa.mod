TITLE Ca activated potassium channel

NEURON {
	SUFFIX KCa
	USEION ca READ cai
	USEION k READ ek WRITE ik
	RANGE gmax, ik, g
}

UNITS {
	(mV)	=	(millivolt)
	(mA)	=	(milliamp)
	(molar)	=	(1/liter)
	(mM)	=	(millimolar)
	(S)	=	(siemens)
}

PARAMETER {
	gmax = 0.02		(S/cm2)
	Kd		= 0.0005		(mM)
}

ASSIGNED {
	ik			(mA/cm2)
	v			(mV)
	ek			(mV)
	cai			(mM)
	i			(mA/cm2)
	g			(S/cm2)
}

INITIAL {
	g = gmax * (cai/(cai+Kd))
	ik = g * (v - ek)
}

BREAKPOINT {
	g = gmax * (cai/(cai+Kd))
	ik = g * (v - ek)
}


