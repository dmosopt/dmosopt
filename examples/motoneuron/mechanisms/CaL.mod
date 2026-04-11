TITLE L-type Calcium channel

NEURON {
	SUFFIX CaL
	USEION ca READ cai, cao WRITE ica
	RANGE gmax, g, eca, ica
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	gmax     = 0 (mho/cm2)
	mtau	 = 60	    (ms)
        R = 8.31441 (VC/Mol/K)
	T = 309.15 (k) 
	Z = 2
	F = 96485.309 (/C)

}

ASSIGNED {
	ica		(nA)
	v		(mV)
	g		(uS)
	i		(nA)
	minf
	cai cao	(mM)
        celsius (degC)
}

STATE {
	m
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gmax * m
        ica = g*ghk(v, cai, cao)
}

: ghk formalism from cal.mod
FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {
        LOCAL nu,f
        f = KTF(celsius)/2
        nu = v/f
        ghk=-f*(1. - (ci/co)*exp(nu))*efun(nu)
}

FUNCTION KTF(celsius (degC)) (mV) {
        KTF = ((25./293.15)*(celsius + 273.15))
}


FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}

INITIAL {
	rates(v)
	m = minf
}

DERIVATIVE states {
	rates(v)
	m' = (minf - m) / mtau
}

PROCEDURE rates(v(mV)) {
	minf = 1 / (1 + exp((v + 40)/-7))
}