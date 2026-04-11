TITLE N-type Calcium channel

NEURON {
	SUFFIX CaN
	USEION ca READ cai, cao WRITE ica
	RANGE gmax, ica, g
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S)  = (siemens)
}

PARAMETER {
	gmax    = .01 (mho/cm2)
	mtau	= 4	(ms)
	htau	= 40	(ms)
	R = 8.31441 (VC/Mol/K)
	T = 309.15 (k) 
	Z = 2
	F = 96485.309 (/C)
}

ASSIGNED {
	v 		(mV)
	eca 	(mv)
	ica		(mA/cm2)
	g 		(S/cm2)
	minf hinf
	cai cao	(mM)
        celsius (degC)
}

STATE {
	m h
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	eca = ((1000*R*T)/(Z*F))*log(cao/cai)
	g = gmax * m * m * h
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
        KTF = ((36./293.15)*(celsius + 273.15))
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
    h = hinf
}

DERIVATIVE states {
	rates(v)
	m' = (minf - m) / mtau
	h' = (hinf - h) / htau
}

PROCEDURE rates(v(mV)) {
	minf = 1 / (1 + exp((v+30)/-5))
	hinf = 1 / (1 + exp((v+45)/5))
}



