COMMENT
        Potassium first order kinetics
ENDCOMMENT

NEURON {
        SUFFIX K_conc
        USEION k READ ik, ki, ko WRITE ki, ko
        RANGE d, beta, ki0, ko0
}

UNITS {
        (mV)    = (millivolt)
        (mA)    = (milliamp)
	(um)    = (micron)
	(molar) = (1/liter)
        (mM)    = (millimolar)
   	FARADAY      = (faraday) (coulomb)
}

PARAMETER {
        d = .2          (um)
        ko0 = 5          (mM)         
        ki0 = 145       (mM)         
        beta = 0.075     (/ms)
}

ASSIGNED {
        celsius 	(degC)
        ik              (mA/cm2)
}

STATE {
	ki (mM)
	ko (mM)

}

INITIAL {
        ki = ki0 
        ko=  ko0
}

BREAKPOINT {
       SOLVE conc METHOD derivimplicit
}

DERIVATIVE conc {
	ki' =  -(ik)/(2*FARADAY*d)*(1e4) - beta*(ki-ki0)
	ko'= (ik)/(2*FARADAY*d)*(1e4) - beta*(ko-ko0)
}


