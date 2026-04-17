COMMENT
        Sodium first order kinetics
ENDCOMMENT

NEURON {
        SUFFIX Na_conc
        USEION na READ ina, nao, nai WRITE nai, nao
        RANGE d, beta, nai0, nao0
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
        d = .2           (um)
        nao0 = 145        (mM)         
        nai0 = 15         (mM)         
        beta = 0.075       (/ms)
}

ASSIGNED {
        celsius 	 (degC)
	ina              (mA/cm2)
    }
    
STATE {
	nai (mM)
	nao (mM)
}

INITIAL {
        nai = nai0 
        nao = nao0 
}

BREAKPOINT {
       SOLVE conc METHOD derivimplicit
}

DERIVATIVE conc {    
        nai' =  -(ina)/(2*FARADAY*d)*(1e4) - beta*(nai-nai0)
        nao' =  (ina)/(2*FARADAY*d)*(1e4) - beta*(nao-nao0)
	
}


