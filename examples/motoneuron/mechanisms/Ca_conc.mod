TITLE Intracellular calcium dynamics

NEURON {
	SUFFIX Ca_conc
	USEION ca READ ica WRITE cai
	RANGE cai, irest, f, kCa, alpha
}

UNITS	{
	(mV) 		= (millivolt)
	(mA) 		= (milliamp)
	FARADAY 	= (faraday) (coulombs)
	(molar) 	= (1/liter)
	(mM) 		= (millimolar)
}

PARAMETER {
	f = 0.004		
	kCa = 8		(/ms)	
	alpha = 1    	(mol/C/cm2)
	cai0 = 1e-5 	(mM)
}

ASSIGNED {
	v			(mV)
	ica			(mA/cm2)
        irest  (mA/cm2)

}

STATE {
	cai (mM) <1e-5>

}

BREAKPOINT {
	SOLVE state METHOD derivimplicit
}

INITIAL {
    cai = cai0
    irest = ica
}

DERIVATIVE state { LOCAL channel_flow
    
    channel_flow = -alpha*(ica - irest)
    if (channel_flow < 0) {
        channel_flow = 0
    }
    
    cai' = f*(channel_flow - kCa * (cai - cai0))
}