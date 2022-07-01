from constrained_sampling import ParamSpacePoints 

if __name__=='__main__':
    SpaceUnc = {
        'gc': [0.01, 50],
        'soma_gnabar': [0.1, 50],
        'soma_gl': [0.001, 0.6],
        'dend_gcabar': [0.01, 10],
        'dend_gkcbar': [0.1, 10],
        'dend_gkahpbar': [0.001, 0.6],
        'dend_gl': [0.001, 0.6],
    }
    
    SpaceCons = {
        "soma_gkdrbar": { 
            'abs': [0.0, 60.0],
            'lb': [('gc', '+ 5'), ('soma_gl', '* 10')], 
            'ub': [('gc', '* 1.5')],
            'method': ('uniform', None, None ),
            },
    
        "soma_gkahpbar":{
            'abs': [0.001, 0.6],
            'method': ('normal', 0, 200 ),
            },
    }

    sampled_points = ParamSpacePoints(10, SpaceUnc)
    print(sampled_points.ParamSample)
