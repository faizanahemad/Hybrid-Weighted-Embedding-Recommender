params = dict(n_dims=64, use_content=True,
              gcn_ncf_params=dict(lr=0.001, gcn_epochs=15, ncf_epochs=15, batch_size=1024,
                                  gcn_layers=2, ncf_layers=2,
                                  ncf_gcn_balance=0.0,
                                  gaussian_noise=0.0, kernel_l2=1e-9,
                                  ns_proportion=1.5, ns_w2v_proportion=1.0, ns_w2v_exponent=0.75))
