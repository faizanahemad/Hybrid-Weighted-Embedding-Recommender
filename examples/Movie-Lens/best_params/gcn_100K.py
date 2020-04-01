params = dict(n_dims=128, n_content_dims=196,
                  collaborative_params=dict(
                      prediction_network_params=dict(lr=0.055, epochs=80, batch_size=512,
                                                     network_depth=5,
                                                     gaussian_noise=0.38, conv_depth=4,
                                                     kernel_l2=9e-7, ),
                      user_item_params=dict(gcn_lr=0.0005, gcn_epochs=5, gcn_layers=3,
                                            gcn_kernel_l2=1e-8, gcn_batch_size=512, conv_depth=2,
                                            margin=1.0,
                                            gaussian_noise=0.0025,
                                            node2vec_params=dict(num_walks=150, q=0.75))))

