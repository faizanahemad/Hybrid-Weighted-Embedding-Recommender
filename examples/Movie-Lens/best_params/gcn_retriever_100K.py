params = dict(n_dims=64, n_content_dims=196,
                  collaborative_params=dict(
                      user_item_params=dict(gcn_lr=0.001, gcn_epochs=3, gcn_layers=1,
                                            gcn_kernel_l2=1e-8, gcn_batch_size=512, conv_depth=4,
                                            margin=1.0,
                                            gaussian_noise=0.005, num_walks=0,
                                            node2vec_params=dict(num_walks=150, q=0.75))))
