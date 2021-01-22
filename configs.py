

class Config(object):
    title_size = 14  # 最长12个字外加<s> </s>
    maxlen = 10  # <s> 七言诗 </s> 共10个字

    bidirectional = True

    # Model Arguments
    max_vocab_cnt = 6000
    emb_size = 300  # size of word embeddings
    n_hidden = 400  # number of hidden units per layer 每层的hidden size
    bow_size = 400
    n_layers = 1  # number of layers
    noise_radius = 0.2  # stdev of noise for autoencoder (regularizer)

    z_size = 400  # dimension of z # 300 performs worse
    full_kl_step = 10000

    init_weight = 0.02   # uniform random from [-init_w, init_w]
    # lambda_gp = 10  # Gradient penalty lambda hyperparameter.  调小 调大,控制penalty的倍数
    # n_d_loss = 1  # 控制discriminator的loss倍数
    # lr_gan_g = 5e-05
    # lr_gan_d = 1e-05
    # n_iters_d = 5
    temp = 1.0  # softmax temperature (lower --> more discrete)

    dropout = 0.3  # dropout applied to layers (0 = no dropout)

    # Training Arguments
    batch_size = 80  # train batch size 80, valid 60, test 1
    epochs = 10  # maximum number of epochs every global iter
    min_epochs = 2  # minimum number of epochs to train for

    lr_ae = 1e-3  # autoencoder learning rate adam
    lr_vae = 1e-3   # autoencoder learning rate adam
    beta1 = 0.9  # beta1 for adam1
    clip = 1.0  # gradient clipping, max norm

    log_every = 50
    valid_every = 200
    test_every = 300

    # Gaussian Mixture Prior Network
    with_sentiment=True
    gumbel_temp = 1
    n_prior_components = 3
    temp_size = 800

    # Model reload
    reload_model = False
    model_name = 'model_global_t_13596_epoch3.pckl'

