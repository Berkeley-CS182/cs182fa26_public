from argparse import Namespace

HP = Namespace(
    batch_size=32,
    lr=1e-3,
    momentum=0.9,
    lr_decay=0.99,
    optim_type="adam",
    l2_reg=0.0,
    epochs=5,
    do_batchnorm=False,
    p_dropout=0.0,
)
