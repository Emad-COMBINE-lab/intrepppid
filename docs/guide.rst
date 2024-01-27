User Guide
==========

Training
--------

The easiest way to start training INTREPPPID is to use the :doc:`CLI <cli>`.

An example of running the training loop with the values used in the INTREPPPID manuscript is as follows:

.. code:: bash

    $ python -m intrepppid train e2e_rnn_triplet DATASET.h5 spm.model 3 100 80 --seed 3927704 --vocab_size 250 --trunc_len 1500 --embedding_size 64 --rnn_num_layers 2 --rnn_dropout_rate 0.3 --variational_dropout false --bi_reduce last --workers 4 --embedding_droprate 0.3 --do_rate 0.3 --log_path logs/e2e_rnn_triplet --beta_classifier 2 --use_projection false --optimizer_type ranger21_xx --lr 1e-2

Checkpoints will be saved in a folder ``logs/e2e_rnn_triplet/model_name/chkpt`` and can be used for inference.

Inference
---------

The easiest way to infer using INTREPPPID is through the website `https://PPI.bio <https://ppi.bio>`_. However, you may wish to infer locally using INTREPPID for various reasons, `e.g.`: to infer using your own custom checkpoints.

Preparing Data
^^^^^^^^^^^^^^

To infer using INTREPPPID, you'll have to use the :doc:`API <api>`.

The first step is to get the amino acid sequences you want to infer. This can be as simple as defining a list of sequence pairs:

.. code:: python

    sequence_pairs = [
       ("MANQRLS","MGPLSS"),
       ("MQQNLSS","MPWNLS"),
    ]

You'll need to encode all the sequence, and you'll need to use the same settings that were used during training. Using the same parameters as used in the dataset:

.. code:: python

    from intrepppid.data.ppi_oma import IntrepppidDataset
    import sentencepiece as sp

    trunc_len = 1500
    spp = sp.SentencePieceProcessor(model_file=SPM_FILE)

    encoded_sequence_pairs = []

    for p1, p2 in sequence_pairs:
        x1 = IntrepppidDataset.static_encode(trunc_len, spp, p1)
        x2 = IntrepppidDataset.static_encode(trunc_len, spp, p2)

        # Infer interactions here


Alternatively, you may be interested in loading sequences from an INTREPPPID dataset to do testing. You can use the :py:class:`intrepppid.data.ppi_oma.IntrepppidDataModule`.

.. code:: python

    from intrepppid.data.ppi_oma import IntrepppidDataModule

    batch_size = 80

    data_module = IntrepppidDataModule(
        batch_size = batch_size,
        dataset_path = DATASET_PATH,
        c_type = 3,
        trunc_len = 1500,
        workers = 4,
        vocab_size = 250,
        model_file = SPM_FILE,
        seed = 8675309,
        sos = False,
        eos = False,
        negative_omid = True
    )

    data_module.setup()

    for batch in data_module.test_dataloader():
        p1_seq, p2_seq, _, _, _, label = batch
        # Infer interactions here

Load the INTREPPPID network
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We must now instantiate the INTREPPPID network and load weights.

If you trained the INTREPPPID with the manuscript defaults, you pass any values to :py:func:`intrepppid.intrepppid_network`.

.. code:: python

    from intrepppid import intrepppid_network

    # steps_per_epoch is 0 here because it is not used for inference
    net = intrepppid_network(0)

    net.eval()

    chkpt = torch.load(CHECKPOINT_PATH)

    net.load_state_dict(chkpt['state_dict'])

Infer Interactions
^^^^^^^^^^^^^^^^^^

Putting everything together, you get:

.. code:: python

    for p1, p2 in sequence_pairs:
        x1 = IntrepppidDataset.static_encode(trunc_len, spp, p1)
        x2 = IntrepppidDataset.static_encode(trunc_len, spp, p2)

        y_hat_logits = net(x1, x2)
        # The forward pass returns logits, so you need to activate with sigmoid
        y_hat = torch.sigmoid(y_hat_logits)

or if you were using the INTREPPPID Data Module

.. code:: python

    for batch in data_module.test_dataloader():
        x1, x2, _, _, _, label = batch

        y_hat_logits = net(x1, x2)
        # The forward pass returns logits, so you need to activate with sigmoid
        y_hat = torch.sigmoid(y_hat_logits)
