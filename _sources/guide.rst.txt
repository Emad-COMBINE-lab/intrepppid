User Guide
==========

Training
--------

The easiest way to start training INTREPPPID is to use the :doc:`CLI <cli>`.

An example of running the training loop with the values used in the INTREPPPID manuscript is as follows:

.. code:: bash

    $ intrepppid train e2e_rnn_triplet DATASET.h5 spm.model 3 100 80 --seed 3927704 --vocab_size 250 --trunc_len 1500 --embedding_size 64 --rnn_num_layers 2 --rnn_dropout_rate 0.3 --variational_dropout false --bi_reduce last --workers 4 --embedding_droprate 0.3 --do_rate 0.3 --log_path logs/e2e_rnn_triplet --beta_classifier 2 --use_projection false --optimizer_type ranger21_xx --lr 1e-2

Checkpoints will be saved in a folder ``logs/e2e_rnn_triplet/model_name/chkpt`` and can be used for inference.

Inference
---------

There are three ways to infer using INTREPPPID:

1. Using PPI.bio
^^^^^^^^^^^^^^^^

The easiest way to infer using INTREPPPID is through the website `https://PPI.bio <https://ppi.bio>`_.

2. Using the CLI
^^^^^^^^^^^^^^^^

The INTREPPPID CLI has an ``infer`` command you can use to infer interaction probabilities.
You can read all the details on the `CLI page <cli.html#infer>`_, but an example of running the command would look like:

.. code:: shell

    intrepppid infer from_csv interactions.csv sequences.fasta weights.ckpt spm.model output.csv

Let's break down all those files we've passed as arguments:

1. interactions.csv: Interactions CSV
2. sequences.fasta: Sequences FASTA
3. weights.ckpt: Weights file
4. spm.model: SentencePiece model file
5. output.csv: Where the inferred interaction probabilities will be written

The interactions CSV must be of the format

.. code::

    INTERACTION ID, PROTEIN ID 1, PROTEIN ID 2

An example of the first few lines of an interactions CSV might be:

.. code::

    1,P08913,P30047
    2,Q71DI3,Q7KZ85
    3,O15498,Q7Z406

The sequence FASTA is just a FASTA file with sequences that correspond to the interactions CSV file.

.. code::

    >P08913
    MFRQEQPLAEGSFAPMGSLQPDAGNASWNG...
    >P30047
    MPYLLISTQIRMEVGPTMVGDEQSDPELMQ...
    >Q71DI3
    MARTKQTARKSTGGKAPRKQLATKAARKSA...

The third component needed is an INTREPPPID weights file. You can either get one from training INTREPPPID from scratch
or `downloading pre-trained weights <data.html#pretrained-weights>`_.

The fourth component is a SentencePiece model file. The easiest way to get one is to just download the one we trained,
which is included in our pre-trained weights as ``spm.model``.

The final component is a CSV with inferred interaction probabilities, which looks like:

.. code::

    1,0.5125845074653625
    2,0.6327105164527893
    3,0.7151195406913757

Note that these interaction probabilities are just examples, they don't correspond to real inferred interaction
probabilities.

3. Using the Python API
^^^^^^^^^^^^^^^^^^^^^^^

**Prepare the Data**

The final way to infer with INTREPPPID is to use the :doc:`Python API <api>`.

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

    for p1, p2 in sequence_pairs:
        x1 = IntrepppidDataset.static_encode(trunc_len, spp, p1)
        x2 = IntrepppidDataset.static_encode(trunc_len, spp, p2)

        x1, x2 = torch.tensor(x1), torch.tensor(x2)

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

**Load the INTREPPPID network**

We must now instantiate the INTREPPPID network and load weights.

If you trained the INTREPPPID with the manuscript defaults, you pass any values to :py:func:`intrepppid.intrepppid_network`.

.. code:: python

    import torch
    from intrepppid import intrepppid_network

    # steps_per_epoch is 0 here because it is not used for inference
    net = intrepppid_network(0)

    net.eval()

    chkpt = torch.load(CHECKPOINT_PATH)

    net.load_state_dict(chkpt['state_dict'])

**Infer Interactions**

Putting everything together, you get:

.. code:: python

    for p1, p2 in sequence_pairs:
        x1 = IntrepppidDataset.static_encode(trunc_len, spp, p1)
        x2 = IntrepppidDataset.static_encode(trunc_len, spp, p2)

        x1, x2 = torch.tensor(x1), torch.tensor(x2)

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
