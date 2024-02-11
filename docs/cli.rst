Command Line Interface
======================

INTREPPPID has a :abbr:`CLI (Command Line Interface)` which can be used to easily train INTREPPPID.

Train
-----

To train the INTREPPPID model as it was in the manuscript, use the ``train e2e_rnn_triplet`` command:

.. code:: bash

    $ intrepppid train e2e_rnn_triplet DATASET.h5 spm.model 3 100 80 --seed 3927704 --vocab_size 250 --trunc_len 1500 --embedding_size 64 --rnn_num_layers 2 --rnn_dropout_rate 0.3 --variational_dropout false --bi_reduce last --workers 4 --embedding_droprate 0.3 --do_rate 0.3 --log_path logs/e2e_rnn_triplet --beta_classifier 2 --use_projection false --optimizer_type ranger21_xx --lr 1e-2

.. list-table:: INTREPPPID Manuscript Values for ``e2e_rnn_triplet``
   :widths: 25 25 25 50
   :header-rows: 1

   * - Argument/Flag
     - Default
     - Manuscript Value
     - Description
   * - ``PPI_DATASET_PATH``
     - None
     - See Data
     - Path to the PPI dataset. Must be in the INTREPPPID HDF5 format.
   * - ``SENTENCEPIECE_PATH``
     - None
     - See Data
     - Path to the SentencePiece model.
   * - ``C_TYPE``
     - None
     - ``3``
     - Specifies which dataset in the INTREPPPID HDF5 dataset to use by specifying the C-type.
   * - ``NUM_EPOCHS``
     - None
     - ``100``
     - Number of epochs to train the model for.
   * - ``BATCH_SIZE``
     - None
     - ``80``
     - The number of samples to use in the batch.
   * - ``--seed``
     - None
     - ``8675309`` or ``5353456`` or ``3927704`` depending on the experiment.
     - The random seed. If not specified, chosen at random.
   * - ``--vocab_size``
     - ``250``
     - ``250``
     - The number of tokens in the SentencePiece vocabulary.
   * - ``--trunc_len``
     - ``1500``
     - ``1500``
     - Length at which to truncate sequences.
   * - ``--embedding_size``
     - ``64``
     - ``64``
     - The size of embeddings.
   * - ``--rnn_num_layers``
     - ``2``
     - ``2``
     - The number of layers in the AWD-LSTM encoder to use.
   * - ``--rnn_dropout_rate``
     - ``0.3``
     - ``0.3``
     - The dropconnect rate for the AWD-LSTM encoder.
   * - ``--variational_dropout``
     - ``false``
     - ``false``
     - Whether to use variational dropout, as described in the AWD-LSTM manuscript.
   * - ``--bi_reduce``
     - ``last``
     - ``last``
     - Method to reduce the two LSTM embeddings for both directions. Must be one of "concat", "max", "mean", "last".
   * - ``--workers``
     - ``4``
     - ``4``
     - The number of processes to use for the DataLoader.
   * - ``--embedding_droprate``
     - ``0.3``
     - ``0.3``
     - The amount of Embedding Dropout to use (a la AWD-LSTM).
   * - ``--do_rate``
     - ``0.3``
     - ``0.3``
     - The amount of dropout to use in the MLP Classifier.
   * - ``--log_path``
     - ``"./logs/e2e_rnn_triplet"``
     - ``"./logs/e2e_rnn_triplet"``
     - The path to save logs.
   * - ``--encoder_only_steps``
     - ``-1`` (No Steps)
     - ``-1`` (No Steps)
     - The number of steps to only train the encoder and not the classifier.
   * - ``--classifier_warm_up``
     - ``-1`` (No Steps)
     - ``-1`` (No Steps)
     - The number of steps to only train the classifier and not the encoder.
   * - ``--beta_classifier``
     - ``4`` (25% contribution of the classifier loss, 75% contribution of the orthologue loss)
     - ``2`` (50% contribution of the classifier loss, 50% contribution of the orthologue loss)
     - Adjusts the amount of weight to give the PPI Classification loss, relative to the Orthologue Locality loss. The loss becomes (1/β)×(classifier_loss) + [1-(1/β)]×(orthologue_loss).
   * - ``--lr``
     - ``1e-2``
     - ``1e-2``
     - Learning rate to use.
   * - ``--use_projection``
     - ``false``
     - ``false``
     - Whether to use a projection network after the encoder.
   * - ``--checkpoint_path``
     - ``log_path / model_name / "chkpt"``
     - ``log_path / model_name / "chkpt"``
     - The location where checkpoints are to be saved.
   * - ``--optimizer_type``
     - ``ranger21``
     - ``ranger21_xx``
     - The optimizer to use while training. Must be one of ``ranger21``, ``ranger21_xx``, ``adamw``, ``adamw_1cycle``, or ``adamw_cosine``.