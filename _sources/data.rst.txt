Data
====

Pretrained Weights
------------------

You can download the pre-trained weights used in the INTREPPPID manuscript from the `GitHub releases page <https://github.com/Emad-COMBINE-lab/intrepppid/releases>`_.

Precomputed Datasets
--------------------

You can download precomputed datasets from the sources below:

1. `Zenodo <https://doi.org/10.5281/zenodo.10594149>`_ (DOI: 10.5281/zenodo.10594149)
2. `Internet Archive <https://archive.org/details/intrepppid_datasets.tar>`_

All datasets are made available under the `Creative Commons Attribution-ShareAlike 4.0 International <https://creativecommons.org/licenses/by-sa/4.0/legalcode>`_ license.

Dataset Format
--------------

INTREPPPID requires that datasets be prepared specifically in `HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ files.

Each INTREPPPID dataset must have the following hierarchical structure

.. code::

   intrepppid.h5
   ├── orthologs
   ├── sequences
   │
   ├── splits
   │   ├── test
   │   ├── train
   │   └── val
   │
   └── interactions
       ├── c1
       │    ├── c1_train
       │    ├── c1_val
       │    └── c1_test
       │
       ├── c2
       │    ├── c2_train
       │    ├── c2_val
       │    └── c2_test
       │
       └── c3
            ├── c2_train
            ├── c2_val
            └── c2_test

All but one of the "c" folders under "interactions" need be present, so long as that is the dataset you specify in the train step with the ``--c_type`` flag.

Here is the schema for the tables:

.. list-table:: ``orthologs`` schema
   :widths: 25 25 25 50
   :header-rows: 1

   * - Field Name
     - Type
     - Example
     - Description
   * - ``ortholog_group_id``
     - ``Int64``
     - ``1048576``
     - The `OMA <https://omabrowser.org/oma/home/>`_ Group ID of the protein in the ``protein_id`` column
   * - ``protein_id``
     - ``String``
     - ``M7ZLH0``
     - The `UniProt <https://www.uniprot.org/>`_ accession of a protein with OMA Group ID ``ortholog_group_id``

.. list-table:: ``sequences`` schema
   :widths: 25 25 25 50
   :header-rows: 1

   * - Field Name
     - Type
     - Example
     - Description
   * - ``name``
     - ``String``
     - ``Q9NZE8``
     - The `UniProt <https://www.uniprot.org/>`_ accession that corresponds to the amino acid sequence in the ``sequence`` column.
   * - ``sequence``
     - ``String``
     - ``MAASAFAGAVRAASGILRPLNI``...
     - The amino acid sequence indicated by the ``name`` column.

.. list-table:: Schema for all tables under ``interactions``
   :widths: 25 25 25 50
   :header-rows: 1

   * - Field Name
     - Type
     - Example
     - Description
   * - ``protein_id1``
     - ``String``
     - ``Q9BQB4``
     - The `UniProt <https://www.uniprot.org/>`_ accession of the first protein in the interaction pair.
   * - ``protein_id2``
     - ``String``
     - ``Q9NYF0``
     - The `UniProt <https://www.uniprot.org/>`_ accession of the second protein in the interaction pair.
   * - ``omid_protein_id``
     - ``String``
     - ``C1MTX6``
     - The `UniProt <https://www.uniprot.org/>`_ accession of the anchor protein for the orthologous locality loss.
   * - ``omid_id``
     - ``Int64``
     - ``737336``
     - The `OMA <https://omabrowser.org/oma/home/>`_ Group ID of the anchor protein, from which a positive protein can be chose for the orthologous locality loss.
   * - ``label``
     - ``Bool``
     - ``False``
     - Label indicating whether ``protein_id1`` and ``protein_id2`` interact with one another.
