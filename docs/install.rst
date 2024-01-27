Installation
============

From Source
-----------

You can install INTREPPPID from source as follows:

**Step 1.** Clone INTREPPPID from GitHub

.. code:: bash

    $ git clone https://github.com/jszym/intrepppid
    $ cd intreppid

**Step 2.** Create a `Virtual Environment <https://virtualenv.pypa.io/en/latest/>`_ and install the requirements. INTREPPPID is tested on Python 3.10 and CUDA 11.10.

.. code:: bash

    $ python -m virtualenv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt

**Step 3.** Run INTREPPPID

.. code:: bash

    $ python -m intreppid --help

    NAME
        __main__.py - The INTREPPPID CLI

    SYNOPSIS
        __main__.py COMMAND

    DESCRIPTION
        The INTREPPPID CLI

    COMMANDS
        COMMAND is one of the following:

         test

         train

