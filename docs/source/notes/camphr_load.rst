Load model with YAML or JSON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Camphr provides top-level function, :code:`camphr.load`:

.. autofunction:: camphr.load

You can pass the model configuration in format that `omegaconf <https://github.com/omry/omegaconf>`_ can parse, such as yaml or json, for example:

    >>> import camphr
    >>> cfg = """
        lang:
            name: en
        pipeline:
            sentencizer:
                punct_chars: [".", "!"]
            transformers_model:
                trf_name_or_path: xlnet-base-cased
        """
    >>> nlp = camphr.load(cfg)
    >>> print(nlp.pipe_names)
    ['sentencizer', 'transformers_tokenizer', 'transformers_model']

"transformers_tokenizer" is automatically added because it is required to use "transformers_model".

Or you can pass *python dict* to :code:`camphr.load`:

    >>> import camphr
    >>> cfg = {
    >>>     "lang": {"name": "en"},
    >>>     "pipeline": {
    >>>         "sentencizer": {"punct_chars": [".", "!"]},
    >>>         "transformers_model": {"trf_name_or_path": "xlnet-base-cased"},
    >>>     },
    >>> }
    >>> nlp = camphr.load(cfg)
    >>> print(nlp.pipe_names)
    ['sentencizer', 'transformers_tokenizer', 'transformers_model']



