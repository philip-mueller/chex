# Training and Evaluation Source Code for Chest X-Ray Explainer (ChEX)
- ChEX main model: ``src/model/chex.py``
- ChEX encoder/decoder/detector components: ``src/model/img_encoder``, ``src/model/txt_encoder``, ``src/model/txt_decoder``, ``src/model/detector``
- ChEX training code: ``src/model/supervisors``
- ChEX evaluation code: ``src/model/eval``

## Configs
- ChEX model configs: ``conf/model/chex_stage1.yaml``, ``conf/model/chex_stage2.yaml``, amd ``conf/model/chex_stage3.yaml``
- ChEX training configs: ``conf/experiment/chex_stage1_train.yaml``, ``conf/experiment/chex_stage2_train.yaml``, ``conf/experiment/chex_stage3_train.yaml``, and ``conf/experiment/pretrain_txtgen.yaml``
- ChEX evaluation task configs: ``conf/task/``
- Prompts used for training and inference: ``conf/prompts/``
- Dataset configs (loading options, class-names, ...): ``conf/dataset/``

## Other Directories and Files:
- ``src/chexzero``: Third-party code for the image and text encoders (ChEX uses CheXZero as the image and text encoder)
- ``src/dataset``: Data-preprocessing and -loading for training and evaluation
- ``src/metrics``: Evaluation metrics for all tasks
- ``src/utils``: Additional utilities
- ``train.py``: Training script (which calls individual supervisors based on the config, see ``conf/experiment/chex_stage*_train.yaml``)
- ``evaluate.py``: Evaluation script (which calls individual evaluators based on the config, see ``conf/task``)
- ``settings.py``: Definitions of paths/environment variables