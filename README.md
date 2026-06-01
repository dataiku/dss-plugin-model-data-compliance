# Model Data Compliance

⚠️ Starting with DSS version 14 this plugin is considered "deprecated".

This plugin helps compare incoming data against reference data in order to detect whether new data still looks like the data used for training.

## Why this plugin is deprecated

Recent DSS versions provide native MLOps capabilities for model evaluation and drift analysis. These built-in features are the recommended path for monitoring deployed models and checking whether scoring data drifts away from reference or training data.

See the DSS documentation:

- https://doc.dataiku.com/dss/latest/mlops/model-evaluations/index.html
- https://doc.dataiku.com/dss/latest/mlops/drift-analysis/index.html

## Scope of the plugin

This plugin provides DSS components to monitor input data:

- a custom metric to compute compliance metrics given a reference dataset or a saved model
- a custom check to validate new data given a reference dataset or a saved model


## Installation and requirements

Please see our [official plugin page](https://www.dataiku.com/product/plugins/model-data-compliance/) for installation.

## Changelog

**Version 1.0.0 (2020-08)**

* Initial release
* Custom metric to compute compliance metrics
* Custom check to validate new data based on compliance metrics

You can log feature requests or issues on our [dedicated Github repository](https://github.com/dataiku/dss-plugin-model-data-compliance/issues).

## License

This plugin is:

   Copyright (c) 2020 Dataiku SAS
   Licensed under the [MIT License](LICENSE).
