# Running Bulldozer

## Prerequisites

Before running **Bulldozer**, ensure that your input data meets the following requirements:
- The DSM is consistent and free from major artefacts.
- `nodata` value is properly defined in the DSM metadata.
- The Coordinate Reference System (CRS) is correctly specified in the DSM metadata.
- The spatial resolution of your DSM  matches your requirements.

If any of these conditions are not satisfied, please refer to the [Input Data Preprocessing](input_data_preprocessing.md) page before proceeding.

> **Note**  
> The reliability and accuracy of the generated DTM are directly dependent on the quality of the input DSM.

---

## Bulldozer interfaces

**Bulldozer** can be executed using several interfaces, depending on your workflow:

- [Command-Line Interface (CLI)](bulldozer_cli.md)
- [Python API](bulldozer_python_api.md)
- [Docker image](bulldozer_docker.md)

For a complete description of available optional parameters, see the [**Bulldozer** parameters](bulldozer_options.md) page.

After execution, the list and description of output products are available on the [**Bulldozer** Outputs](bulldozer_outputs.md) page.


---

## When to use CLI vs Python API?

| Use case                                  | Recommended interface |
|-------------------------------------------|-----------------------|
| Quick processing                          | CLI                   |
| HPC batch jobs                            | CLI + config file     |
| Interactive analysis                      | Python API            |
| Integration into a larger Python pipeline | Python API            |