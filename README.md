# MIRA-DB

MIRA-DB is a database of compartmental epidemiology models extracted from
scientific literature. Each entry is parsed into a structured,
ontology-grounded [MIRA](https://github.com/gyorilab/mira) `TemplateModel`,
so that models from different papers can be searched, compared, and reused
under a common representation. It is built as a companion to the MIRA
framework, extending it with a structured PostgreSQL backend.

A public web instance with the current model corpus is available at
[https://epimodels.io](https://epimodels.io).

## Overview

MIRA-DB provides a PostgreSQL backend for storing MIRA `TemplateModel`
representations along with their source metadata, and a web application for
browsing, searching, and inspecting models and their grounded components.

MIRA-DB serves as the storage and retrieval layer in a broader modeling
ecosystem. It is designed to work alongside MIRA's extraction
pipeline, which populates the database with models derived from epidemiology
publications. This repository contains the database schema, the manager API
used by the pipeline to write models in, and the Flask web service that powers
the explorer UI. Extraction and grounding code lives in the
[MIRA](https://github.com/gyorilab/mira) repository.

## Resources

- MIRA `TemplateModel` schema:
  [schema.json](https://github.com/gyorilab/mira/blob/main/mira/metamodel/schema.json)
- Epidemiology Domain Knowledge Graph (DKG):
  [DKG service](http://mira-epi-dkg-lb-dc1e19b273dedaa2.elb.us-east-1.amazonaws.com)
- MIRA documentation: [miramodel.readthedocs.io](https://miramodel.readthedocs.io)

## Pipeline

MIRA-DB is populated by a four-stage pipeline. Publications are first
acquired from PubMed and PubMedCentral using topic and keyword queries.
For each paper, equation content is extracted from the full text via one
or more complementary methods: traversal of MathML/LaTeX tags in the
PubMedCentral XML, structured HTML produced by
[Marker](https://github.com/VikParuchuri/marker), and text or image
output from [MinerU](https://github.com/opendatalab/MinerU). The
extracted equations are then parsed into symbolic form using a large
language model, state variables are grounded against domain ontologies
(IDO, Apollo SV, and others), and the resulting system of ODEs is
assembled into a MIRA `TemplateModel` via a hypergraph algorithm that
recognizes conversion-type processes from term sums on the right-hand
sides. The final `TemplateModel`, the intermediate ODE expressions, and
the source publication metadata are stored in the relational schema
described below.

## Architecture

MIRA-DB is organized around a PostgreSQL backend with five core tables that
track the provenance chain from source publication through to grounded MIRA
model. The schema draws on and adapts patterns from
[EMMAA](https://github.com/gyorilab/emmaa).

| Table | Description |
| --- | --- |
| `text_references` | Bibliographic metadata for source publications (PMID, DOI, PMCID, authors, title, journal, year, keywords) |
| `extraction_method` | Registry of PDF extraction methods used (e.g., `mineru_image`, `mineru_text`, `marker`) |
| `text_contents` | Links a publication to its extracted PDF output, recording the extraction method and file paths |
| `ode_expressions` | Raw and corrected ODE strings parsed from extracted content, linked to a `text_contents` record |
| `mira_template_models` | Grounded MIRA `TemplateModel` JSON and grounded concepts, linked to a source `ode_expressions` record |

### Model provenance

Each record in the database traces a full lineage from paper to model:

```
text_references → text_contents → ode_expressions → mira_template_models
```

A single publication (`text_references`) may have multiple extraction
attempts (`text_contents`) using different methods. Each extraction can yield
one ODE expression (`ode_expressions`), and each ODE expression can
produce a grounded MIRA `TemplateModel` (`mira_template_models`).

Extraction methods are encoded as integer enum values so that adding a new
method does not require a schema migration. The one-to-many chain from
`text_references` down lets multiple extraction methods coexist for the same
paper, enabling head-to-head method comparison.

### Equation extraction benchmarking

Automatically extracted model equations are compared to a gold standard
using three-layer scoring system:

1. **Compartment Jaccard similarity** — fuzzy compartment name matching via
   `rapidfuzz`.
2. **Term-set Jaccard similarity** — symbolic ODE term comparison with scalar
   stripping via `SymPy`.
3. **Tree Edit Distance (TED)** — structural comparison of expression trees
   via the `zss` library. The derived Tree Edit Similarity (TES) is used 
   as a normalized score between 0 and 1.

## Web explorer

A Flask app under `miradb/sources/` serves the model explorer at the
`/explorer` route. It supports text search across publication metadata
and grounded concepts, and renders each stored `TemplateModel` back to
LaTeX ODE equations using MIRA's `OdeModel`. Run it locally with:

```bash
python -m miradb.sources.app
```

<<<<<<< HEAD
## REST API

The web service also exposes a REST API under `/api`, with an OpenAPI
spec at `/api/docs/openapi.json` and interactive documentation at
`/api/docs/swagger`. Models are addressed by PMID and extraction method.

| Endpoint | Description |
| --- | --- |
| `GET /api/models` | Search models by `q`, `pmid`, `method` and `concept` (a grounding CURIE), paginated |
| `GET /api/models/<pmid>` | The highest priority model of a publication |
| `GET /api/models/<pmid>/<method>` | The model extracted by a given method |
| `GET`/`POST /api/download` | All models matching the search filters, as JSON Lines or a zip of SBML or SymPy files |
| `GET /api/publications` | List or search publications |
| `GET /api/publications/<pmid>` | Publication metadata and its models |
| `GET /api/publications/<pmid>/scenarios` | Scenarios and parameters of a publication |

Model endpoints take `format=json|sbml|sympy`, and `best=true` keeps only
the highest priority model per publication. For example:

```bash
curl "https://epimodels.io/api/models?concept=ido:0000514&best=true"
curl -o models.jsonl "https://epimodels.io/api/download"
```

=======
## MCP server

An [MCP](https://modelcontextprotocol.io) server under `miradb/mcp/` lets
LLM agents query MIRA-DB, work with its models through MIRA, and simulate
them. Install it with the `mcp` extra and register it with a client, e.g.
Claude Code:

```bash
python -m pip install -e ".[mcp]"
claude mcp add miradb -- miradb-mcp
```

or run it as a shared service with
`miradb-mcp --transport streamable-http --port 8004`. `--db` selects the
database section of `db_config.ini`; a role with only `SELECT` grants is
recommended.

Models are passed between tools by handle: `db:<id>` for a row of
`mira_template_models`, `ws:<id>` for a model derived during the session.

| Group | Tools |
| --- | --- |
| Database | `db_search_publications`, `db_get_publication`, `db_get_odes`, `db_get_scenarios`, `db_get_scenario`, `db_describe_schema`, `db_run_sql` (read-only, row limit, timeout) |
| MIRA | `mira_get_model`, `mira_get_equations`, `mira_stratify`, `mira_simplify_rate_laws`, `mira_set_values`, `mira_compare_models`, `mira_export_model` (TemplateModel or Petri net AMR), `mira_list_workspace` |
| Simulation | `simulate_model` (optionally applying a DB scenario's parameters and `X_0` initials), `get_simulation` |

Resources: `miradb://schema`, `miradb://model/{model_ref}` and
`miradb://simulation/{sim_id}.csv`.

>>>>>>> 3a1f2d3 (Adding mcp server)
## Docker

A two-container setup (database + app) is defined under [`docker/`](docker/).
See [docker/README.md](docker/README.md) for more information about setting up the 
docker containers from a database dump and running the stack with 
`docker compose`.

## Installation

Requires Python 3.10 or later.

The most recent code can be installed directly from GitHub with:

```bash
python -m pip install git+https://github.com/gyorilab/miradb.git
```

Core dependencies (`flask`, `sqlalchemy>=2`, and `mira`) are installed
automatically.

MIRA-DB requires a running PostgreSQL instance. The database connection can
be configured by editing [config](db/default_db_config.ini) with your
PostgreSQL server details.

## Funding

The development of MIRA-DB is funded under the DARPA ASKEM program, grant
number HR00112220036.
