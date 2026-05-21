# MIRA-DB

MIRA-DB is a database pipeline for extracting, storing, and comparing
epidemiological ODE models from scientific literature. It is built as a
companion to the [MIRA](https://github.com/gyorilab/mira) framework,
extending it with a structured PostgreSQL backend.


## Overview

MIRA-DB is a structured database and web explorer for epidemiological ODE
models extracted from scientific literature. It provides a PostgreSQL backend
for storing MIRA `TemplateModel` representations along with their source
metadata, and a web application for browsing, searching, and inspecting
models and their grounded components.
 
MIRA-DB serves as the storage and retrieval layer in a broader modeling
ecosystem as it is designed to work alongside tools like MIRA's agentic
extraction pipeline, which populates the database with models derived from
epidemiology publications.

## Resources

- MIRA TemplateModel schema:
  [schema.json](https://github.com/gyorilab/mira/blob/main/mira/metamodel/schema.json)
- Epidemiology Domain Knowledge Graph (DKG):
  [DKG service](http://mira-epi-dkg-lb-dc1e19b273dedaa2.elb.us-east-1.amazonaws.com)
- MIRA documentation: [miramodel.readthedocs.io](https://miramodel.readthedocs.io)

## Architecture

MIRA-DB is organized around a PostgreSQL backend with four core tables that
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
 
### Provenance Chain
 
Each record in the database traces a full lineage from paper to model:
 
```
text_references → text_contents → ode_expressions → mira_template_models
```
 
A single publication (`text_references`) may have multiple extraction
attempts (`text_contents`) using different methods. Each extraction can yield
one ODE expression (`ode_expressions`), and each ODE expression can
produce a grounded MIRA `TemplateModel` (`mira_template_models`).

### Model Similarity Scoring

Pairwise model comparison uses a three-layer scoring system:

1. **Compartment Jaccard similarity** — fuzzy compartment name matching via
   `rapidfuzz`
2. **Term-set Jaccard similarity** — symbolic ODE term comparison with scalar
   stripping via `SymPy`
3. **Tree Edit Distance (TED)** — structural comparison of expression trees
   via the `zss` library


## Installation
 
Requires Python 3.10 or later.
 
The most recent code can be installed directly from GitHub with:
 
```bash
python -m pip install git+https://github.com/gyorilab/miradb.git
```
 
Core dependencies (`flask`, `sqlalchemy>=2`, and `mira`) are installed
automatically.

MIRA-DB requires a running PostgreSQL instance. The database connection can 
be configured by editing [config](db/default_db_config.ini) with your PostgreSQL
server details.