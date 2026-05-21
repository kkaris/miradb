# MIRA-DB

MIRA-DB is a database of compartmental epidemiology models extracted from
the scientific literature. Each entry is parsed into a structured,
ontology-grounded [MIRA](https://github.com/gyorilab/mira) Template Model,
so that models from different papers can be searched, compared, and reused
under a common representation rather than read off PDF equations by hand.

A public web instance with the current model corpus is available at
[https://epimodels.io](https://epimodels.io).

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
assembled into a MIRA Template Model via a hypergraph algorithm that
recognizes conversion-type processes from term sums on the right-hand
sides. The final Template Model, the intermediate ODE expressions, and
the source publication metadata are stored in the relational schema
described below.

This repository contains the database schema, the manager API used by
the pipeline to write models in, and the Flask web service that powers
the explorer UI. Extraction and grounding code lives in the
[MIRA](https://github.com/gyorilab/mira) repository.

## Schema

Models are stored across four tables that mirror the stages of the
pipeline:

- `text_references`: bibliographic metadata for a publication (PMID, DOI,
  PMCID, authors, title, journal, year, keywords).
- `text_contents`: one row per (publication, extraction method) pair,
  recording where the extracted artifact lives on disk.
- `ode_expressions`: the raw and (optionally) error-corrected symbolic
  ODE strings produced from a `text_contents` row.
- `mira_template_models`: the final grounded MIRA Template Model
  serialized as JSON, plus the concept grounding metadata, linked to its
  source `ode_expressions` row.

Extraction methods are encoded as integer enum values so that adding a
new method does not require a schema migration. The one-to-many chain
from `text_references` down lets multiple extraction methods coexist for
the same paper, which is what makes head-to-head method comparison
possible.

## Web explorer

The Flask app under `miradb/sources/` serves the model explorer at the
`/explorer` route. It supports text search across publication metadata
and grounded concepts, and renders each stored Template Model back to
LaTeX ODE equations using MIRA's `OdeModel`. Run it locally with
`python -m miradb.sources.app`.


## Funding

The development of MIRA-DB is funded under the DARPA ASKEM program, grant number HR00112220036.
