from sqlalchemy import func, select
from miradb.db.client import MiraDatabaseClient
from miradb.db.schema import ODEs, TextContent, TextRef


def list_publication_summaries(client: MiraDatabaseClient) -> list[dict]:
    """Return every text_reference with a count of associated ode_expressions.
    
    Parameters
    ----------
    client :
        An instance of MiraDatabaseClient.

    Returns
    -------
    :
        A list of dictionaries, each containing a publication summary.
        The dictionary keys are:
        - pmid: The PubMed ID of the publication.
        - title: The title of the publication.
        - author_list: The list of authors of the publication.
        - pub_year: The year of the publication.
        - model_count: The number of ODE expressions associated with the publication.

    Example
    -------
    [
        {
            "pmid": "1234567890",
            "title": "A study of the effects of caffeine on productivity",
            "author_list": "John Doe, Jane Doe",
            "pub_year": 2020,
            "model_count": 2,
        },
        ...
    ]
    """
    ode_count_sq = (
        select(
            TextContent.text_ref,
            func.count(ODEs.id).label("ode_count"),
        )
        .join(ODEs, ODEs.txt_content_ref == TextContent.id, isouter=True)
        .group_by(TextContent.text_ref)
        .subquery()
    )
    stmt = (
        select(
            TextRef.pmid,
            TextRef.title,
            TextRef.authors,
            TextRef.year,
            func.coalesce(ode_count_sq.c.ode_count, 0).label("model_count"),
        )
        .outerjoin(ode_count_sq, ode_count_sq.c.text_ref == TextRef.id)
        .order_by(func.coalesce(ode_count_sq.c.ode_count, 0).desc())
    )
    rows = client.query(stmt)
    return [
        {
            "pmid": int(row["pmid"]),
            "title": row["title"] or "",
            "author_list": ", ".join(row["authors"]) if row["authors"] else "",
            "pub_year": int(row["year"]),
            "model_count": int(row["model_count"]),
        }
        for row in rows
    ]
