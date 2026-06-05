# Compatibility facade: CRUD lives in miradb.db.queries; DDL in MiraDatabaseManager.
import logging
import re

from sqlalchemy import inspect, create_engine
from .schema import Base, EpiTable
from .client import MiraDatabaseSessionManager, MiraDatabaseClient
from .config import get_databases
from . import queries

logger = logging.getLogger(__name__)


def get_db(name):
    """Get a db instance based on its name in the config or env."""

    defaults = get_databases()
    db_name, _ = defaults[name]
    m = re.match(r'([^:]+)://.*?/*([\w.]+)', db_name)
    if m is None:
        logger.error("Poorly formed db name: %s" % db_name)
        return
    return MiraModelManager(db_name, label=name)


class MiraDatabaseManager(object):
    """A parent class used to manage sessions within the MIRA-DB database."""

    table_order = [ 'text_references', 'extraction_method', 'text_contents', 'ode_expressions', 'mira_template_models']
    table_parent_class = EpiTable

    def __init__(self, host, label=None):
        self.host = host
        self.label = label
        self.engine = create_engine(host)
        self.tables = {tbl.__tablename__: tbl
                       for tbl in self.table_parent_class.__subclasses__()}
        self.session = None
        return

    def get_session(self):
        """Return a session manager for the database."""
        return MiraDatabaseSessionManager(self.engine)

    def create_tables(self, tables=None):
        """Create the tables from the MIRA-DB database

        Optionally specify `tables` to be created. List may contain either
        table objects or the string names of the tables.
        """

        if tables is not None:
            tables = [tbl.__tablename__ if isinstance(tbl, EpiTable) else tbl
                      for tbl in tables]

        if tables is None:
            tables = set(self.tables.keys())

        else:
            tables = set(tables)

        for tbl_name in self.table_order:
            if tbl_name in tables:
                logger.info(f"Creating {tbl_name} table")
                inspector = inspect(self.engine)
                if not inspector.has_table(self.tables[tbl_name].__tablename__):
                    self.tables[tbl_name].__table__.create(bind=self.engine)
                    logger.debug("Table created!")
                else:
                    logger.warning(f"Table {tbl_name} already exists! "
                                   f"No action taken.")
        return

    def drop_tables(self, tables=None, force=False):
        """Drop the tables from the MIRA-DB database given in `tables`.

        If `tables` is None, all tables will be dropped. Note that if `force`
        is False, a warning prompt will be raised to asking for confirmation,
        as this action will remove all data from that table.
        """
        # Regularize the type of input to table objects.
        if tables is not None:
            tables = [tbl if isinstance(tbl, EpiTable) else self.tables[tbl]
                      for tbl in tables]

        if not force:
            if tables is None:
                msg = ("Do you really want to clear the %s database? [y/N]: "
                       % self.label)
            else:
                msg = "You are going to clear the following tables:\n"
                msg += '\n'.join(['\t-' + tbl.__tablename__ for tbl in tables])
                msg += '\n'
                msg += ("Do you really want to clear these tables from %s? "
                        "[y/N]: " % self.label)

            resp = input(msg)
            if resp != 'y' and resp != 'yes':
                logger.info('Aborting drop.')
                return False

        if tables is None:
            logger.info("Removing all tables...")
            Base.metadata.drop_all(self.engine)
            logger.debug("All tables removed.")
        else:
            for tbl in tables:
                logger.info("Removing %s..." % tbl.__tablename__)
                if tbl.__table__.exists(self.engine):
                    tbl.__table__.drop(self.engine)
                    logger.debug("Table removed.")
                else:
                    logger.debug("Table doesn't exist.")
        return True

    def table_exists(self, table_name):
        return table_name in inspect(self.engine).get_table_names()


class MiraModelManager(MiraDatabaseManager):
    """
    Domain manager for all epidemiological modeling artifact tables.
    """

    def __init__(self, host, label=None):
        super().__init__(host, label)
        self._client = MiraDatabaseClient(host, label=label, engine=self.engine)

    # ── TextRef ────────────────────────────────────────────────────────────────

    def add_text_ref(self, pmid: str, pmcid: str = None, doi: str = None, authors: str = None, title: str = None, journal: str = None, year: int = None, keywords: list = None):
        return queries.add_text_ref(
            self._client, pmid, pmcid=pmcid, doi=doi, authors=authors,
            title=title, journal=journal, year=year, keywords=keywords,
        )

    def update_text_ref(self, pmid: str, pmcid: str = None, doi: str = None, authors: str = None, title: str = None, journal: str = None, year: int = None, keywords: list = None):
        return queries.update_text_ref(
            self._client, pmid, pmcid=pmcid, doi=doi, authors=authors,
            title=title, journal=journal, year=year, keywords=keywords,
        )

    def get_text_ref(self, pmid: str):
        return queries.get_text_ref(self._client, pmid)

    def get_all_text_refs(self):
        return queries.get_all_text_refs(self._client)

    def remove_text_ref(self, pmid: str):
        return queries.remove_text_ref(self._client, pmid)

    # ── TextContent ──────────────────────────────────────────────────────────────

    def add_text_content(self, text_ref: int, folder_path: str, extraction_method_id: int, extracted_info_path: str):
        return queries.add_text_content(
            self._client, text_ref, folder_path, extraction_method_id, extracted_info_path,
        )

    def update_text_content(self, id: int, folder_path: str = None, extraction_method_id: int = None, extracted_info_path: str = None):
        return queries.update_text_content(
            self._client, id, folder_path=folder_path,
            extraction_method_id=extraction_method_id,
            extracted_info_path=extracted_info_path,
        )

    def get_text_content(self, text_ref: int):
        return queries.get_text_content(self._client, text_ref)

    def get_all_text_contents(self):
        return queries.get_all_text_contents(self._client)

    def remove_text_content(self, text_ref: int):
        return queries.remove_text_content(self._client, text_ref)

    # ── ODEs ───────────────────────────────────────────────────────────────────

    def add_odes(self, txt_content_ref: int, extraction_method_id: int, ode: str, corrected_ode: str = None):
        return queries.add_odes(
            self._client, txt_content_ref, extraction_method_id, ode, corrected_ode,
        )

    def update_odes(self, txt_content_ref: int, extraction_method_id: int, ode: str = None, corrected_ode: str = None):
        return queries.update_odes(
            self._client, txt_content_ref, extraction_method_id, ode, corrected_ode,
        )

    def get_odes(self, txt_content_ref: int):
        return queries.get_odes(self._client, txt_content_ref)

    def get_all_odes(self):
        return queries.get_all_odes(self._client)

    def remove_odes(self, txt_content_ref: int):
        return queries.remove_odes(self._client, txt_content_ref)

    # ── MiraModel ──────────────────────────────────────────────────────────────

    def add_tm(self, ode_ref: int, grounded_concepts: dict, mira_template_model: dict = None):
        return queries.add_tm(self._client, ode_ref, grounded_concepts, mira_template_model)

    def update_tm(self, ode_ref: int, grounded_concepts: dict = None, mira_template_model: dict = None):
        return queries.update_tm(self._client, ode_ref, grounded_concepts, mira_template_model)

    def get_tm(self, ode_ref: int):
        return queries.get_tm(self._client, ode_ref)

    def get_all_tms(self):
        return queries.get_all_tms(self._client)

    def remove_tm(self, ode_ref: int):
        return queries.remove_tm(self._client, ode_ref)
