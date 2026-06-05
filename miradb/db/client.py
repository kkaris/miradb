import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, inspect, text
from miradb.db.config import get_databases
from miradb.db.schema import Base, EpiTable


logger = logging.getLogger(__name__)


class MiraDatabaseError(Exception):
    pass


class MiraDatabaseSessionManager(object):
    """Database session context manager

    Parameters
    ----------
    engine : Engine
        The engine of the database.
    """

    def __init__(self, engine):
        logger.debug(f"Grabbing a session to {engine.url}...")
        DBSession = sessionmaker(bind=engine)
        logger.debug("Session grabbed.")
        self.session = DBSession()
        if self.session is None:
            raise MiraDatabaseError("Could not acquire session.")

    def __enter__(self):
        return self.session

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type:
            logger.exception(exception_value)
            logger.info("Got exception: rolling back.")
            self.session.rollback()
        else:
            logger.debug("Committing changes...")
            self.session.commit()

        # Close the session.
        self.session.close()


class MiraDatabaseClient:
    """Database client

    Parameters
    ----------
    url : str
        The URL of the database.
    label : str, optional
        The label of the database e.g., "primary".
    """

    def __init__(self, url: str, *, label: str | None = None, engine=None):
        self.url = url
        self.label = label
        self.engine = engine or create_engine(url)
        self.table_mapping = {
            tbl.__tablename__: tbl for tbl in EpiTable.__subclasses__()
        }

    def session(self):
        return MiraDatabaseSessionManager(self.engine)

    def query(self, statement, params=None):
        """Execute a SELECT statement and return all rows as mappings."""
        with self.session() as sess:
            return sess.execute(statement, params).mappings().all()

    def query_one(self, statement, params=None):
        """Execute a SELECT statement and return the first row as a mapping."""
        with self.session() as sess:
            return sess.execute(statement, params).mappings().first()

    def mutate(self, fn):
        """Run a function that modifies the database and commit on success."""
        with self.session() as sess:
            return fn(sess)  # session manager already commits

    def query_sql(self, sql: str, params: dict | None = None):
        """Execute a SQL statement and return all rows as mappings."""
        with self.session() as sess:
            return sess.execute(text(sql), params or {}).mappings().all()

    def create_tables(self, tables: list[EpiTable | str] = None):
        """Create the tables in the MIRA-DB database

        Parameters
        ----------
        tables :
            A list of tables to create. If None, all tables will be created.
        """

        if tables is None:
            tables = set(self.table_mapping.keys())
        else:
            tables = {
                tbl.__tablename__ if isinstance(tbl, EpiTable) else tbl
                for tbl in tables
            }

        for tbl_name in TABLE_ORDER:
            if tbl_name in tables:
                logger.info(f"Creating {tbl_name} table")
                inspector = inspect(self.engine)
                if not inspector.has_table(
                    self.table_mapping[tbl_name].__tablename__
                ):
                    self.table_mapping[tbl_name].__table__.create(
                        bind=self.engine
                    )
                    logger.debug("Table created!")
                else:
                    logger.warning(
                        f"Table {tbl_name} already exists! " f"No action taken."
                    )
        return

    def drop_tables(self, tables=None, force=False):
        """Drop the tables from the MIRA-DB database given in `tables`.

        If `tables` is None, all tables will be dropped. Note that if `force`
        is False, a warning prompt will be raised to asking for confirmation,
        as this action will remove all data from that table.
        """
        # Regularize the type of input to table objects.
        if tables is not None:
            tables = [
                tbl if isinstance(tbl, EpiTable) else self.table_mapping[tbl]
                for tbl in tables
            ]

        if not force:
            if tables is None:
                msg = (
                    "Do you really want to clear the %s database? [y/N]: "
                    % self.label
                )
            else:
                msg = "You are going to clear the following tables:\n"
                msg += "\n".join(["\t-" + tbl.__tablename__ for tbl in tables])
                msg += "\n"
                msg += (
                    "Do you really want to clear these tables from %s? "
                    "[y/N]: " % self.label
                )

            resp = input(msg)
            if resp != "y" and resp != "yes":
                logger.info("Aborting drop.")
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


def get_client(name: str = "primary") -> MiraDatabaseClient:
    url, _ = get_databases()[name]
    return MiraDatabaseClient(url, label=name)


TABLE_ORDER = [
    "text_references",
    "extraction_method",
    "text_contents",
    "ode_expressions",
    "mira_template_models",
]
