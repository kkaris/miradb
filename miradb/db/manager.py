import logging
import re

from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from sqlalchemy import create_engine
from .schema import Base, EpiTable, TextRef, TextContent, ODEs, MiraModel
from .session import MiraDatabaseSessionManager
from .config import get_databases

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
        return MiraDatabaseSessionManager(self.host, self.engine)
    
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

    # ── TextRef ────────────────────────────────────────────────────────────────

    def add_text_ref(self, pmid: str, pmcid: str = None, doi: str = None, authors: str = None, title: str = None, journal: str = None, year: int = None, keywords: list = None):
        """Add a paper to the database, returning the new paper's ID.

        Parameters
        ----------
        pmid : str
            The PubMed ID of the paper.
        pmcid : str, optional
            The PubMed Central ID of the paper.
        doi : str, optional
            The DOI of the paper.
        authors : str, optional
            The authors of the paper.
        title : str, optional
            The title of the paper.
        journal : str, optional 
            The journal of the paper.
        year : int, optional
            The publication year of the paper.
        keywords : list, optional
            A list of keywords associated with the paper.

        Returns
        -------
        int
            The ID of the paper that was added.
        """
        try:
            with self.get_session() as sess:
                p = TextRef(pmid=pmid, pmcid=pmcid, doi=doi, authors=authors, title=title, journal=journal, year=year, keywords=keywords)
                sess.add(p)
                sess.flush()
                sess.refresh(p)
                paper_id = p.id
                logger.info("Registered paper '%s'.", pmid)
        except IntegrityError as e:
            logger.warning(f"A paper with pmid {pmid} already exists.")
            return
        return paper_id 
    
    def update_text_ref(self, pmid: str, pmcid: str = None, doi: str = None, authors: str = None, title: str = None, journal: str = None, year: int = None, keywords: list = None):
        """Update a paper's identifiers in the database.

        Parameters
        ----------
        pmid : str
            The PubMed ID of the paper to update.
        pmcid : str, optional
            The new PubMed Central ID of the paper.
        doi : str, optional
            The new DOI of the paper.
        authors : str, optional
            The authors of the paper.
        title : str, optional
            The title of the paper.
        journal : str, optional 
            The journal of the paper.
        year : int, optional
            The publication year of the paper.
        keywords : list, optional
            A list of keywords associated with the paper.

        Returns
        -------
        bool
            True if the paper was found and updated, False if the paper was not found.
        """
        with self.get_session() as sess:
            p = sess.query(TextRef).filter_by(pmid=pmid).first()
            if not p:
                logger.warning("Paper '%s' not found.", pmid)
                return False
            if pmcid is not None:
                p.pmcid = pmcid
            if doi is not None:
                p.doi = doi
            if authors is not None:
                p.authors = authors
            if title is not None:
                p.title = title 
            if journal is not None:
                p.journal = journal
            if year is not None:
                p.year = year
            if keywords is not None:
                p.keywords = keywords
            logger.info("Updated paper '%s'.", pmid)
        return True
    
    def get_text_ref(self, pmid: str):
        """
        Retrieve a paper's text identifiers from the MIRA-DB database by its PubMed ID.
        Parameters
        ----------  
        pmid : str
            The PubMed ID of the paper to retrieve.
        Returns
        -------
        dict or None
            A dictionary containing the paper's information if found, or None if not found.
            Dictionary keys: 'id', 'pmid', 'doi', 'pmcid', 'created_at', 'updated_at', 'authors', 'title', 'journal', 'year', 'keywords'
        """
        with self.get_session() as sess:
            paper = sess.query(TextRef).filter_by(pmid=pmid).first()
            return paper.to_dict() if paper else None

    def get_all_text_refs(self):
        """
        Retrieve all papers' text identifiers from the MIRA-DB database.
        Returns
        -------
        list of dict
            A list of dictionaries, each containing a paper's information.
            Dictionary keys: 'id', 'pmid', 'doi', 'pmcid', 'created_at', 'updated_at', 'authors', 'title', 'journal', 'year', 'keywords'
        """
        with self.get_session() as sess:
            papers = sess.query(TextRef).order_by(TextRef.created_at.desc()).all()
            return [p.to_dict() for p in papers]

    def remove_text_ref(self, pmid: str):
        """
        Delete a paper and all FK-linked rows (CASCADE).
        Parameters
        ----------
        pmid : str
            The PubMed ID of the paper to delete.
        Returns
        -------
        bool 
            True if the paper was found and deleted, False if the paper was not found.
        """
        with self.get_session() as sess:
            p = sess.query(TextRef).filter_by(pmid=pmid).first()
            if not p:
                logger.warning("Paper '%s' not found.", pmid)
                return False
            sess.delete(p)
            logger.info("Deleted paper '%s' and all linked rows.", pmid)
        return True

    # ── Paper Source ────────────────────────────────────────────────────────────────

    def add_text_content(self, text_ref: int, folder_path: str, extraction_method_id: int, extracted_info_path: str):
        """
        Add a paper's source locations to the database, returning the new row's identifier.
        Parameters
        ----------  
        text_ref : int
            The ID of the text context (from TextContent) that this source information is linked to.
        folder_path : str
            The relativefile path to the folder containing the paper's source files (e.g., PDFs, images).
        extraction_method_id : int, optional
            The ID of the extraction method used to extract the ODE string (0 = MinerU, 1 = UniMERNet, 2 = marker).
        extracted_info_path : str, optional
            The relative file path to the extracted information for the paper.
        Returns
        -------
        int
            The ID of the paper source entry that was added.
        """
        try:
            with self.get_session() as sess:
                row = TextContent(text_ref=text_ref, folder_path=folder_path, extraction_method_id=extraction_method_id, extracted_info_path=extracted_info_path)
                sess.add(row)
                sess.flush()
                sess.refresh(row)
                source_id = row.id
                logger.info("Registered paper source '%s'.", text_ref)
        except IntegrityError as e:
            logger.warning(f"A paper source with text_ref {text_ref} already exists.")
            return
        return source_id
    
    def update_text_content(self, id: int, folder_path: str = None, extraction_method_id: int = None, extracted_info_path: str = None):
        """
        Update a paper source's information in the database.

        Parameters
        ----------
        id : int
            The ID of the text context (from TextContent) that this source information is linked to.
        folder_path : str, optional
            The realtive file path to the folder containing the paper's source files (e.g., PDFs, images).
        extraction_method_id : int, optional
            The ID of the extraction method used to extract the ODE string (0 = MinerU, 1 = UniMERNet, 2 = marker).
        extracted_info_path : str, optional
            The relative file path to the extracted information for the paper.

        Returns
        -------
        bool
            True if the paper source was found and updated, False if the paper source was not found.
        """
        with self.get_session() as sess:
            p_source = sess.query(TextContent).filter_by(id=id).first()
            if not p_source:
                logger.warning("Paper source '%s' not found.", id)
                return False
            if folder_path is not None:
                p_source.folder_path = folder_path
            if extraction_method_id is not None:
                p_source.extraction_method_id = extraction_method_id
            if extracted_info_path is not None:
                p_source.extracted_info_path = extracted_info_path
            logger.info("Updated paper source '%s'.", id)
        return True
    
    def get_text_content(self, text_ref: int):
        """
        Retrieve a paper's source information from the MIRA-DB database by the paper's ID.

        Parameters
        ----------  
        text_ref : int
            The ID of the text context (from TextContent) that this source information is linked to.
        Returns
        -------
        dict or None
            A dictionary containing the paper source information if found, or None if not found.
            Dictionary keys: 'source_id', 'text_ref', 'folder_path', 'extraction_method_id', 'extracted_info_path', 'created_at', 'updated_at'
        """
        with self.get_session() as sess:
            p_source = sess.query(TextContent).filter_by(text_ref=text_ref).all()
            return [p.to_dict() for p in p_source] if p_source else None

    def get_all_text_contents(self):
        """
        Retrieve all papers' source information from the MIRA-DB database.

        Returns
        -------
        list of dict
            A list of dictionaries, each containing a paper source's information.
            Dictionary keys: 'source_id', 'text_ref', 'folder_path', 'extraction_method', 'extracted_info_path', 'created_at', 'updated_at'
        """
        with self.get_session() as sess:
            p_sources = sess.query(TextContent).order_by(TextContent.created_at.desc()).all()
            return [p.to_dict() for p in p_sources]
        
    def remove_text_content(self, text_ref: int):
        """
        Delete a text content and all FK-linked rows (CASCADE).
        
        Parameters
        ----------
        text_ref : int
            The ID of the text context (from TextContent) that this source information is linked to.

        Returns
        -------
        bool            
            True if the paper source was found and deleted, False if the paper source was not found.
        """
        with self.get_session() as sess:
            p = sess.query(TextContent).filter_by(text_ref=text_ref).all()
            if not p:
                logger.warning("Paper source '%s' not found.", text_ref)
                return False
            for item in p:
                sess.delete(item)
            logger.info("Deleted paper source '%s' and all linked rows.", text_ref)
        return True
    
    # ── ODE Equations ─────────────────────────────────────────────────────────

    def add_odes(self, txt_content_ref: int, extraction_method_id: int, ode: str, corrected_ode: str= None):
        """
        Add a paper's ODE equations extracted by MinerU to the database, returning the new row's identifier.

        Parameters
        ----------  
        txt_context_ref : int
            The ID of the text context (from TextContent) that this ODE information is linked to.
        ode : str
            The ODE string extracted by MinerU for this paper.
        corrected_ode : str, optional
            A corrected version of the ODE string, if it exists. Can be the same as `ode` if no correction was needed.
        extraction_method_id : int
            The ID of the extraction method used to extract the ODE string (0 = MinerU, 1 = UniMERNet, 2 = marker).

        Returns
        -------
        int
            The ID of the ODE entry that was added.
        """
        try:
            with self.get_session() as sess:
                p = ODEs(txt_content_ref=txt_content_ref, ode=ode, corrected_ode=corrected_ode, extraction_method_id=extraction_method_id)
                sess.add(p)
                sess.flush()
                sess.refresh(p)
                ode_id = p.id
                logger.info("Registered ODEs for txt_content_ref '%s'.", txt_content_ref)
        except IntegrityError as e:
            if 'ode' in str(e.orig):
                logger.warning(f"ODE already exists for txt_content_ref {txt_content_ref}, skipping.")
            else:
                logger.warning(f"An ODE entry for txt_content_ref {txt_content_ref} already exists.")
            return
        return ode_id
    
    def update_odes(self, txt_content_ref: int, extraction_method_id: int, ode: str = None, corrected_ode: str = None):
        """
        Update a paper's ODE equations in the database.

        Parameters
        ----------  
        txt_context_ref : int
            The ID of the text context (from TextContent) that this ODE information is linked to.
        ode : str, optional
            The new ODE string extracted by MinerU for this paper.
        corrected_ode : str, optional
            A new corrected version of the ODE string, if it exists. Can be the same as `ode` if no correction was needed.
        extraction_method_id : int
            The ID of the extraction method used to extract the ODE string (0 = MinerU, 1 = UniMERNet, 2 = marker).

        Returns
        -------
        bool
            True if the ODE entry was found and updated, False if the ODE entry was not found.
        """
        with self.get_session() as sess:
            odes = sess.query(ODEs).filter_by(txt_content_ref=txt_content_ref).first()
            if not odes:
                logger.warning("ODEs for txt_content_ref '%s' not found.", txt_content_ref)
                return False
            if ode is not None:
                odes.ode = ode
            if corrected_ode is not None:
                odes.corrected_ode = corrected_ode
            if extraction_method_id is not None:
                odes.extraction_method_id = extraction_method_id
            logger.info("Updated ODEs for txt_content_ref '%s'.", txt_content_ref)
        return True
    
    def get_odes(self, txt_content_ref: int):
        """
        Retrieve a paper's ODE equations from the MIRA-DB database by the txt_content_ref.

        Parameters
        ----------  
        txt_content_ref : int
            The ID of the text content (from TextContent) that this ODE information is linked to.
        Returns
        -------
        dict or None
            A dictionary containing the ODE information if found, or None if not found.
            Dictionary keys: 'ode_id', 'txt_content_ref', 'ode', 'corrected_ode', 'extraction_method_id', 'created_at', 'updated_at'
        """
        with self.get_session() as sess:
            odes = sess.query(ODEs).filter_by(txt_content_ref=txt_content_ref).first()
            return odes.to_dict() if odes else None

    def get_all_odes(self):
        """
        Retrieve all papers' ODE equations from the MIRA-DB database.

        Returns
        -------
        list of dict
            A list of dictionaries, each containing a paper's ODE information.
            Dictionary keys: 'ode_id', 'txt_context_ref', 'ode', 'corrected_ode', 'extraction_method', 'created_at', 'updated_at'
        """
        with self.get_session() as sess:
            odes = sess.query(ODEs).order_by(ODEs.created_at.desc()).all()
            return [o.to_dict() for o in odes]  
        
    def remove_odes(self, txt_content_ref: int):
        """
        Delete ODEs for a paper and all FK-linked rows (CASCADE).
        
        Parameters
        ----------
        txt_context_ref : int
            The ID of the text context (from TextContent) that this ODE information is linked to.  

        Returns
        -------
        bool            
            True if the ODE entry was found and deleted, False if the ODE entry was not found.
        """
        with self.get_session() as sess:
            p = sess.query(ODEs).filter_by(txt_content_ref=txt_content_ref).first()
            if not p:
                logger.warning("ODEs for txt_content_ref '%s' not found.", txt_content_ref)
                return False
            sess.delete(p)
            logger.info("Deleted ODEs for txt_content_ref '%s' and all linked rows.", txt_content_ref)
        return True
    
    # ── TemplateModels ────────────────────────────────────────────────────────

    def add_tm(self, ode_ref: int, grounded_concepts: dict, mira_template_model: dict = None):
        """
        Add a MIRA template model extracted from a paper to the database, returning the new row's identifier.

        Parameters
        ----------  
        paper_ref : int
            The ID of the paper (from PaperID) that this template model information is linked to.
        grounded_concepts : dict
            A dictionary json convertible) containing the grounded concepts extracted.
        mira_template_model : dict, optional
            A dictionary (json convertible) containing the MIRA template model extracted from the paper.

        Returns
        -------
        int
            The ID of the template model entry that was added.
        """
        try:
            with self.get_session() as sess:
                p = MiraModel(ode_ref=ode_ref, grounded_concepts=grounded_concepts, mira_template_model=mira_template_model)
                sess.add(p)
                sess.flush()
                sess.refresh(p)
                model_id = p.id
                logger.info("Registered MiraModel for ode_ref '%s'.", ode_ref)
        except IntegrityError as e:
            logger.warning(f"A MiraModel with ode_ref {ode_ref} already exists.")
            return
        return model_id
    
    def update_tm(self, ode_ref: int, grounded_concepts: dict = None, mira_template_model: dict = None):
        """
        Updates a MIRA template model's information in the database.

        Parameters
        ----------
        paper_ref : int
            The ID of the paper (from PaperID) that this template model information is linked to.
        grounded_concepts : dict, optional
            A dictionary (json convertible) containing the grounded concepts extracted from the paper.
        mira_template_model : dict, optional
            A dictionary (json convertible) containing the MIRA template model extracted from the paper.

        Returns
        -------
        bool
            True if the template model entry was found and updated, False if the template model entry was not found.
        """
        with self.get_session() as sess:
            tm = sess.query(MiraModel).filter_by(ode_ref=ode_ref).first()
            if not tm:
                logger.warning("MiraModel for ode_ref '%s' not found.", ode_ref)
                return False
            if grounded_concepts is not None:
                tm.grounded_concepts = grounded_concepts
            if mira_template_model is not None:
                tm.mira_template_model = mira_template_model
            logger.info("Updated MiraModel for ode_ref '%s'.", ode_ref)
        return True
    
    def get_tm(self, ode_ref: int):
        """
        Retrieve an ODE's MIRA template model information from the MIRA-DB database by the ODE's ID.

        Parameters
        ----------
        ode_ref : int
            The ID of the ODE (from ODEs) that this template model information is linked to.

        Returns
        -------
        dict or None
            A dictionary containing the template model information if found, or None if not found.
            Dictionary keys: 'model_id', 'ode_ref', 'grounded_concepts', 'mira_template_model', 'created_at', 'updated_at'
        """
        with self.get_session() as sess:
            tm = sess.query(MiraModel).filter_by(ode_ref=ode_ref).first()
            return tm.to_dict() if tm else None

    def get_all_tms(self):
        """
        Retrieve all papers' MIRA template model information from the MIRA-DB database.

        Returns
        -------
        list of dict
            A list of dictionaries, each containing a paper's MIRA template model information.
            Dictionary keys: 'model_id', 'paper_ref', 'grounded_concepts', 'mira_template_model', 'created_at', 'updated_at'
        """
        with self.get_session() as sess:
            tms = sess.query(MiraModel).order_by(MiraModel.created_at.desc()).all()
            return [tm.to_dict() for tm in tms]
        
    def remove_tm(self, ode_ref: int):
        """
        Delete an ODE and all FK-linked rows (CASCADE).
        
        Parameters
        ----------
        ode_ref : int
            The ID of the ODE (from ODEs) that this template model information is linked to.
            
        Returns
        -------
        bool
            True if the MIRA template model was deleted, False if not found.
        """
        with self.get_session() as sess:
            p = sess.query(MiraModel).filter_by(ode_ref=ode_ref).first()
            if not p:
                logger.warning("MiraModel for ode_ref '%s' not found.", ode_ref)
                return False
            sess.delete(p)
            logger.info("Deleted MiraModel for ode_ref '%s' and all linked rows.", ode_ref)
        return True