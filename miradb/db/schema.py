import logging

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, CheckConstraint, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON, ARRAY

from datetime import datetime, timezone


logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


Base = declarative_base()

class EpiTable:
    """
    Base class that tags ORM classes as owned by EpiModelManager
    to auto-discover tables via mapper registry.
    """


class TextRef(Base, EpiTable):
    __tablename__ = 'text_references'

    id = Column(Integer, primary_key=True)
    pmid = Column(String, unique=True, nullable=False)
    doi = Column(String, unique=True, nullable=True)
    pmcid = Column(String, unique=True, nullable=True)
    authors = Column(ARRAY(String), nullable=True)
    title = Column(String, nullable=True)
    journal = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    keywords = Column(ARRAY(String), nullable=True)

    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    text_contents = relationship("TextContent", back_populates="text_ref_obj", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TextRef(id='{self.id}', pmid='{self.pmid}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'pmid': self.pmid,
            'doi': self.doi,
            'pmcid': self.pmcid,
            'authors': self.authors,
            'title': self.title,
            'journal': self.journal,
            'year': self.year,
            'keywords': self.keywords,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class ExtractionMethod(Base, EpiTable):
    __tablename__ = 'extraction_method'

    id = Column(Integer, primary_key=True)
    extraction_method = Column(String, nullable=False) # {"mineru_image", "mineru_text", "marker"}
    extraction_method_desc = Column(String, nullable=True)

    text_ref = relationship("TextContent", back_populates="extraction_method_obj", cascade="all, delete-orphan")
    ode_ref = relationship("ODEs", back_populates="extraction_method_obj", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ExtractionMethod(id='{self.id}', extraction_method='{self.extraction_method}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'extraction_method': self.extraction_method,
            'extraction_method_desc': self.extraction_method_desc,
        }


class TextContent(Base, EpiTable):
    __tablename__ = 'text_contents'

    id = Column(Integer, primary_key=True)
    text_ref = Column(Integer, ForeignKey('text_references.id', ondelete="CASCADE"), nullable=False)
    folder_path = Column(String, nullable=False)
    extraction_method_id = Column(Integer, ForeignKey('extraction_method.id'), nullable=False)
    extracted_info_path = Column(String, nullable=False)

    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    text_ref_obj = relationship("TextRef", back_populates="text_contents")
    extraction_method_obj = relationship("ExtractionMethod", back_populates="text_ref")

    odes = relationship("ODEs", back_populates="text_con_obj", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint('text_ref', 'extraction_method_id', name='uq_text_ref_extraction_method'),
    )

    def __repr__(self):
        return f"<TextContent(id='{self.id}', text_ref='{self.text_ref}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'text_ref': self.text_ref,
            'folder_path': self.folder_path,
            'extraction_method_id': self.extraction_method_id,
            'extracted_info_path': self.extracted_info_path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class ODEs(Base, EpiTable):
    __tablename__ = 'ode_expressions'

    id = Column(Integer, primary_key=True)
    txt_content_ref = Column(Integer, ForeignKey('text_contents.id', ondelete="CASCADE") ,nullable=False)
    ode = Column(String, nullable=False)
    corrected_ode = Column(String, nullable=True)
    extraction_method_id = Column(Integer, ForeignKey('extraction_method.id'), nullable=False)

    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    mira_model = relationship("MiraModel", back_populates="ode_obj", cascade="all, delete-orphan")
    extraction_method_obj = relationship("ExtractionMethod", back_populates="ode_ref")

    __table_args__ = (
        CheckConstraint(
            "ode IS NOT NULL OR corrected_ode IS NOT NULL",
            name="ck_at_least_one_not_null_ode"
        ),
        UniqueConstraint('txt_content_ref', name='uq_txt_content_ref'),
    )

    text_con_obj = relationship("TextContent", back_populates="odes")

    def __repr__(self):
        return f"<ODEs(id='{self.id}', ode='{self.ode}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'txt_content_ref': self.txt_content_ref,
            'ode': self.ode,
            'corrected_ode': self.corrected_ode,
            'extraction_method_id': self.extraction_method_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class MiraModel(Base, EpiTable):
    __tablename__ = 'mira_template_models'

    id = Column(Integer, primary_key=True)
    ode_ref = Column(Integer, ForeignKey('ode_expressions.id', ondelete="CASCADE"), nullable=False)
    grounded_concepts = Column(JSON, nullable=False)
    mira_template_model = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    ode_obj = relationship("ODEs", back_populates="mira_model")

    def __repr__(self):
        return f"<MiraModel(id='{self.id}', template_model='{self.mira_template_model}')>"

    __table_args__ = (
    UniqueConstraint('ode_ref', name='uq_ode_ref'),
)

    def to_dict(self):
        return {
            'id': self.id,
            'ode_ref': self.ode_ref,
            'grounded_concepts': self.grounded_concepts,
            'mira_template_model': self.mira_template_model,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
