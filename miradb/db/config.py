import logging
from os import path, makedirs, environ
from shutil import copyfile
from configparser import ConfigParser

HERE = path.dirname(path.abspath(__file__))

DB_CONFIG_DIR = path.expanduser('~/.config/mira')
DB_CONFIG_PATH = path.join(DB_CONFIG_DIR, 'db_config.ini')
DEFAULT_DB_CONFIG_PATH = path.join(HERE, 'default_db_config.ini')

DB_STR_FMT = "{prefix}://{username}{password}{host}{port}/{name}"
ENV_PREFIX = 'MIRADB'


logger = logging.getLogger(__name__)


CONFIG_EXISTS = True
if not path.exists(DB_CONFIG_DIR):
    try:
        makedirs(DB_CONFIG_DIR)
    except Exception as e:
        CONFIG_EXISTS = False
        logger.warning("Unable to create config dir: %s" % e)


if not path.exists(DB_CONFIG_PATH) and CONFIG_EXISTS:
    try:
        copyfile(DEFAULT_DB_CONFIG_PATH, DB_CONFIG_PATH)
    except Exception as e:
        CONFIG_EXISTS = False
        logger.warning("Unable to copy config file into config dir: %s" % e)


DATABASES = None


def _format_db_url(def_dict):
    if def_dict['host']:
        def_dict['host'] = '@' + def_dict['host']
    def_dict['prefix'] = def_dict['dialect']
    if def_dict['driver']:
        def_dict['prefix'] += '+' + def_dict['driver']
    if def_dict['port']:
        def_dict['port'] = ':' + def_dict['port']
    if def_dict['password']:
        def_dict['password'] = ':' + def_dict['password']
    return DB_STR_FMT.format(**def_dict)


def _get_db_with_type(db_url):
    parts = db_url.split(';', 1)
    url = parts[0]
    db_type = parts[1] if len(parts) > 1 else 'query'
    return url, db_type


def get_databases(force_update=False, include_config=True):
    global DATABASES
    if DATABASES is None or force_update:
        DATABASES = {}
        if CONFIG_EXISTS and include_config:
            parser = ConfigParser()
            parser.read(DB_CONFIG_PATH)
            for db_name in parser.sections():
                def_dict = {k: parser.get(db_name, k)
                            for k in parser.options(db_name)}
                DATABASES[db_name] = (_format_db_url(def_dict),
                                      def_dict.get('type', 'query'))

        db_host = environ.get(f'{ENV_PREFIX}_DB_HOST')
        if db_host:
            component_dict = {
                'dialect': environ.get(f'{ENV_PREFIX}_DB_DIALECT', 'postgresql'),
                'driver': environ.get(f'{ENV_PREFIX}_DB_DRIVER', 'psycopg'),
                'username': environ.get(f'{ENV_PREFIX}_DB_USER', 'postgres'),
                'password': environ.get(f'{ENV_PREFIX}_DB_PASSWORD', 'miradb'),
                'host': db_host,
                'port': environ.get(f'{ENV_PREFIX}_DB_PORT', '5432'),
                'name': environ.get(f'{ENV_PREFIX}_DB_NAME', 'mira_db'),
            }
            DATABASES['primary'] = (
                _format_db_url(component_dict),
                environ.get(f'{ENV_PREFIX}_DB_TYPE', 'query')
            )

        DATABASES.update({k[len(ENV_PREFIX):].lstrip('_').lower():
                          _get_db_with_type(v)
                          for k, v in environ.items()
                          if k.startswith(ENV_PREFIX)
                          and not k.startswith(f'{ENV_PREFIX}_DB_')})
    return DATABASES