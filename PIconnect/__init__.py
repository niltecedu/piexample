""" PIconnect
    Connector to the OSISoft PI and PI-AF databases.
"""
# pragma pylint: disable=unused-import
from datapipelines.pidata.PIconnect.AFSDK import AF, AF_SDK_VERSION
from datapipelines.pidata.PIconnect.config import PIConfig
from datapipelines.pidata.PIconnect.PI import PIServer
from datapipelines.pidata.PIconnect.PIAF import PIAFDatabase

# pragma pylint: enable=unused-import

__version__ = "0.9.1"
__sdk_version = tuple(int(x) for x in AF.PISystems().Version.split("."))
