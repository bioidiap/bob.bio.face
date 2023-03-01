"""The main entry for bob.bio (click-based) scripts.
"""
import click
import pkg_resources

from clapper.click import AliasedGroup
from click_plugins import with_plugins


@with_plugins(pkg_resources.iter_entry_points("bob.bio.face.cli"))
@click.group(cls=AliasedGroup)
def face():
    """Customized Face recognition commands"""
    pass
