"""The main entry for bob.bio (click-based) scripts.
"""
import click
import pkg_resources

from click_plugins import with_plugins

from bob.extension.scripts.click_helper import AliasedGroup


@with_plugins(pkg_resources.iter_entry_points("bob.bio.face.plots.cli"))
@click.group(cls=AliasedGroup)
def plots():
    """Customized Face Recognition plots"""
    pass
