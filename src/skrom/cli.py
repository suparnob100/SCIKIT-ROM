"""Command-line interface for scikit-rom.

TL;DR
-----
This module exposes commands for creating starter problem folders from bundled templates.

Notes
-----
The commands are small wrappers around template discovery and filesystem copying.
"""

import shutil
from pathlib import Path
import click
import importlib.resources as pkg_res

@click.group()
def cli():
    """Command-line tool for skrom.
    
    TL;DR
    -----
    Command-line tool for skrom.
    """
    pass

@cli.command("create-problem")
@click.argument("name")
def create_problem(name):
    """Create a new problem_<name> folder with starter code.
    
    TL;DR
    -----
    Create a new problem_<name> folder with starter code.
    """
    target = Path(f"problem_{name}")
    if target.exists():
        click.echo(f"[!] {target} already exists.")
        return

    # 1) Copy the entire template tree into problem_<name>
    src = pkg_res.files("skrom") / "templates" / "problem_template"
    shutil.copytree(src, target)

    # 2) Rename the notebook stub at the template root
    nb = target / "problem.ipynb"
    if nb.exists():
        new_nb = target / f"problem_{name}.ipynb"
        nb.rename(new_nb)
    else:
        click.echo(f"[!] No notebook stub found at {nb}; skipping.")

    # 3) Patch the {{problem_name}} placeholder in problem_def.py
    pd_file = target / "problem_def.py"
    text = pd_file.read_text()
    pd_file.write_text(text.replace("{{problem_name}}", name))

    click.echo(f"✔ Created problem folder at: {target}")
