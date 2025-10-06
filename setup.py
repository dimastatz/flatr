from pathlib import Path
from setuptools import setup
from flatr import __version__
from pkg_resources import parse_requirements


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='flatr',
    version=__version__,
    url='https://github.com/dimastatz/flatr',
    author='Dima Statz',
    author_email='dima.statz@gmail.com',
    py_modules=['flatr'],
    python_requires=">=3.9",
    install_requires=[
        str(r)
        for r in parse_requirements(
            Path(__file__).with_name("requirements.txt").open()
        )
    ],
    description='Flatten GitHub Repos into Markdown for LLM-Friendly Code Exploration',
    long_description = long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={'': ['static/*']},
)