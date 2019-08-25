from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cif',
    version='0.0.5',
    description='Composite Indicators Framework for Business Cycle Analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='gpl-3.0',
    packages=find_packages(),
    install_requires=['requests', 'pandas', 'matplotlib', 'statsmodels'],
    python_requires='>=3, <4',
    author='Lenka Vraná',
    author_email='lenka.vrana@gmail.com',
    keywords=['business cycle analysis', 'composite indicators'],
    url='https://github.com/LenkaV/CIF'
)