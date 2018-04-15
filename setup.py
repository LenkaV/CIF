from setuptools import setup, find_packages

setup(
    name='cif',
    version='0.0.2',
    description='Composite Indicators Framework for Business Cycle Analysis',
    license='gpl-3.0',
	packages=find_packages(),
	python_requires='>=3, <4',
    author='Lenka Vraná',
    author_email='lenka.vrana@gmail.com',
    keywords=['business cycle analysis', 'composite indicators'],
    url='https://github.com/LenkaV/CIF'
)