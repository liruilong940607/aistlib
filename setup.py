import setuptools

INSTALL_REQUIREMENTS = ['numpy', 'absl-py', 'matplotlib']

setuptools.setup(
    name='aistlib',
    url='',
    description='',
    version='0.0.1',
    author='Ruilong Li',
    author_email='ruilongl@usc.edu',
    license='MIT License',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIREMENTS,
)