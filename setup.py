from setuptools import setup


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='jax-fcpe',
    description='The JAX implementation of Fast Context-based Pitch Estimation (FCPE)',
    version='0.0.4',
    author='flyingblackshark',
    author_email='aliu2000@outlook.com',
    url='https://github.com/flyingblackshark/jax-fcpe',
    install_requires=['jax', 'flax', 'torch', 'numpy','librosa'],
    packages=['jax_fcpe'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=['License :: OSI Approved :: MIT License'])