from setuptools import setup, find_packages

setup(
    name='textsimilarity',
    version='0.0.1',
    description='Text cleaning and similarity ranking.',
    author='Nalani Schumacher',
    url='https://github.com/NalaniKai/TextSimilarity',
    license='MIT',
    packages=find_packages(include=['textsimilarity']),
    install_requires=[
        'pandas',
        'nltk',
        'spacy',
        'profanity_filter',
        'transformers'
    ]
)