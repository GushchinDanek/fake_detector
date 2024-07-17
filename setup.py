import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="fake_news",
    version="0.3",
    install_requires=required,
    python_requires=">=3.8",
    packages=['fake_news'],
    package_dir = {'fake_news':'src/fake_news'}
    
)