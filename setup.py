from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='instrument_recognition_task',
    version='0.0.0',
    description='experiments on the musical instrument recognition task',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(),
    author='Hugo Flores Garcia',
    author_email='hugofloresgarcia@u.northwestern.edu',
    # keywords=['Audio', 'Dataset', 'PyTorch'],
    # url='https://github.com/hugofloresgarcia/instrument-recognition',
    # download_url='https://pypi.org/project/philharmonia-dataset/'
)

install_requires = [
    'torch',
    'pytorch_lightning',
    'numpy',
    'pandas',
    'matplotlib',
    'test-tube',
    'tensorboard==2.2',
    'tqdm', 
    'soundfile',
    'librosa', 
    'sox',  
    'ray[tune]'
    'uncertainty_metrics', 
    'sklearn']

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)