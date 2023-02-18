from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='datapredictor',
    version='0.0.1',
    description='Prediction data',
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='',
    author='Umidyor',
    author_email='actualnew001@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='predict,data prediction,data predict',
    packages=find_packages(),
    install_requires=['scikit-learn','sklearn','pandas','numpy']
)